import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt


class nnUNetLogger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False, reconstruction: bool = False):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
        }
        self.verbose = verbose
        self.reconstruction = reconstruction
        self.my_fantastic_logging.update({'train_seg_losses': list(),
                                        'train_recon_losses': list(),
                                        'val_seg_losses': list(),
                                        'val_recon_losses': list(),
                                        'ema_union': list(),
                                        'ema_ssim': list(),
                                        'ema_psnr': list(),
                                        'mean_ssim': list(),
                                        'mean_psnr': list(),
                                        'mean_union': list()})
        # shut up, this logging is great

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        ema_keys = ['mean_fg_dice', 'mean_psnr', 'mean_ssim', 'mean_union']
        if key in ema_keys:
            ema_key = key.replace('mean', 'ema')
            new_ema_val = self.my_fantastic_logging[ema_key][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging[ema_key]) > 0 else value
            self.log(ema_key, new_ema_val, epoch)

    def plot_progress_png(self, output_folder):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)",
                 linewidth=4)
        if self.reconstruction:
            ax.plot(x_values, self.my_fantastic_logging['train_seg_losses'][:epoch + 1], color='m', ls='-', label="tr_seg", linewidth=4)
            ax.plot(x_values, self.my_fantastic_logging['train_recon_losses'][:epoch + 1], color='k', ls='-', label="tr_rec", linewidth=4)
            ax.plot(x_values, self.my_fantastic_logging['val_seg_losses'][:epoch + 1], color='y', ls='-', label="val_seg", linewidth=4)
            ax.plot(x_values, self.my_fantastic_logging['val_recon_losses'][:epoch + 1], color='c', ls='-', label="val_rec", linewidth=4)
            ax2.plot(x_values, self.my_fantastic_logging['ema_union'][:epoch + 1], color='r', ls='dotted', label="pseudo union",
                    linewidth=3)
            ax2.plot(x_values, self.my_fantastic_logging['ema_ssim'][:epoch + 1], color='m', ls='-', label="pseudo ssim",
                    linewidth=4)
            ax2.plot(x_values, self.my_fantastic_logging['ema_psnr'][:epoch + 1], color='k', ls='-', label="pseudo psnr",
                    linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[1]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
