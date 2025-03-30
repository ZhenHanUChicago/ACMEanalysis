from .binCalculator import binCalculator
from .binCutter import binCutter


import pickle
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import mpld3
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.ioff()


class binVisualizer:

    def __init__(self, parameter_file_path=None, binresult_file_path=None, bincut_file_path = None, bincutresult_file_path = None, figure_folder_path = None):
        self.binresult_file_path = binresult_file_path
        self.binresult = binVisualizer._load_results(binresult_file_path)
        if self.binresult.swap_xy is True:
            self.binresult.x_mask, self.binresult.y_mask = self.binresult.y_mask, self.binresult.x_mask

        self.parameter_file_path = parameter_file_path
        self.parameter = binCalculator.Parameters(self)
        self.parameter._load_parameters_from_json(self.parameter_file_path)

        self.bincutresult_file_path = bincutresult_file_path
        self.bincutresult = binVisualizer._load_results(bincutresult_file_path)

        self.bincutrule = binCutter.binCutRules()
        self.bincutrule._load_bincut_from_json(bincut_file_path)
        self.figures = {} # In the format of {'name': (fig, axes)}
        self._inspect_yield()
        self._inspect_sipm_noise(self.binresult.name)
        self._inspect_cutting(self.binresult.name)
        self._inspect_binning(inspect_begin_ms= 7, filestring  = self.binresult.name)
        self.figure_folder_path = figure_folder_path
        self.save_figures()
    
    @staticmethod
    def _load_results(result_file_path):
        """Load the pickled results from the file."""
        try:
            with open(result_file_path, 'rb') as f:
                binresult = pickle.load(f)
            return binresult
        except Exception as e:
            print(f"Error loading results: {e}")
            return None

    def __plot_photon_counts(matrix, filestring="", ylim = 8E6):
        # Get the dimensions of the matrix
        rows, cols = matrix.shape

        # Flatten the matrix
        flat_data = matrix.flatten()

        # Create the index list
        index_list = np.arange(len(flat_data))
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(index_list, flat_data, marker='o', linestyle='', color='b', markersize=1.5)

        # Add vertical dashed lines to separate every 'cols' data points
        for i in range(0, rows+1):
            ax.axvline(x=i * cols - 0.5, color='k', linestyle='--', linewidth=0.5)

        # Set labels and limits
        ax.set_xlabel('Shot #', fontsize=14)
        ax.set_ylabel('Photon counts per shot', fontsize=14)
        ax.set_ylim(0, ylim)


        # Set title and font size
        ax.set_title('Photon counts vs Shot # for file: ' + filestring, fontsize=16)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        # Enable the main grid for y-ticks
        ax.grid(True, which='major', axis='y')

        # Get current y-ticks and define new minor grid locations at 1/10 of the distance between major ticks
        y_ticks = ax.get_yticks()
        minor_ticks = np.linspace(y_ticks[0], y_ticks[-1], len(y_ticks) * 10-1)

        # Add horizontal grid lines for the minor ticks
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5)
                # Return the figure object
        return fig, ax

    def _inspect_yield(self):
        if self.parameter.trace_already_summed is False:
            ylim = 8E6
        else:
            ylim = 8E6 * 25
        fig, ax = binVisualizer.__plot_photon_counts(self.binresult.shot_yield, self.binresult.name, ylim = ylim)
        self.figures['shot_yield'] = (fig, ax)

    def _inspect_sipm_noise(self, filestring=""):
        average_begin = self.bincutresult.grand_left
        average_end = self.bincutresult.grand_right
        # Copy data and background to ensure no in-place modification
        data_copy = self.binresult.red_chi_square
        background_copy = self.binresult.N0

        # Determine dimensions from data
        _, _, num_plots, num_points = data_copy.shape

        # Average the data and background along the first two dimensions (16 and 25)
        averaged_data = data_copy.mean(axis=(0, 1))
        averaged_background = background_copy.mean(axis=(0, 1))
        normalized_averaged_background = (averaged_background - averaged_background.min()) / (averaged_background.max() - averaged_background.min()) *3

        # Calculate the mean and 1-sigma range before averaging
        flattened_data = data_copy.reshape(-1, num_plots, num_points)
        mean_flattened = flattened_data.mean(axis=0)
        std_flattened = flattened_data.std(axis=0)

        # Set up the plot with 8 vertically stacked subplots
        fig, axs = plt.subplots(num_plots, 1, figsize=(8, 8), sharex=True)

        # Define colors for the scatter plots
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

        x_data = np.arange(num_points)

        # Plot each of the sets of data
        for i in range(num_plots):
            y_data = averaged_data[i, :]
            
            # Normalize the background to the range of the data y limits
            y_background_normalized = normalized_averaged_background[i, :]
            
            axs[i].fill_between(x_data, 0, y_background_normalized, color='black', alpha=0.1)
            axs[i].scatter(x_data, y_data, color=colors[i], label=f'SIPM {i}, $\\chi^2_{{red [within\\ group]}} = {y_data[average_begin:average_end].mean():.2f}$', s=1.5)
            axs[i].set_ylabel('$\chi^2_{red}$')
            axs[i].legend(loc='upper left')
            
            # Calculate the mean value in the specified range
            mean_range = y_data[average_begin:average_end].mean()
            axs[i].axhline(mean_range, color='black', linestyle='--', linewidth=1)
            
            # Plot the 1-sigma band
            mean_y = mean_flattened[i, :]
            std_y = std_flattened[i, :]
            axs[i].fill_between(x_data, mean_y - std_y, mean_y + std_y, color=colors[i], alpha=0.5, label='1-Sigma Band')
            
            # Plot vertical lines at average_begin and average_end
            axs[i].axvline(average_begin, color='black', linestyle='--', linewidth=1)
            axs[i].axvline(average_end, color='black', linestyle='--', linewidth=1)
            
            # Set y-limits
            axs[i].set_ylim(0, 3)

        # Labeling the plot
        axs[-1].set_xlabel('Group Index')
        fig.suptitle('Excess Noise for Asymmetry in one group, every shot\n'+filestring, fontsize=16)

        # Show plot
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.figures['sipm noise'] = (fig, axs)
        return fig, axs
    
    def _inspect_cutting(self, filestring=""):
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 6))

        # Step plot for the frac_mask (black)
        ax1.step(range(len(self.bincutresult.frac_mask)), self.bincutresult.frac_mask, where='mid', color='black', linewidth = 2)
        ax1.set_title(f"Fractional and Absolute Cut Masks, frac threshold = {self.bincutrule.frac_threshold} \n"+ "abs threshold (all sipm 25 shots single group) >" + f"{self.bincutrule.absgrouptrace_threshold}" + 'photon' +'\n' +filestring, fontsize=20)
        ax1.set_xlabel("Group Index", fontsize=20)
        
        # Step plot for the abs_mask (blue)
        ax1.step(range(len(self.bincutresult.abs_mask)), self.bincutresult.abs_mask, where='mid', color='blue', linewidth = 3)

        # Step plot for the grand_mask (red)
        ax1.step(range(len(self.bincutresult.grand_mask)), self.bincutresult.grand_mask, where='mid', color='red',linestyle='--', linewidth = 1)

        # Create a second axis for the background plot
        ax2 = ax1.twinx()

        # Data for the background plot
        background_data = self.binresult.N0.mean(axis=(0, 1, 2))
        normalized_background = background_data / max(background_data)  # Normalizing

        # Fill plot for the background
        ax2.fill_between(range(len(background_data)), normalized_background, color='grey', alpha=0.3)

        # Adding thin dashed vertical lines at each integer
        for i in range(len(self.bincutresult.frac_mask)):
            ax1.axvline(x=i, color='black', linestyle='dashed', linewidth=0.1)

        # Setting the limits and labels for the second y-axis
        ax2.set_ylim(0, 1)
        
        # Adding legends for the fractional, absolute, and grand masks
        ax1.legend([
            f"Fractional Cut Mask [{self.bincutresult.frac_left},{self.bincutresult.frac_right})",
            f"Absolute Cut Mask [{self.bincutresult.abs_left},{self.bincutresult.abs_right})",
            f"Grand Cut Mask [{self.bincutresult.grand_left},{self.bincutresult.grand_right})"
        ], loc='upper left', fontsize=15)
        
        self.figures['cutting'] = (fig, (ax1, ax2))
        fig.tight_layout()
        return fig, (ax1,ax2)
    
    def _inspect_binning(self, inspect_begin_ms=7, filestring = ""):
        # Updating bin_offset and bin_size references
        while self.parameter.bin_offset < 0:
            self.parameter.bin_offset += self.parameter.bin_size
        while self.parameter.bin_offset >= self.parameter.bin_size:
            self.parameter.bin_offset -= self.parameter.bin_size

        inspect_begin_index = int(inspect_begin_ms * 1000000 // 80) - self.parameter.bin_offset
        inspect_begin_bin_index = inspect_begin_index // self.parameter.bin_size
        inspect_end_bin_index = inspect_begin_bin_index + 2

        fig = plt.figure(figsize=(12, 7))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, :])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])

        # Updated the data source for averaged_data
        averaged_data = self.binresult.bin_pair_inspection_data.reshape(-1)

        # Color Filling Plot
        blacklist = []
        redlist = []
        bluelist = []
        x = []
        y = []
        for i in range(inspect_begin_index, inspect_begin_index + 2 * self.parameter.bin_size):
            l = i % self.parameter.bin_size
            yp = averaged_data[i]
            xp = (i + self.parameter.bin_offset) * 80 / 1000000
            y.append(yp)
            x.append(xp)
            if self.binresult.x_mask[l] == 1:
                bluelist.append(1)
                redlist.append(0)
                blacklist.append(0)
                rect = patches.Rectangle((xp - 80 / 1000000 / 2, 0), 80 / 1000000, yp, linewidth=0, edgecolor='none', facecolor='blue', alpha=0.5)
                ax1.add_patch(rect)
            if self.binresult.y_mask[l] == 1:
                redlist.append(1)
                bluelist.append(0)
                blacklist.append(0)
                rect = patches.Rectangle((xp - 80 / 1000000 / 2, 0), 80 / 1000000, yp, linewidth=0, edgecolor='none', facecolor='red', alpha=0.5)
                ax1.add_patch(rect)
            if self.binresult.x_mask[l] == 0 and self.binresult.y_mask[l] == 0:
                redlist.append(0)
                bluelist.append(0)
                blacklist.append(1)
        ax1.plot(x, y, color='black')
        x = np.array(x)
        y = np.array(y)
        redlist = np.array(redlist)
        bluelist = np.array(bluelist)
        blacklist = np.array(blacklist)

        # Updated the data source for inspect_data
        inspect_data = self.binresult.bin_pair_inspection_data[inspect_begin_bin_index, :].reshape(-1)

        # Unnormalized Plot for showing power difference
        inspect_x = inspect_data[0:self.parameter.bin_size // 2]
        inspect_y = inspect_data[self.parameter.bin_size // 2:self.parameter.bin_size]
        max_x = inspect_x.max()
        max_y = inspect_y.max()
        normalized_x = inspect_x / max_x
        normalized_y = inspect_y / max_y
        ax4.plot(normalized_x, alpha=0.7, c='blue')
        ax4.plot(normalized_y, alpha=0.7, c='red')
        ax5.plot(inspect_x, alpha=0.7, c='blue')
        ax5.plot(inspect_y, alpha=0.7, c='red')

        # Replace ba._find_crossings with the double underscore helper function
        def __find_crossings(arr, threshold=0.5):
            descending_crossings = []  # For transitions from >0.5 to <0.5
            ascending_crossings = []  # For transitions from <0.5 to >0.5

            for i in range(len(arr) - 1):
                if arr[i] > threshold and arr[i + 1] < threshold:
                    # Descending crossing
                    slope = arr[i + 1] - arr[i]
                    intercept = arr[i] - slope * i
                    crossing_idx = (threshold - intercept) / slope
                    descending_crossings.append(crossing_idx)
                elif arr[i] <= threshold and arr[i + 1] > threshold:
                    # Ascending crossing
                    slope = arr[i + 1] - arr[i]
                    intercept = arr[i] - slope * i
                    crossing_idx = (threshold - intercept) / slope
                    ascending_crossings.append(crossing_idx)

            return descending_crossings, ascending_crossings

        try:
            x_d, x_a = __find_crossings(normalized_x, threshold=0.5)
            y_d, y_a = __find_crossings(normalized_y, threshold=0.5)
            ax4.scatter(x_d, [0.5] * len(x_d), color='blue')
            ax4.scatter(x_a, [0.5] * len(x_a), color='blue')
            ax4.scatter(y_d, [0.5] * len(y_d), color='red')
            ax4.scatter(y_a, [0.5] * len(y_a), color='red')
            # print(x_d, x_a, y_d, y_a)
            try:
                ax4.set_title("Bin misalignment X-Y={0:.1f} ns".format(((x_d[0] - y_d[0]) + (x_a[0] - y_a[0])) / 2 * 80), fontsize=24)
            except:
                pass
            fig.suptitle('X-Y polarization comparison reports\n' + filestring +'\n Bin Offset = ' + str(self.parameter.bin_offset), fontsize=30)
            ax1.set_title('X-Y polarization comparison, Blue = X, Red = Y', fontsize=18)
        finally:
            for ax in [ax1, ax4, ax5]:
                ax.grid(True)
                ax.set_xlabel('Sample time index, 1 sample = {0}ns'.format(80), fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=12)
            try:
                ax5.set_title("x_d:{0:.1f}, x_a:{1:.1f}, y_d:{2:.1f}, y_a:{3:.1f}".format(x_d[0], x_a[0], y_d[0], y_a[0]), fontsize=14)
            except:
                pass
            ax1.set_xlabel('ms')
            fig.tight_layout()
            self.figures['binning'] = (fig, (ax1, ax4, ax5))
        return (fig, (ax1,ax4, ax5))

    def save_figures(self):
        for name, (fig, _) in self.figures.items():
            fig.savefig(f"{self.figure_folder_path}/{name}"+"_"+self.binresult.name+".png")

    def close_all_figures(self):
        for name, (fig, _) in self.figures.items():
            plt.close(fig)