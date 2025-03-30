from .binCalculator import binCalculator
from .binCutter import binCutter
from .blockCalculator import blockCalculator
from .blockCutter import blockCutter


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


class blockVisualizer:

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

    def __init__(self, blockresult_file_path, figure_folder_path, blockcutresult_file_path, visualizeN = True, visualizeF = False, visualizeblockparity = True, visualizeSipm = True, visualizeDegenTrace = True, visualizeenvlopphi = False):
        if blockcutresult_file_path is not None:
            self.blockcutresult = self._load_results(blockcutresult_file_path)
            if self.blockcutresult.does_this_block_stay is False:
                return None
        self.blockresult = self._load_results(blockresult_file_path)
        self.file_and_blind_string = "\n" + self.blockresult.block_string + "\n" +self.blockresult.blinded.blind_id
        self.figure_folder_path = figure_folder_path
        self.sipm_color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
        self.figures = {}
        if visualizeSipm:
            self.visualize_summary_vs_sipm()
        if visualizeF:
            self.visualize_F_vs_group()
        if visualizeN:
            self.visualize_numberparity_vs_group()
        if visualizeblockparity:
            self.visualize_blockparity_vs_group()
        if visualizeDegenTrace:
            self.visualize_red_chi_square_trace_shot()
        if visualizeenvlopphi:
            self.visualize_phi_envlope()
        self.save_figures()
        self.close_all_figures()

    def _propagate_error_bar(A, dA2, axis_to_take_average, nan = False):
        if nan:
            if isinstance(axis_to_take_average, int):
                axis_to_take_average = (axis_to_take_average,)

            selected_dimension = [A.shape[i] for i in axis_to_take_average]

            if np.prod(selected_dimension) == 1:
                A_mean = A
                dA_mean_square_scaled = dA2
                red_chi_square = np.full(A.shape, np.nan)
                dA_mean_square_unscaled = dA2
                return A_mean, dA_mean_square_unscaled, dA_mean_square_scaled, red_chi_square

            weights = 1 / dA2
            A_mean = np.nansum(A * weights, axis=axis_to_take_average, keepdims=True) / np.nansum(weights, axis=axis_to_take_average, keepdims=True)

            dA_mean_square_unscaled = 1 / np.nansum(weights, axis=axis_to_take_average, keepdims=True)

            total_number_of_points = np.nanprod([A.shape[axis] for axis in axis_to_take_average])
            residuals = (A - A_mean)**2
            red_chi_square = np.nansum(residuals / dA2, axis=axis_to_take_average, keepdims=True) / (total_number_of_points - 1)

            dA_mean_square_scaled = dA_mean_square_unscaled * red_chi_square

            return A_mean, dA_mean_square_unscaled, dA_mean_square_scaled, red_chi_square
        
        else:

            if isinstance(axis_to_take_average, int):
                axis_to_take_average = (axis_to_take_average,)

            selected_dimension = [A.shape[i] for i in axis_to_take_average]

            if np.prod(selected_dimension) == 1:
                A_mean = A
                dA_mean_square_scaled = dA2
                red_chi_square = np.full(A.shape, np.nan)
                dA_mean_square_unscaled = dA2
                return A_mean, dA_mean_square_unscaled, dA_mean_square_scaled, red_chi_square

            weights = 1 / dA2
            A_mean = np.sum(A * weights, axis=axis_to_take_average, keepdims=True) / np.sum(weights, axis=axis_to_take_average, keepdims=True)

            dA_mean_square_unscaled = 1 / np.sum(weights, axis=axis_to_take_average, keepdims=True)

            total_number_of_points = np.prod([A.shape[axis] for axis in axis_to_take_average])
            residuals = (A - A_mean)**2
            red_chi_square = np.sum(residuals / dA2, axis=axis_to_take_average, keepdims=True) / (total_number_of_points - 1)

            dA_mean_square_scaled = dA_mean_square_unscaled * red_chi_square

            return A_mean, dA_mean_square_unscaled, dA_mean_square_scaled, red_chi_square
        
    def _visualize_summary_vs_sipm(A, dA2, A_mean, dA2_mean, parity_labels, red_chi_square, grand_title, yunit, colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']):
        # Determine if we have 1x8 or 8x8 matrices
        dim = A.ndim
        if dim == 2:
            channels = A.shape[1]
        else:
            channels = A.shape[0]

        # Create the figure and axes
        if dim == 2:
            fig, axes = plt.subplots(4, 2, figsize=(10, 16))
            axes = axes.flatten()  # Flatten the 2D array of axes to easily index
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            axes = [ax]  # Wrap single axis in a list for uniform handling

        num_plots = A.shape[0] if dim == 2 else 1   
        # Iterate over each row of A and dA2 for plotting
        for i in range(num_plots):
            ax = axes[i]
            x = range(channels)  # X values are just range(8)

            # Plot the ith row of A and dA2 as error bars
            for j in range(channels):
                if dim == 2:
                    ax.errorbar(x[j], A[i, j], yerr=np.sqrt(dA2[i, j]), fmt='o', color=colors[j], capsize=5)
                else:
                    ax.errorbar(x[j], A[j], yerr=np.sqrt(dA2[j]), fmt='o', color=colors[j], capsize=5)

            # Plot the A_mean and dA2_mean with thicker error bars in black, including thicker horizontal caps
            ax.errorbar(3.5, A_mean[i], yerr=np.sqrt(dA2_mean[i]), fmt='o', color='black', capsize=10, elinewidth=3, lw=3, label="Mean")

            if dim == 2:
            # Set the title: first line from parity_labels, second line is the chi-square value formatted to 2 decimal places
                ax.set_title(f"{parity_labels[i]}\n$\\chi^2_{{red}}(SiPM) = {red_chi_square[i]:.2f}$", fontsize=20)
            else:
                ax.set_title(f"{parity_labels[0]}\n$\\chi^2_{{red}}(SiPM) = {red_chi_square:.2f}$", fontsize=20)

            # Add y-axis label with the yunit
            ax.set_ylabel(f"{yunit}", fontsize=12)

            ax.grid(True, linestyle='--')
            # Add legend
            ax.legend(loc='upper right', fontsize=10)

        # Set the grand title for the figure
        if num_plots == 8:
            fig.suptitle(grand_title, fontsize=56)
            fig.tight_layout(rect=[0, 0, 1, 0.99])  # Adjust layout to prevent overlap
        else:
            ax.set_title(f"{parity_labels[0]}\n$\\chi^2_{{red}}(SiPM) = {red_chi_square:.2f}$", fontsize=20)
            fig.suptitle(grand_title, fontsize=36)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

        return fig, axes
        
    def visualize_summary_vs_sipm(self):
        A = self.blockresult.blinded.result_summary['omega_summary'].squeeze()
        dA2 = self.blockresult.blinded.result_summary['domega2_summary'].squeeze()
        A_mean, dA2_mean, _ , red_chi_square = blockVisualizer._propagate_error_bar(A, dA2, 1)
        red_chi_square = red_chi_square.squeeze()
        parity_labels = self.blockresult.parity_labels
        grand_title = "$\omega$"
        fig, axes = blockVisualizer._visualize_summary_vs_sipm(A, dA2, A_mean, dA2_mean, parity_labels, red_chi_square, grand_title, 'rad/s')
        self.figures['omegaSummary'] = (fig, axes)

        A = self.blockresult.blinded.result_summary['C_summary'].squeeze()
        dA2 = self.blockresult.blinded.result_summary['dC2_summary'].squeeze()
        A_mean, dA2_mean, _ , red_chi_square = blockVisualizer._propagate_error_bar(A, dA2, 1)
        red_chi_square = red_chi_square.squeeze()
        parity_labels = self.blockresult.parity_labels
        grand_title = "$C$"
        fig, axes = blockVisualizer._visualize_summary_vs_sipm(A, dA2, A_mean, dA2_mean, parity_labels, red_chi_square, grand_title, '')
        self.figures['CSummary'] = (fig, axes)

        A = self.blockresult.blinded.result_summary['tau_summary'].squeeze()
        dA2 = self.blockresult.blinded.result_summary['dtau2_summary'].squeeze()
        A_mean, dA2_mean, _ , red_chi_square = blockVisualizer._propagate_error_bar(A, dA2, 0)
        red_chi_square = red_chi_square.squeeze()
        parity_labels = self.blockresult.parity_labels
        grand_title = "$\\tau$"
        fig, axes = blockVisualizer._visualize_summary_vs_sipm(A, dA2, A_mean, dA2_mean, parity_labels, red_chi_square, grand_title, 's')
        self.figures['tauSummary'] = (fig, axes)

        A = self.blockresult.blinded.result_summary['phi_summary'].squeeze()
        dA2 = self.blockresult.blinded.result_summary['dphi2_summary'].squeeze()
        A_mean, dA2_mean, _ , red_chi_square = blockVisualizer._propagate_error_bar(A, dA2, 1)
        red_chi_square = red_chi_square.squeeze()
        parity_labels = self.blockresult.parity_labels
        grand_title = "$\\phi$"
        fig, axes = blockVisualizer._visualize_summary_vs_sipm(A, dA2, A_mean, dA2_mean, parity_labels, red_chi_square, grand_title, 'rad')
        self.figures['phiSummary'] = (fig, axes)

    def _visualize_red_chi_square_trace_shot(A, begin, end, title):
        try:
            # Average A over all axes except for the last axis
            averaged_A = np.mean(A, axis=tuple(range(A.ndim - 1)))

            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(6, 4))

            # Scatter plot of the averaged_A
            x = np.arange(averaged_A.shape[-1])  # X values based on the last axis length
            ax.scatter(x, averaged_A, color='blue', label='')

            # Plot dashed line at y=1
            ax.axhline(1, color='red', linestyle='--', linewidth=2, label='y = 1')

            # Calculate the mean of averaged_A[begin:end]
            mean_value = np.mean(averaged_A[begin:end])
            
            # Plot a solid line for the mean value over the specified range
            ax.axhline(mean_value, color='green', linestyle='-', linewidth=2, label=f'y = {mean_value:.2f} (mean)')

            # Shade the regions where x < begin and x > end
            ax.axvspan(0, begin, color='gray', alpha=0.3)
            ax.axvspan(end, len(x), color='gray', alpha=0.3)

            # Append chi-square_red (trace shot)=x.xx to the title
            chi_square_red = averaged_A[begin:end].mean()
            ax.set_title(f"{title}\n$\\chi^2_{{red}}(trace\\ shot) = {chi_square_red:.2f}$", fontsize=20)

            # Set labels and grid
            ax.set_xlabel('Group #', fontsize=14)
            ax.grid(True, linestyle='--')

            # Add legend
            ax.legend(loc='lower right', fontsize=10)
            ax.set_ylim(0, np.max(averaged_A[begin:end]))
            # Adjust layout to prevent overlap
            fig.tight_layout()

            return fig, ax
        except Exception as e:
            return fig, ax
    
    def visualize_red_chi_square_trace_shot(self):
        try:
            fig, ax = blockVisualizer._visualize_red_chi_square_trace_shot(self.blockresult.red_chi_square_trace_shot_A, self.blockresult.blockcut_left, self.blockresult.blockcut_right, 'Asymmetry: Degenerate traces and shots')
            self.figures['redChiSquareTraceShotA'] = (fig, ax)
        except:
            try:
                plt.close(fig)
            except:
                pass

    def _visualize_blockparity_vs_group(A, dA2, A_mean, dA2_mean, A_mean_mean, begin, end, title, yunit, parity_labels, red_chi_square, colors=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray'], ylim = None):
        # Determine the number of subplots (8 or 1 based on shape[0] of A)
        num_plots = A.shape[0]

        # Create the figure and subplots
        if num_plots == 8:
            fig, axes = plt.subplots(4, 2, figsize=(12, 16))
            axes = axes.flatten()
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            axes = [ax]  # Ensure consistency with a single plot wrapped in a list

        # Iterate over each plot (either 1 or 8 depending on num_plots)
        for i in range(num_plots):
            ax = axes[i]
            x = np.arange(A.shape[-1])  # X values based on the last axis length

            # Plot the j lines (no error bars) for A[i,0,0,j,:]
            for j in range(A.shape[3]):
                ax.plot(x, A[i, 0, 0, j, :], color=colors[j], label=f'Line {j+1}', lw = 0.5, alpha = 0.1)

            # Plot the A_mean and dA2_mean with error bars (black, thick with caps)
            mean_line = A_mean[i, 0, 0, 0, :]
            error_bars = np.sqrt(dA2_mean[i, 0, 0, 0, :])
            ax.errorbar(x, mean_line, yerr=error_bars, fmt='.', color='black', capsize=2, elinewidth=1, lw=1, label='Mean')
            if ylim is None or i == 0 or i == 3:
                # Set y-limits based on the range [begin:end]
                y_min = np.min(A_mean[i, 0, 0, 0, begin:end]) - np.max(np.sqrt(dA2_mean[i, 0, 0, 0, begin:end]))
                y_max = np.max(A_mean[i, 0, 0, 0, begin:end]) + np.max(np.sqrt(dA2_mean[i, 0, 0, 0, begin:end]))
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_ylim(*ylim)

            # Shade the regions where x < begin and x > end
            ax.axvspan(0, begin, color='gray', alpha=0.3)
            ax.axvspan(end, len(x), color='gray', alpha=0.3)

            # Plot dashed line at y=0
            ax.axhline(0, color='red', linestyle='--', linewidth=2, label='y = 0')

            # Plot dashed line for the mean of A_mean[i,0,0,0,begin:end]
            mean_value = np.mean(A_mean[i, 0, 0, 0, begin:end])
            ax.axhline(mean_value, color='green', linestyle='--', linewidth=2, label=f'Mean = {mean_value:.2f}')

            # Append chi-square_red (trace shot)=x.xx to the title
            chi_square_red_value = red_chi_square[i, 0, 0, 0, 0]
            ax.set_title(f"{parity_labels[i]}\n"+f"{A_mean_mean[i,0,0,0,0]:.3g}"+yunit+"\n"+f"$\\chi^2_{{red}}(group) = {chi_square_red_value:.2f}$", fontsize=20)

            # Set y-axis label with the yunit
            ax.set_ylabel(f"{yunit}", fontsize=12)

            # Set x-axis label
            ax.set_xlabel('Group', fontsize=14)

            # Enable grid
            ax.grid(True, linestyle='--')


        # Set the grand title for the figure
        fig.suptitle(title, fontsize=36)

        # Adjust layout to prevent overlap
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        return fig, axes
    
    def visualize_blockparity_vs_group(self):
        A = self.blockresult.blinded.result['omega']
        dA2 = self.blockresult.blinded.result['domega2']
        A_mean, dA2_mean, _ , _ = blockVisualizer._propagate_error_bar(A, dA2, -2)
        A_mean_mean, _, _, red_chi_square = blockVisualizer._propagate_error_bar(A_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], dA2_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], -1)
        parity_labels = self.blockresult.parity_labels
        grand_title = "$\omega$" + self.file_and_blind_string
        fig, axes = blockVisualizer._visualize_blockparity_vs_group(A, dA2, A_mean, dA2_mean, A_mean_mean, self.blockresult.blockcut_left, self.blockresult.blockcut_right, grand_title, 'rad/s', parity_labels, red_chi_square, ylim = (-0.3,0.3))
        self.figures['omega'] = (fig, axes) 

        A = self.blockresult.blinded.result['phi']
        dA2 = self.blockresult.blinded.result['dphi2']
        A_mean, dA2_mean, _ , _ = blockVisualizer._propagate_error_bar(A, dA2, -2)
        A_mean_mean, _, _, red_chi_square = blockVisualizer._propagate_error_bar(A_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], dA2_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], -1)
        parity_labels = self.blockresult.parity_labels
        grand_title = "$\phi$" + self.file_and_blind_string
        fig, axes = blockVisualizer._visualize_blockparity_vs_group(A, dA2, A_mean, dA2_mean, A_mean_mean, self.blockresult.blockcut_left, self.blockresult.blockcut_right, grand_title, 'rad', parity_labels, red_chi_square, ylim = (-0.001, 0.001))
        self.figures['phi'] = (fig, axes)

        A = self.blockresult.blinded.result['C']
        dA2 = self.blockresult.blinded.result['dC2']
        A_mean, dA2_mean, _ , _ = blockVisualizer._propagate_error_bar(A, dA2, -2)
        A_mean_mean, _, _, red_chi_square = blockVisualizer._propagate_error_bar(A_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], dA2_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], -1)
        parity_labels = self.blockresult.parity_labels
        grand_title = "$C$" + self.file_and_blind_string
        fig, axes = blockVisualizer._visualize_blockparity_vs_group(A, dA2, A_mean, dA2_mean, A_mean_mean, self.blockresult.blockcut_left, self.blockresult.blockcut_right, grand_title, '', parity_labels, red_chi_square, ylim = (-0.015,0.015))
        self.figures['C'] = (fig, axes)

        A = self.blockresult.blinded.result['A']
        dA2 = self.blockresult.blinded.result['dA2']
        A_mean, dA2_mean, _ , _ = blockVisualizer._propagate_error_bar(A, dA2, -2)
        A_mean_mean, _, _, red_chi_square = blockVisualizer._propagate_error_bar(A_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], dA2_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], -1)
        parity_labels = self.blockresult.parity_labels
        grand_title = "$A$" + self.file_and_blind_string
        fig, axes = blockVisualizer._visualize_blockparity_vs_group(A, dA2, A_mean, dA2_mean, A_mean_mean, self.blockresult.blockcut_left, self.blockresult.blockcut_right, grand_title, '', parity_labels, red_chi_square, ylim = (-0.0015,0.0015))
        self.figures['A'] = (fig, axes)

        A = self.blockresult.blinded.result['tau']
        dA2 = self.blockresult.blinded.result['dtau2']
        A_mean, dA2_mean, _ , _ = blockVisualizer._propagate_error_bar(A, dA2, -2)
        A_mean_mean, _, _, red_chi_square = blockVisualizer._propagate_error_bar(A_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], dA2_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], -1)
        parity_labels = self.blockresult.parity_labels
        grand_title = "$\\tau$" + self.file_and_blind_string
        fig, axes = blockVisualizer._visualize_blockparity_vs_group(A, dA2, A_mean, dA2_mean, A_mean_mean, self.blockresult.blockcut_left, self.blockresult.blockcut_right, grand_title, 's', parity_labels, red_chi_square, ylim = (0.0035,0.0055))
        self.figures['tau'] = (fig, axes)

    def visualize_Ap_Am(self):
        A = self.blockresult.blinded.result['Am']
        dA2 = self.blockresult.blinded.result['dAm2']
        A_mean, dA2_mean, _ , _ = blockVisualizer._propagate_error_bar(A, dA2, -2)
        A_mean_mean, _, _, red_chi_square = blockVisualizer._propagate_error_bar(A_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], dA2_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], -1)
        parity_labels = self.blockresult.parity_labels
        grand_title = "$Am$" + self.file_and_blind_string
        fig, axes = blockVisualizer._visualize_blockparity_vs_group(A, dA2, A_mean, dA2_mean, A_mean_mean, self.blockresult.blockcut_left, self.blockresult.blockcut_right, grand_title, '', parity_labels, red_chi_square, ylim = None)
        self.figures['Am'] = (fig, axes)

        A = self.blockresult.blinded.result['Ap']
        dA2 = self.blockresult.blinded.result['dAp2']
        A_mean, dA2_mean, _ , _ = blockVisualizer._propagate_error_bar(A, dA2, -2)
        A_mean_mean, _, _, red_chi_square = blockVisualizer._propagate_error_bar(A_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], dA2_mean[...,self.blockresult.blockcut_left: self.blockresult.blockcut_right], -1)
        parity_labels = self.blockresult.parity_labels
        grand_title = "$Ap$" + self.file_and_blind_string
        fig, axes = blockVisualizer._visualize_blockparity_vs_group(A, dA2, A_mean, dA2_mean, A_mean_mean, self.blockresult.blockcut_left, self.blockresult.blockcut_right, grand_title, '', parity_labels, red_chi_square, ylim = None)
        self.figures['Ap'] = (fig, axes)

    def _visualize_numberparity_vs_group(A, begin, end, title, yunit, parity_labels, ylim=None):
        # Determine the number of subplots (8 or 1 based on shape[0] of A)
        num_plots = A.shape[0]

        # Create the figure and subplots
        if num_plots == 8:
            fig, axes = plt.subplots(4, 2, figsize=(12, 16))
            axes = axes.flatten()
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            axes = [ax]  # Ensure consistency with a single plot wrapped in a list

        # Iterate over each plot (either 1 or 8 depending on num_plots)
        for i in range(num_plots):
            ax = axes[i]
            x = np.arange(A.shape[-1])  # X values based on the last axis length

            # Plot the A values as a line
            mean_line = A[i, 0, 0, 0, :]
            ax.plot(x, mean_line, color='black', lw=1, label='Mean')

            # Handle ylim based on the provided 2D array for each subplot
            if ylim is not None:
                ax.set_ylim(ylim[i][0], ylim[i][1])

            # Shade the regions where x < begin and x > end
            ax.axvspan(0, begin, color='gray', alpha=0.3)
            ax.axvspan(end, len(x), color='gray', alpha=0.3)

            # Calculate the sum of A[i, 0, 0, 0, begin:end] for the text below each subplot
            sum_value = A[i, 0, 0, 0, begin:end].sum(axis=-1)

            # Set the title with the parity label and the sum value
            ax.set_title(f"{parity_labels[i]}\n" + f"{sum_value:.3g} {yunit}", fontsize=20)

            # Set y-axis label with the yunit
            ax.set_ylabel(f"{yunit}", fontsize=12)

            # Set x-axis label
            ax.set_xlabel('Group', fontsize=14)

            # Enable grid
            ax.grid(True, linestyle='--')

        # Set the grand title for the figure
        fig.suptitle(title, fontsize=36)

        # Adjust layout to prevent overlap
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        return fig, axes

    def visualize_numberparity_vs_group(self):
        A = self.blockresult.blinded.result_sipmsum['N_sipmsum'] 
        parity_labels = self.blockresult.parity_labels
        grand_title = "$N$" + self.file_and_blind_string
        fig, axes = blockVisualizer._visualize_numberparity_vs_group(A, self.blockresult.blockcut_left, self.blockresult.blockcut_right, grand_title, 'p.e.', parity_labels,  ylim =[[0, 1E7],[-2E4,2E4],[-2E4,2E4],[-2E4,2E4],[-2E4,2E4],[-2E4,2E4],[-2E4,2E4],[-2E4,2E4]])
        self.figures['N'] = (fig, axes) 
    
    def visualize_F_vs_group(self):
        A = self.blockresult.blinded.result_sipmsum['FA_sipmsum'] 
        parity_labels = self.blockresult.parity_labels
        grand_title = "$F$" + self.file_and_blind_string
        fig, axes = blockVisualizer._visualize_numberparity_vs_group(A, self.blockresult.blockcut_left, self.blockresult.blockcut_right, grand_title, 'p.e.', parity_labels, ylim =[[-1E3, 1E3],[-1E3,1E3],[-1E3,1E3],[-1E3, 1E3],[-1E3,1E3],[-1E3,1E3],[-1E3,1E3],[-1E3,1E3],])
        self.figures['F'] = (fig, axes) 

    def visualize_phi_envlope(self):
        A = self.blockresult.blinded.result['envelop_phi_all_sipm']
        parity_labels = self.blockresult.parity_labels
        grand_title = "$\phi_{envlope}$" + self.file_and_blind_string
        fig, axes = blockVisualizer._visualize_numberparity_vs_group(A, self.blockresult.blockcut_left, self.blockresult.blockcut_right, grand_title, 'rad', parity_labels, ylim =[[-0.005,0.005],[-0.001,0.001],[-0.001,0.001],[-0.03,0.03],[-0.001,0.001],[-0.001,0.001],[-0.001,0.001],[-0.001,0.001]])
        self.figures['envlopephi'] = (fig, axes)

    def save_figures(self):
        for name, (fig, _) in self.figures.items():
            fig.savefig(f"{self.figure_folder_path}/{name}"+"_"+self.blockresult.block_string+".png")

    def close_all_figures(self):
        for name, (fig, _) in self.figures.items():
            plt.close(fig)

            