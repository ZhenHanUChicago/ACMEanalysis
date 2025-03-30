from .sequenceCalculator import sequenceCalculator
from .parityStateTransfrom import parityStateTransform, combine_switches


import pickle
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import mpld3
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pandas as pd
plt.ioff()

class sequenceVisualizer:

    @staticmethod
    def _load_results(result_file_path):
        """Load the pickled results from the file."""
        try:
            with open(result_file_path, 'rb') as f:
                sequenceresult = pickle.load(f)
            return sequenceresult
        except Exception as e:
            print(f"Error loading results: {e}")
            return None
        
    def __init__(self, sequenceresult_file_path, figure_folder_path, sequence_json_path):
        self.sequenceresult = self._load_results(sequenceresult_file_path)
        self.sequenceparameter = sequenceCalculator.Parameters()
        self.sequenceparameter._load_parameters_from_json(sequence_json_path)
        self.sequencename = self.sequenceresult.sequence_name
        self.sequencestring = self.sequenceresult.sequence_string
        self.blind_id = self.sequenceresult.blinded.blind_id
        self.superblock_switches = self.sequenceresult.superblock_parity_switches
        self.non_parity_switches = self.sequenceresult.non_parity_switches
        self.block_switches = ['N', 'E', 'B']
        _, self.superblock_parity_labels, self.superblock_state_labels = parityStateTransform(channelName= self.superblock_switches)
        _, self.block_parity_labels, self.block_state_labels = parityStateTransform(channelName= self.block_switches)
        self.labels = combine_switches(self.superblock_parity_labels, self.block_parity_labels)[1]
        self.figure_folder_path = figure_folder_path
        self.sipm_color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

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

    def _visualize_final_results_1d(self, quantity, dquantity2, sipm_red_chi_square_quantity, unit="", folder = None):
        """
        Visualize the final results for the given quantity in 1D plots, one for each element in self.labels.
        
        Parameters:
        - quantity: the key for the quantity in self.sequenceresult.blinded.final_result
        - dquantity2: the key for the error associated with quantity
        - sipm_red_chi_square_quantity: the key for the reduced chi-square quantity
        - unit: the unit string to attach to the displayed value (default: "")
        """
        for i in range(len(self.labels)):

            j_dim = list(self.sequenceresult.blinded.final_result[quantity].values())[0].shape[2]
            for j in range(j_dim):
                x_values = []
                y_values = []
                yerr_values = []
                non_parity_keys = []
                for non_parity_key in self.sequenceresult.blinded.final_result[quantity].keys():
                    label = self.labels[i][j] if j < len(self.labels[i]) else f'Unknown_Label_{i}_{j}'
                    
                    # Extract x, y, and yerr values
                    x_values.append(non_parity_key[0] if len(non_parity_key) > 0 else 0)
                    y_values.append(self.sequenceresult.blinded.final_result[quantity][non_parity_key][i, 0, j, 0, 0, 0, 0])
                    yerr_values.append(np.sqrt(self.sequenceresult.blinded.final_result[dquantity2][non_parity_key][i, 0, j, 0, 0, 0, 0]))
                    non_parity_keys.append(non_parity_key)
                # Create a new figure for each plot
                fig, ax = plt.subplots(figsize=(12,8))

                # Plot the point with error bars in black
                ax.errorbar(x_values, y_values, yerr=yerr_values, fmt='o', 
                            color='black', capsize=5)

                # Set axis labels
                ax.set_xlabel(f"{','.join(self.non_parity_switches)}")
                ax.set_ylabel(f'{quantity} {unit}')
                
                # Add grid
                ax.grid(True)

                # Set the plot title with multiple lines
                title = f"{self.labels[i][j]}\n" \
                        f"{','.join(self.non_parity_switches)}\n" \
                        f"{self.sequencename + self.sequencestring}\n" \
                        f"{self.blind_id}"
                ax.set_title(title, fontsize=12)
                # Get current y-limits to place the text correctly
                ylim = ax.get_ylim()
                for x_value,y_value,y_err,non_parity_key in zip(x_values, y_values, yerr_values, non_parity_keys):
                # Format the y_value for scientific notation
                    y_value_sci = "{:.3e}".format(y_value)
                    y_err_sci = "{:.3e}".format(y_err)
                    # Add text displaying the value in scientific notation
                    ax.text(x_value, y_value - 0.15 * (ylim[1] - ylim[0]), f"{y_value_sci} {unit}", 
                            ha='center', va='top', fontsize=10)

                    ax.text(x_value, y_value - 0.25 * (ylim[1] - ylim[0]), f"$\pm${y_err_sci} {unit}", 
                            ha='center', va='top', fontsize=10)

                    # Display the reduced chi-square value below the y_value
                    red_chi_square_value = self.sequenceresult.blinded.final_result[sipm_red_chi_square_quantity][non_parity_key][i, 0, j, 0, 0, 0, 0]
                    ax.text(x_value, y_value - 0.05 * (ylim[1] - ylim[0]), 
                            rf"$\chi^2_{{red}}(SiPM) = {red_chi_square_value:.2f}$", 
                            ha='center', va='top', fontsize=10)

                # Apply a tight layout to ensure the elements fit well
                fig.tight_layout()

                if quantity == "omega":
                    if np.max(np.abs(y_values)) > 0.1:
                        fig.patch.set_facecolor('yellow')

                # Save the individual plot with the appropriate filename
                if label == "nr":
                    fig_filename = f"{folder}/nonreversing_{quantity}.png"
                else:
                    fig_filename = f"{folder}/{label}_{quantity}.png"
                plt.savefig(fig_filename)
                plt.close(fig)

    def visualize_final_results_1d(self):
        folder_path = os.path.join(self.figure_folder_path, 'result_vs_nonparity')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for subfolder_name in ('omega', 'tau', 'phi', 'C', 'A', 'Ap', 'Am'):
            subfolder_path = os.path.join(folder_path, subfolder_name)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

        self._visualize_final_results_1d('omega', 'domega2', 'sipm_red_chi_square_omega', 'rad/s', folder=os.path.join(folder_path, 'omega'))
        self._visualize_final_results_1d('tau', 'dtau2', 'sipm_red_chi_square_tau', 's', folder=os.path.join(folder_path, 'tau'))
        self._visualize_final_results_1d('phi', 'dphi2', 'sipm_red_chi_square_phi', 'rad', folder=os.path.join(folder_path, 'phi'))
        self._visualize_final_results_1d('C', 'dC2', 'sipm_red_chi_square_C', '', folder=os.path.join(folder_path, 'C'))
        self._visualize_final_results_1d('A', 'dA2', 'sipm_red_chi_square_A', '', folder=os.path.join(folder_path, 'A'))
        self._visualize_final_results_1d('Ap', 'dAp2', 'sipm_red_chi_square_Ap', '', folder=os.path.join(folder_path, 'Ap'))
        self._visualize_final_results_1d('Am', 'dAm2', 'sipm_red_chi_square_Am', '', folder=os.path.join(folder_path, 'Am'))

    def visualize_degenerate_blocks(self, blockheader_df, additional_columns):
        def plot_block_quantities(df, blockheader_df, save_path, superblock_parity_switches, non_parity_switches, additional_columns):
            # Merge df with blockheader_df on 'run', 'sequence', and 'block'
            df = pd.merge(df, blockheader_df, on=['run', 'sequence', 'block'], how='left', suffixes= (None, '_duplicate'))
            
            # Define quantities and channels
            quantities = ["C", "phi", "omega", "tau", "A", "Ap", "Am"]
            channels = ["nr", "N", "E", "B", "NE", "NB", "EB", "NEB"]
            sipm_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

            # Function to create tick labels with formatted mean and chi-square
            def create_tick_label(row, switches, mean_value=None, red_chi2=None):
                labels = []
                for switch in switches:
                    sign = "+" if row[switch] > 0 else ""
                    labels.append(f"{switch}\n{sign}{row[switch]}")
                if mean_value is not None and red_chi2 is not None:
                    labels.append(f"mean\n{mean_value:.2g}\n$\chi^2_{{red}}[blocks]$ \n{red_chi2:.2g}")
                return "\n".join(labels)

            for quantity in quantities:
                quantity_path = os.path.join(save_path, f"block_{quantity}")
                os.makedirs(quantity_path, exist_ok=True)
                channels_to_plot = channels if quantity != "tau" else ["nr"]

                for channel in channels_to_plot:
                    fig, ax = plt.subplots(figsize=(24, 12))
                    x = range(len(df))

                    # Plot SiPM values with reduced alpha
                    for i in range(8):
                        if f'block_{quantity}_{channel}_sipm{i}' in df.columns:
                            sipm_y = df[f'block_{quantity}_{channel}_sipm{i}']
                            ax.scatter(x, sipm_y, color=sipm_colors[i], label=f'SiPM {i}', alpha=0.4, marker='.', s=100)

                    # Plot center values with error bars
# Plot center values with error bars and annotate block numbers
# Plot center values with error bars and annotate block numbers
                    y = df[f'block_{quantity}_{channel}']
                    y_err = df[f'blockuncertainty_{quantity}_{channel}']
                    ax.errorbar(x, y, yerr=y_err, fmt='.', capsize=4, capthick=4, ecolor='black', color='black', markersize=6)

                    # Annotate block numbers in bright green (#00FF00) with bold text and larger font size
                    for i, (xi, yi, bi) in enumerate(zip(x, y, df['block'])):
                        if not pd.isnull(yi):  # Only annotate valid points
                            ax.text(
                                xi, yi + y_err.iloc[i] + 0.05 * max(y_err),  # Adjust position to appear just above the error bar
                                f"{int(bi)}",  # Convert block number to integer for display
                                color='#FF0000',  # Bright green color
                                fontsize=10,  # Slightly larger font size
                                fontweight='bold',  # Make text bold
                                ha='center',  # Center-align the text horizontally
                                va='bottom'  # Align text to the bottom to ensure it appears above the error bar
                            )


                    # Prepare to collect all legends
                    lines_labels = [ax.get_legend_handles_labels()]
                    
                    # Plot additional columns with separate y-axes for each one
                    for i, col in enumerate(additional_columns):
                        ax2 = ax.twinx()
                        ax2.spines["right"].set_position(("outward", 60 * i))  # Separate y-axes for each column
                        line = ax2.plot(x, df[col], label=col, linestyle='-', color=f'C{i}')
                        ax2.set_ylabel(col)
                        ax2.tick_params(axis='y')
                        ax2.yaxis.label.set_color(f'C{i}')
                        
                        # Collect handles and labels for the current axis
                        lines_labels.append(ax2.get_legend_handles_labels())

                    # Combine all the legends into one on the main axis
                    handles, labels = [sum(lol, []) for lol in zip(*lines_labels)]
                    ax.legend(handles, labels, loc='upper right')

                    # Variables for interval processing
                    ticks, tick_labels = [], []
                    prev_non_parity, prev_superblock = None, None
                    superblock_start = 0
                    interval_y, interval_y_err = [], []

                    # Loop through data
                    for index, row in df.iterrows():
                        non_parity_values = tuple(row[switch] for switch in non_parity_switches)
                        superblock_values = tuple(row[switch] for switch in superblock_parity_switches)

                        # Handle non-parity changes
                        if prev_non_parity is None or non_parity_values != prev_non_parity:
                            if prev_non_parity is not None:
                                ax.axvline(x=index - 0.5, linestyle='--', color='black', linewidth=1)
                            prev_non_parity = non_parity_values

                        # Handle superblock changes
                        if prev_superblock is None or superblock_values != prev_superblock:
                            if prev_superblock is not None:
                                ax.axvline(x=index - 0.5, linestyle='--', color='black', linewidth=1)


                                """
                                # Calculate weighted mean and chi-square
                                if interval_y:
                                    weights = 1 / np.square(interval_y_err)
                                    weighted_avg = np.sum(interval_y * weights) / np.sum(weights)
                                    chi_square = np.sum(weights * (interval_y - weighted_avg)**2) / (len(interval_y) - 1)
                                    ax.hlines(weighted_avg, superblock_start - 0.5, index - 0.5, colors='red', linestyles='dashed')

                                    # Label for previous interval
                                    mid_point = (superblock_start + index) // 2
                                    tick_label = create_tick_label(df.iloc[superblock_start], superblock_parity_switches+non_parity_switches, weighted_avg, chi_square)
                                    ticks.append(mid_point)
                                    tick_labels.append(tick_label)
                                """

# Calculate robust mean and chi-square
                                if interval_y:
                                    # Convert interval values to numpy arrays
                                    interval_y = np.array(interval_y)
                                    interval_y_err = np.array(interval_y_err)
                                    
                                    # Calculate initial mean and standard deviation
                                    initial_mean = np.mean(interval_y)
                                    std_dev = np.std(interval_y)

                                    # Apply sigma clipping: keep only values within 3 sigma of the mean
                                    mask = np.abs(interval_y - initial_mean) <= 3 * std_dev
                                    clipped_y = interval_y[mask]
                                    clipped_y_err = interval_y_err[mask]

                                    # Check if enough points remain after clipping
                                    if len(clipped_y) > 1:
                                        weights = 1 / np.square(clipped_y_err)
                                        robust_mean = np.sum(clipped_y * weights) / np.sum(weights)
                                        chi_square = np.sum(weights * (clipped_y - robust_mean) ** 2) / (len(clipped_y) - 1)
                                    else:
                                        # Fallback to mean without clipping if not enough points
                                        robust_mean = np.mean(interval_y)
                                        chi_square = np.nan  # Cannot calculate chi-square with fewer points

                                    ax.hlines(robust_mean, superblock_start - 0.5, index - 0.5, colors='red', linestyles='dashed')

                                    # Label for previous interval
                                    mid_point = (superblock_start + index) // 2
                                    tick_label = create_tick_label(df.iloc[superblock_start], superblock_parity_switches + non_parity_switches, robust_mean, chi_square)
                                    ticks.append(mid_point)
                                    tick_labels.append(tick_label)


                            # Reset for next interval
                            superblock_start = index
                            interval_y, interval_y_err = [], []
                            prev_superblock = superblock_values

                        # Collect values for the current interval
                        interval_y.append(row[f'block_{quantity}_{channel}'])
                        interval_y_err.append(row[f'blockuncertainty_{quantity}_{channel}'])

                    # Last interval
                    if interval_y:
                        weights = 1 / np.square(interval_y_err)
                        weighted_avg = np.sum(interval_y * weights) / np.sum(weights)
                        chi_square = np.sum(weights * (interval_y - weighted_avg)**2) / (len(interval_y) - 1)
                        ax.hlines(weighted_avg, superblock_start - 0.5, len(df) - 0.5, colors='red', linestyles='dashed')
                        mid_point = (superblock_start + len(df)) // 2
                        tick_label = create_tick_label(df.iloc[superblock_start], superblock_parity_switches + non_parity_switches , weighted_avg, chi_square)
                        ticks.append(mid_point)
                        tick_labels.append(tick_label)

                    # Apply labels and save
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(tick_labels)
                    ax.tick_params(axis='x', labelsize=7)
                    ax.tick_params(axis='y', labelsize=15)
                    ax.set_title(f"{quantity} - {channel}", fontsize=40)
                    ax.grid(axis='y')
                    plt.tight_layout()
                    fig.savefig(os.path.join(quantity_path, f"{quantity}_{channel}.png"))
                    plt.close(fig)


        def plot_block_N(df, blockheader_df, save_path, superblock_parity_switches, non_parity_switches, additional_columns):
            # Merge df with blockheader_df on 'run', 'sequence', and 'block'
            df = pd.merge(df, blockheader_df, on=['run', 'sequence', 'block'], how='left', suffixes= (None, '_duplicate'))

            # Define channels and colors
            channels = ["nr", "N", "E", "B", "NE", "NB", "EB", "NEB"]
            sipm_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

            # Path setup for 'N' quantities
            quantity_path = os.path.join(save_path, "block_N")
            os.makedirs(quantity_path, exist_ok=True)

            # Function to create tick labels without mean or chi-square
            def create_tick_label(row, switches):
                return "\n".join([f"{switch}\n{'+' if row[switch] > 0 else ''}{row[switch]}" for switch in switches])

            for channel in channels:
                fig, ax = plt.subplots(figsize=(24, 12))
                x = range(len(df))

                # Scatter plot for SiPM values with reduced alpha
                for i in range(8):
                    if f'block_N_{channel}_sipm{i}' in df.columns:
                        sipm_y = df[f'block_N_{channel}_sipm{i}']
                        ax.scatter(x, sipm_y, color=sipm_colors[i], label=f'SiPM {i}', alpha=0.4, marker='.', s=100)

                # Plot center values as black dots (sum over 8 SiPMs)
                y = df[f'block_N_{channel}'] / 8
                ax.plot(x, y, 'k.', markersize=6)

                # Prepare to collect all legends
                lines_labels = [ax.get_legend_handles_labels()]

                # Plot additional columns with separate y-axes for each one
                for i, col in enumerate(additional_columns):
                    ax2 = ax.twinx()
                    ax2.spines["right"].set_position(("outward", 60 * i))  # Separate y-axes for each column
                    line = ax2.plot(x, df[col], label=col, linestyle='-', color=f'C{i}')
                    ax2.set_ylabel(col)
                    ax2.tick_params(axis='y')
                    ax2.yaxis.label.set_color(f'C{i}')
                    
                    # Collect handles and labels for the current axis
                    lines_labels.append(ax2.get_legend_handles_labels())

                # Combine all the legends into one on the main axis
                handles, labels = [sum(lol, []) for lol in zip(*lines_labels)]
                ax.legend(handles, labels, loc='upper right')

                # Variables for interval processing
                ticks, tick_labels = [], []
                prev_non_parity, prev_superblock = None, None
                superblock_start = 0

                # Loop through data
                for index, row in df.iterrows():
                    non_parity_values = tuple(row[switch] for switch in non_parity_switches)
                    superblock_values = tuple(row[switch] for switch in superblock_parity_switches)

                    # Handle non-parity changes
                    if prev_non_parity is None or non_parity_values != prev_non_parity:
                        if prev_non_parity is not None:
                            ax.axvline(x=index - 0.5, linestyle='--', color='black', linewidth=4)
                        prev_non_parity = non_parity_values

                    # Handle superblock changes
                    if prev_superblock is None or superblock_values != prev_superblock:
                        if prev_superblock is not None:
                            ax.axvline(x=index - 0.5, linestyle='--', color='black', linewidth=4)

                            # Label for previous interval
                            mid_point = (superblock_start + index) // 2
                            tick_label = create_tick_label(df.iloc[superblock_start], superblock_parity_switches+non_parity_switches)
                            ticks.append(mid_point)
                            tick_labels.append(tick_label)

                        superblock_start = index
                        prev_superblock = superblock_values

                # Apply labels and save
                ax.set_xticks(ticks)
                ax.set_xticklabels(tick_labels)
                ax.tick_params(axis='x', labelsize=7)
                ax.tick_params(axis='y', labelsize=15)
                ax.set_title(f"N - {channel} /8", fontsize=40)
                ax.grid(axis='y')
                plt.tight_layout()
                fig.savefig(os.path.join(quantity_path, f"N_{channel}.png"))
                plt.close(fig)

        folder_path = os.path.join(self.figure_folder_path, 'blockview_timeorder')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        df = self.sequenceresult.blockdf
        save_path = folder_path
        superblock_parity_switches = self.superblock_switches
        non_parity_switches = self.non_parity_switches

        plot_block_quantities(df, blockheader_df, save_path, superblock_parity_switches, non_parity_switches, additional_columns)
        plot_block_N(df, blockheader_df,save_path, superblock_parity_switches, non_parity_switches, additional_columns)

        sort_columns = non_parity_switches + superblock_parity_switches + ['run', 'sequence', 'block']
        df_sorted = df.sort_values(by=sort_columns)
        df_sorted.reset_index(inplace=True)
        folder_path = os.path.join(self.figure_folder_path, 'blockview_stateorder')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path = folder_path
        plot_block_quantities(df_sorted, blockheader_df,save_path, superblock_parity_switches, non_parity_switches, additional_columns)
        plot_block_N(df_sorted, blockheader_df, save_path, superblock_parity_switches, non_parity_switches, additional_columns)

    def _visualize_groups(dic, error2_dic, labels, name, non_parity_switches, figure_path, k_begin, k_end, unit):
        # Ensure the figure path exists
        os.makedirs(figure_path, exist_ok=True)

        # Extract the shape i and j from the first item in dic
        sample_matrix = next(iter(dic.values()))
        i, j = sample_matrix.shape[0], sample_matrix.shape[2]

        # Sort dictionary keys for consistent ordering in plots
        sorted_keys = sorted(dic.keys())

        # Initialize color map for gradual coloring
        cmap = plt.cm.viridis
        colors = [cmap(x / len(sorted_keys)) for x in range(len(sorted_keys))]

        for idx_i in range(i):
            for idx_j in range(j):
                # Create figure with specified size
                fig, ax = plt.subplots(figsize=(20, 12))  # 2x size increase for each dimension

                x_data = np.arange(k_begin, k_end)

                for line_idx, key in enumerate(sorted_keys):
                    # Extract the data for the specific slice and key
                    y_data = dic[key][idx_i, 0, idx_j, 0, 0, 0, k_begin:k_end]

                    # Calculate ±1 sigma range if error2_dic is provided
                    if error2_dic:
                        y_errors = np.sqrt(error2_dic[key][idx_i, 0, idx_j, 0, 0, 0, k_begin:k_end])
                        ax.fill_between(x_data, y_data - y_errors, y_data + y_errors,
                                        color=colors[line_idx], alpha=0.1)

                    # Plot the data points
                    line_label = ', '.join([f"{np_switch} = {state}" for np_switch, state in zip(non_parity_switches, key)])
                    ax.plot(x_data, y_data, label=line_label, color=colors[line_idx],
                            marker='.',  markersize=15, alpha=0.5)

                # Set plot labels and add grid
                ax.set_xlabel('Groups')
                ax.set_ylabel(unit)
                ax.grid(True)

                # Set title with increased font size
                ax.set_title(f"{name}_{labels[idx_i][idx_j]}", fontsize=32)

                # Add a thick black dashed line at y=0
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

                # Set axis limits and increase y-axis tick density
                ax.set_xlim(k_begin, k_end)
                ax.yaxis.set_major_locator(plt.MaxNLocator(20))

                # Add legend to the top right corner
                ax.legend(loc='upper right')

                # Save the figure
                fig_name = f"group_{name}_{labels[idx_i][idx_j]}.png"
                fig_path = os.path.join(figure_path, fig_name)
                plt.savefig(fig_path)
                plt.close(fig)

    def visualize_groups(self):
        folder_path = os.path.join(self.figure_folder_path, 'result_vs_groups')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for subfolder_name in ('omega', 'tau', 'phi', 'C', 'A'):
            subfolder_path = os.path.join(folder_path, subfolder_name)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
        omega = {}
        domega2 = {}
        tau = {}
        dtau2 = {}
        phi = {}
        dphi2 = {}
        C = {}
        dC2 = {}
        A = {}
        dA2 = {}
        for key in self.sequenceresult.blinded.result['omega'].keys():
            omega[key], domega2[key], _,_ = sequenceVisualizer._propagate_error_bar(self.sequenceresult.blinded.result['omega'][key], self.sequenceresult.blinded.result['domega2'][key], -2)
            tau[key], dtau2[key], _,_ = sequenceVisualizer._propagate_error_bar(self.sequenceresult.blinded.result['tau'][key], self.sequenceresult.blinded.result['dtau2'][key], -2)
            phi[key], dphi2[key], _,_ = sequenceVisualizer._propagate_error_bar(self.sequenceresult.blinded.result['phi'][key], self.sequenceresult.blinded.result['dphi2'][key], -2)
            C[key], dC2[key], _,_ = sequenceVisualizer._propagate_error_bar(self.sequenceresult.blinded.result['C'][key], self.sequenceresult.blinded.result['dC2'][key], -2)
            A[key], dA2[key], _,_ = sequenceVisualizer._propagate_error_bar(self.sequenceresult.blinded.result['A'][key], self.sequenceresult.blinded.result['dA2'][key], -2)
        sequenceVisualizer._visualize_groups(omega, domega2, self.sequenceresult.labels, 'omega', self.non_parity_switches, os.path.join(folder_path, 'omega'), self.sequenceresult.sequence_group_left, self.sequenceresult.sequence_group_right, 'rad/s')
        sequenceVisualizer._visualize_groups(tau, dtau2, self.sequenceresult.labels, 'tau', self.non_parity_switches, os.path.join(folder_path, 'tau'), self.sequenceresult.sequence_group_left, self.sequenceresult.sequence_group_right, 's')
        sequenceVisualizer._visualize_groups(phi, dphi2, self.sequenceresult.labels, 'phi', self.non_parity_switches, os.path.join(folder_path, 'phi'), self.sequenceresult.sequence_group_left, self.sequenceresult.sequence_group_right, 'rad')
        sequenceVisualizer._visualize_groups(C, dC2, self.sequenceresult.labels, 'C', self.non_parity_switches, os.path.join(folder_path, 'C'), self.sequenceresult.sequence_group_left, self.sequenceresult.sequence_group_right, '')
        sequenceVisualizer._visualize_groups(A, dA2, self.sequenceresult.labels, 'A', self.non_parity_switches, os.path.join(folder_path, 'A'), self.sequenceresult.sequence_group_left, self.sequenceresult.sequence_group_right, '')