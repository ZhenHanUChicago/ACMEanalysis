from .parityStateTransfrom import *
from .headerHandler import *


import json
import numpy as np
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import warnings
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import pickle
import copy

class sequenceCalculator:

    class blind_object:
        def __init__(self, blind_value_in_rad_s, blind_id):
            self._blind_value_in_rad_s = blind_value_in_rad_s
            self.blind_id = blind_id

        def __str__(self):
            return f"************rad/s, blind name: {self.blind_id}"

        def __repr__(self):
            return f"************rad/s, blind name: {self.blind_id}"
        
    class Parameters:
        def __init__(self):
            self.non_parity_switches =  []
            self.superblock_parity_switches = ["P", "L", "R"]

        def _load_parameters_from_json(self, parameter_file_path):
            try:
                with open(parameter_file_path, 'r') as f:
                    param_dict = json.load(f)
                
                for key, value in param_dict.items():
                    if hasattr(self, key) and value is not None:
                        setattr(self, key, value)
            except FileNotFoundError:
                print(f"Sequence switch specification {parameter_file_path} not found, using default values.")
            except json.JSONDecodeError:
                print(f"Sequence: Error decoding JSON file {parameter_file_path}, using default values.")
            except Exception as e:
                print(f"Sequence: Unexpected error: {e}")

    class SequenceResults:

        class Blinded:
            def __init__(self):
                self.blind_id = None
                self.result = {}
                self.result_summary = {}
                self.superblock_state_result = {}
                self.final_result = {}

        class Unblinded:
            def __init__(self):
                self.result = {}
                self.result_summary = {}
                self.superblock_state_result = {}
                self.final_result = {}

        def __init__(self):
            self.blinded = self.Blinded()
            self.__unblinded = self.Unblinded()
            self.sequence_range = None
            self.sequence_run_sequence = None
            self.sequence_type = None
            self.non_parity_switches = []
            self.superblock_parity_switches = []
            self.superblock_parity_labels = None
            self.superblock_state_labels = None
            self.sequence_name = None
            self.sequence_string = None
            self.labels = None
            self.blockdf = None
            self.sequencedf = None
            self.sequencesipmdf = None
            self.sequence_group_left = None
            self.sequence_group_right = None

    def read_blind(self, folder_path):
        # Define the file paths
        blind_bytes_path = os.path.join(folder_path, 'blind_bytes.txt')
        private_key_path = os.path.join(folder_path, 'private_key.txt')

        # Try to read the blind bytes from the txt file
        try:
            with open(blind_bytes_path, 'rb') as f:
                extracted_bytes = f.read()
        except FileNotFoundError as e:
            raise Exception(f"Failed to open blind bytes file: {blind_bytes_path}") from e
        except Exception as e:
            raise Exception(f"An error occurred while reading the blind bytes file: {blind_bytes_path}") from e

        # Try to read the private key from the file
        try:
            with open(private_key_path, 'rb') as f:
                private_key_data = f.read()
        except FileNotFoundError as e:
            raise Exception(f"Failed to open private key file: {private_key_path}") from e
        except Exception as e:
            raise Exception(f"An error occurred while reading the private key file: {private_key_path}") from e

        # Skip the first line of the private key and extract the last four characters of the second line
        try:
            private_key_lines = private_key_data.splitlines()
            key_id_start = private_key_lines[1][-4:].decode('utf-8', errors='ignore')
        except Exception as e:
            raise Exception("Failed to extract the last four letters from the second line of the private key") from e

        # Try to deserialize the private key
        try:
            private_key = serialization.load_pem_private_key(
                private_key_data,
                password=None
            )
        except Exception as e:
            raise Exception("Failed to deserialize the private key") from e

        # Try to decrypt the blind using the private key
        try:
            decrypted_blind = private_key.decrypt(
                extracted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        except Exception as e:
            raise Exception("Failed to decrypt the blind value") from e

        # Try to convert the decrypted blind back to a double
        try:
            blind_value_in_rad_s = np.frombuffer(decrypted_blind, dtype=np.float64)[0]
        except Exception as e:
            raise Exception("Failed to convert decrypted blind to float64") from e

        # Create the blind ID using the folder name and the last four letters of the second line of the private key
        folder_name = os.path.basename(folder_path)
        blind_id = f"{folder_name}-{key_id_start}"

        # Return the blind_object
        self.blind =  sequenceCalculator.blind_object(blind_value_in_rad_s, blind_id)

    def __init__(self, sequence_json_path = None, blockresults_path_list = None, blockcutresults_path_list = None, sequence_result_folder_path = None, df = None, blind_path = None, sequence_name = None):
        self.sequence_result_folder_path = sequence_result_folder_path
        self.blockresults_path_list = blockresults_path_list
        self.blockcutresults_path_list = blockcutresults_path_list
        
        self.blind = None
        self.read_blind(blind_path)
        self.parameters = self.Parameters()
        self.parameters._load_parameters_from_json(sequence_json_path)

        self.P, self.superblock_parity_labels, self.superblock_state_labels = parityStateTransform(channelName= self.parameters.superblock_parity_switches)
        _, self.block_parity_labels, self.block_state_labels = parityStateTransform(channelName= ['N','E','B'])
        _, self.labels = combine_switches(self.superblock_parity_labels, self.block_parity_labels)

        self.sequenceresult = self.SequenceResults()
        self.sequenceresult.blinded.blind_id = self.blind.blind_id
        self.sequenceresult.superblock_parity_labels = self.superblock_parity_labels
        self.sequenceresult.superblock_state_labels = self.superblock_state_labels
        self.sequenceresult.labels = self.labels
                

        self.sequenceresult.non_parity_switches = self.parameters.non_parity_switches
        self.non_parity_switches = self.parameters.non_parity_switches

        self.sequenceresult.superblock_parity_switches = self.parameters.superblock_parity_switches
        self.superblock_parity_switches = self.parameters.superblock_parity_switches
    
        self.sequenceresult.sequence_type = os.path.splitext(os.path.basename(sequence_json_path))[0]

        self.sequenceresult.sequence_range = []
        for path in blockresults_path_list:
            filename = os.path.basename(path)  # Get the file name from the path
            parts = filename.replace('blockresult_', '').replace('.pkl', '').split('.')  # Split to get run, sequence, block
            run = int(parts[0])  # Extract run number
            sequence = int(parts[1])  # Extract sequence number
            block = int(parts[2])  # Extract block number
            self.sequenceresult.sequence_range.append((run, sequence, block))  # Append (run, sequence) to the list
        
        self.sequenceresult.sequence_range.sort()
        self.sequence_run_seqeuence = sorted(list(set([(r[0], r[1]) for r in self.sequenceresult.sequence_range])))

        sequence_string = ""
        for run,sequence in self.sequence_run_seqeuence:
            sequence_string += "_" + str(int(run)).zfill(4) + "." + str(int(sequence)).zfill(4)
        
        self.sequenceresult.sequence_string = sequence_string
        self.sequenceresult.sequence_run_sequence = self.sequence_run_seqeuence
        self.sequenceresult.sequence_name = sequence_name
        self.df = df

        self._load_blockresults()
        self._create_superblock_state_dict()
        self._create_sequence_results()
        self._deep_copy_results()
        self._apply_blinding()
        self.saveresults()

    def blockresult_to_df(b):
        flat_dict = {}
        
        # Extract run, sequence, and block from b.block_string
        block_string = b.block_string
        run, sequence, block = [int(x) for x in block_string.split('.')]
        flat_dict['run'] = run
        flat_dict['sequence'] = sequence
        flat_dict['block'] = block
        for key, matrix in b.blinded.result_summary.items():
            # Remove "_summary" from the key
            base_name = key.replace('_summary', '')
            
            # Determine if it's a variance (starts with 'd' and ends with '2')
            is_variance = key.startswith('d') and key.endswith('2_summary')
            
            if is_variance:
                # Strip 'd' and '2' from base_name
                base_name = base_name[1:-1]
            
            # Get the size of the matrix
            shape = matrix.shape
            
            # Iterate over each element of the matrix
            for i in range(shape[0]):  # Adjust dimension index based on shape
                for p in range(shape[3]):  # 'p' is for SIPM in [k, 1, 1, p, 1]
                    # Get the parity label from b.parity_labels
                    channel = b.parity_labels[i]
                    
                    # Determine the SIPM number (sipm = p)
                    sipm = p
                    
                    # Build the name for this entry
                    if is_variance:
                        name = f"blockuncertainty_{base_name}_{channel}_sipm{sipm}"
                        value = np.sqrt(matrix[i, 0, 0, p, 0])  # Take sqrt for variance
                    else:
                        name = f"block_{base_name}_{channel}_sipm{sipm}"
                        value = matrix[i, 0, 0, p, 0]  # Leave value as is for quantity
                    
                    # Add run, sequence, and block information
                    flat_dict[name] = value

        
        # Convert the flat dictionary to a one-line DataFrame
        df = pd.DataFrame([flat_dict])
        return df

    def _load_blockresults(self):
        # Initialize the dictionary to store the block results
        self.sequence_begin_group = 0
        self.sequence_end_group = 10000

        self.blockdict = {}
        self.sequenceresult.blockdf = []
        # Loop over all blockresult files in the blockresults_path_list
        for blockresult_path in self.blockresults_path_list:
            # Extract run, sequence, and block number from the filename
            filename = os.path.basename(blockresult_path)
            parts = filename.replace('blockresult_', '').replace('.pkl', '').split('.')
            run, sequence, block = int(parts[0]), int(parts[1]), int(parts[2])

            # Construct the corresponding blockcutresult file path
            blockcutresult_path = blockresult_path.replace('blockresult_', 'blockcutresult_')

            # Load the blockresult and blockcutresult pkl files
            with open(blockresult_path, 'rb') as f:
                blockresult = pickle.load(f)

            with open(blockcutresult_path, 'rb') as f:
                blockcutresult = pickle.load(f)

            # Check if the block stays based on blockcutresult
            if not blockcutresult.does_this_block_stay:
                continue  # Skip this block if it doesn't stay

            self.sequence_begin_group = max(self.sequence_begin_group, blockresult.blockcut_left)
            self.sequence_end_group = min(self.sequence_end_group, blockresult.blockcut_right)

            block_df = sequenceCalculator.blockresult_to_df(blockresult)
            


            # Quantities from blockresult to be used as first layer keys
            quantities = {
                'C': blockresult._BlockResults__unblinded.result['C'],
                'phi': blockresult._BlockResults__unblinded.result['phi'],
                'omega': blockresult._BlockResults__unblinded.result['omega'],
                'tau': blockresult._BlockResults__unblinded.result['tau'],
                'dC2': blockresult._BlockResults__unblinded.result['dC2'],
                'dphi2': blockresult._BlockResults__unblinded.result['dphi2'],
                'domega2': blockresult._BlockResults__unblinded.result['domega2'],
                'dtau2': blockresult._BlockResults__unblinded.result['dtau2'],
                'N': blockresult._BlockResults__unblinded.result['N'],
                'A': blockresult._BlockResults__unblinded.result['A'],
                'dA2': blockresult._BlockResults__unblinded.result['dA2'],
                'xsipm': blockresult._BlockResults__unblinded.result['xsipm'],
                'ysipm': blockresult._BlockResults__unblinded.result['ysipm'],
                'zsipm': blockresult._BlockResults__unblinded.result['zsipm'],
                'yzsipm': blockresult._BlockResults__unblinded.result['yzsipm'],
                'Ap': blockresult._BlockResults__unblinded.result['Ap'],
                'dAp2': blockresult._BlockResults__unblinded.result['dAp2'],
                'Am': blockresult._BlockResults__unblinded.result['Am'],
                'dAm2': blockresult._BlockResults__unblinded.result['dAm2'],
                'C_summary': blockresult._BlockResults__unblinded.result_summary['C_summary'],
                'phi_summary': blockresult._BlockResults__unblinded.result_summary['phi_summary'],
                'omega_summary': blockresult._BlockResults__unblinded.result_summary['omega_summary'],
                'tau_summary': blockresult._BlockResults__unblinded.result_summary['tau_summary'],
                'dC2_summary': blockresult._BlockResults__unblinded.result_summary['dC2_summary'],
                'dphi2_summary': blockresult._BlockResults__unblinded.result_summary['dphi2_summary'],
                'domega2_summary': blockresult._BlockResults__unblinded.result_summary['domega2_summary'],
                'dtau2_summary': blockresult._BlockResults__unblinded.result_summary['dtau2_summary'],
                'N_summary': blockresult._BlockResults__unblinded.result_summary['N_summary'],
                'A_summary': blockresult._BlockResults__unblinded.result_summary['A_summary'],
                'dA2_summary': blockresult._BlockResults__unblinded.result_summary['dA2_summary'],
                'xsipm_summary': blockresult._BlockResults__unblinded.result_summary['xsipm_summary'],
                'ysipm_summary': blockresult._BlockResults__unblinded.result_summary['ysipm_summary'],
                'zsipm_summary': blockresult._BlockResults__unblinded.result_summary['zsipm_summary'],
                'yzsipm_summary': blockresult._BlockResults__unblinded.result_summary['yzsipm_summary'],
                'Ap_summary': blockresult._BlockResults__unblinded.result_summary['Ap_summary'],
                'dAp2_summary': blockresult._BlockResults__unblinded.result_summary['dAp2_summary'],
                'Am_summary': blockresult._BlockResults__unblinded.result_summary['Am_summary'],
                'dAm2_summary': blockresult._BlockResults__unblinded.result_summary['dAm2_summary']
            }

            # Extract non-parity and superblock parity switch values from self.df
            temp_df = self.df[(self.df['run'] == run) & (self.df['sequence'] == sequence) & (self.df['block'] == block)]
            
            # Get the tuple for non_parity_switches
            if self.non_parity_switches:
                non_parity_tuple = tuple(temp_df[sw].values[0] for sw in self.non_parity_switches)
            else:
                non_parity_tuple = ()

            # Get the tuple for superblock_parity_switches
            if self.superblock_parity_switches:
                superblock_parity_tuple = tuple(temp_df[sw].values[0] for sw in self.superblock_parity_switches)
            else:
                superblock_parity_tuple = ()

            block_df_old_columns = block_df.columns 

            for idx, label in enumerate(self.superblock_parity_switches):
                block_df[label] = superblock_parity_tuple[idx]
            
            for idx, label in enumerate(self.non_parity_switches):
                block_df[label] = non_parity_tuple[idx]

            # move the superblock_parity_tuple to the front
            block_df = block_df[["run", "sequence", "block"] + self.non_parity_switches + self.superblock_parity_switches + list(block_df_old_columns[3:])]
            self.sequenceresult.blockdf.append(block_df)

            # Populate the blockdict with the arrays from blockresult
            for quantity, array in quantities.items():
                if quantity not in self.blockdict:
                    self.blockdict[quantity] = {}

                if non_parity_tuple not in self.blockdict[quantity]:
                    self.blockdict[quantity][non_parity_tuple] = {}

                if superblock_parity_tuple not in self.blockdict[quantity][non_parity_tuple]:
                    self.blockdict[quantity][non_parity_tuple][superblock_parity_tuple] = []

                # Append the array to the corresponding dictionary entry
                self.blockdict[quantity][non_parity_tuple][superblock_parity_tuple].append(array)
        self.sequenceresult.blockdf = pd.concat(self.sequenceresult.blockdf, ignore_index=True)

        self.sequenceresult.sequence_group_left = self.sequence_begin_group
        self.sequenceresult.sequence_group_right = self.sequence_end_group

    def _diag_along_axis(A, axis):
        shape = A.shape
        new_shape = list(shape)
        new_shape.insert(axis, shape[axis])
        
        diag_mask = np.eye(shape[axis], dtype=A.dtype)
        expanded_diag_shape = [1] * (len(shape) + 1)
        expanded_diag_shape[axis] = shape[axis]
        expanded_diag_shape[axis + 1] = shape[axis]
        diag_mask = diag_mask.reshape(expanded_diag_shape)
        
        expanded_A_shape = list(shape)
        expanded_A_shape.insert(axis, 1)
        expanded_A = A.reshape(expanded_A_shape)
        
        B = diag_mask * expanded_A
        
        return B

    def _extract_diagonal_along_axes(matrix, axis):
        """
        Extract the diagonal elements along the specified axis and the next axis,
        and bring the diagonal axis to the position of the specified axis.

        Parameters:
        matrix (np.ndarray): The input n-dimensional array.
        axis (int): The axis along which to extract diagonals with the next axis.

        Returns:
        np.ndarray: The resultant array after extracting diagonals.
        """
        if axis < 0 or axis >= matrix.ndim - 1:
            raise ValueError("Axis out of bounds or too high for the given matrix dimensions.")
        
        # Extract the diagonal along the specified axis and the next axis
        diag_matrix = np.diagonal(matrix, axis1=axis, axis2=axis + 1)
        
        # Move the diagonal axis to the original position of the specified axis
        new_axes_order = list(range(diag_matrix.ndim))
        new_axes_order.insert(axis, new_axes_order.pop(-1))
        
        result = np.transpose(diag_matrix, new_axes_order)
        
        return result

    def parity_transformation_for_value(self, matrix):
        return np.tensordot(self.P,  matrix, axes=(1, 0))

    def parity_transformation_for_variance(self, matrix):
        sigma_state = sequenceCalculator._diag_along_axis(matrix, 0)
        result = np.tensordot(self.P, np.moveaxis(np.tensordot(sigma_state, self.P.T, axes=([1], [0])),-1,1), axes=([1], [0]))
        return sequenceCalculator._extract_diagonal_along_axes(result, 0)

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
        
    def _create_superblock_state_dict(self):
        # Initialize the superblock_state_dict
        self.superblock_state_dict = {}

        # List of quantity pairs for error propagation
        quantity_pairs = [
            ('C', 'dC2', 'degen_block_red_chi_square_C'),
            ('phi', 'dphi2', 'degen_block_red_chi_square_phi'),
            ('tau', 'dtau2', 'degen_block_red_chi_square_tau'),
            ('omega', 'domega2', 'degen_block_red_chi_square_omega'),
            ('C_summary', 'dC2_summary', 'degen_block_red_chi_square_C_summary'),
            ('phi_summary', 'dphi2_summary', 'degen_block_red_chi_square_phi_summary'),
            ('tau_summary', 'dtau2_summary', 'degen_block_red_chi_square_tau_summary'),
            ('omega_summary', 'domega2_summary', 'degen_block_red_chi_square_omega_summary'),
            ('A', 'dA2', 'degen_block_red_chi_square_A'),
            ('Ap', 'dAp2', 'degen_block_red_chi_square_Ap'),
            ('Am', 'dAm2', 'degen_block_red_chi_square_Am'),
            ('A_summary', 'dA2_summary', 'degen_block_red_chi_square_A_summary'),
            ('Ap_summary', 'dAp2_summary', 'degen_block_red_chi_square_Ap_summary'),
            ('Am_summary', 'dAm2_summary', 'degen_block_red_chi_square_Am_summary')
        ]

        # Process the blockdict for each quantity
        for quantity1, quantity2, red_chi_square_quantity in quantity_pairs:
            if quantity1 not in self.superblock_state_dict:
                self.superblock_state_dict[quantity1] = {}
            if quantity2 not in self.superblock_state_dict:
                self.superblock_state_dict[quantity2] = {}
            if red_chi_square_quantity not in self.superblock_state_dict:
                self.superblock_state_dict[red_chi_square_quantity] = {}

            # Iterate over keys for non_parity_switches
            for non_parity_key in self.blockdict[quantity1].keys():
                if non_parity_key not in self.superblock_state_dict[quantity1]:
                    self.superblock_state_dict[quantity1][non_parity_key] = {}
                    self.superblock_state_dict[quantity2][non_parity_key] = {}
                    self.superblock_state_dict[red_chi_square_quantity][non_parity_key] = {}

                # Iterate over keys for superblock_parity_switches
                for superblock_parity_key in self.blockdict[quantity1][non_parity_key].keys():
                    # Stack arrays for quantity1 and quantity2 along axis 0
                    A_list = self.blockdict[quantity1][non_parity_key][superblock_parity_key]
                    dA2_list = self.blockdict[quantity2][non_parity_key][superblock_parity_key]
                    
                    A_stacked = np.stack(A_list, axis=0)
                    dA2_stacked = np.stack(dA2_list, axis=0)

                    # Apply error propagation
                    A_mean, dA_mean_square_unscaled, _, red_chi_square = sequenceCalculator._propagate_error_bar(A_stacked, dA2_stacked, axis_to_take_average=0, nan=False)

                    # Store results in superblock_state_dict
                    self.superblock_state_dict[quantity1][non_parity_key][superblock_parity_key] = A_mean
                    self.superblock_state_dict[quantity2][non_parity_key][superblock_parity_key] = dA_mean_square_unscaled
                    self.superblock_state_dict[red_chi_square_quantity][non_parity_key][superblock_parity_key] = red_chi_square

        # Now handle 'N' and 'N_summary' by summing the arrays
        for quantity in ['N', 'N_summary']:
            if quantity not in self.superblock_state_dict:
                self.superblock_state_dict[quantity] = {}

            # Iterate over keys for non_parity_switches
            for non_parity_key in self.blockdict[quantity].keys():
                if non_parity_key not in self.superblock_state_dict[quantity]:
                    self.superblock_state_dict[quantity][non_parity_key] = {}

                # Iterate over keys for superblock_parity_switches
                for superblock_parity_key in self.blockdict[quantity][non_parity_key].keys():
                    N_list = self.blockdict[quantity][non_parity_key][superblock_parity_key]
                    N_summed = np.sum(N_list, axis=0, keepdims=True)

                    # Store summed result in superblock_state_dict
                    self.superblock_state_dict[quantity][non_parity_key][superblock_parity_key] = N_summed

        # Now handle 'xsipm' and 'xsipm_summary' by averaging the arrays
        for quantity in ['xsipm', 'xsipm_summary']:
            if quantity not in self.superblock_state_dict:
                self.superblock_state_dict[quantity] = {}

            # Iterate over keys for non_parity_switches
            for non_parity_key in self.blockdict[quantity].keys():
                if non_parity_key not in self.superblock_state_dict[quantity]:
                    self.superblock_state_dict[quantity][non_parity_key] = {}

                # Iterate over keys for superblock_parity_switches
                for superblock_parity_key in self.blockdict[quantity][non_parity_key].keys():
                    xsipm_list = self.blockdict[quantity][non_parity_key][superblock_parity_key]
                    xsipm_mean = np.mean(xsipm_list, axis=0, keepdims=True)

                    # Store averaged result in superblock_state_dict
                    self.superblock_state_dict[quantity][non_parity_key][superblock_parity_key] = xsipm_mean
        
        # Now handle 'ysipm' and 'ysipm_summary' by averaging the arrays
        for quantity in ['ysipm', 'ysipm_summary']:
            if quantity not in self.superblock_state_dict:
                self.superblock_state_dict[quantity] = {}

            # Iterate over keys for non_parity_switches
            for non_parity_key in self.blockdict[quantity].keys():
                if non_parity_key not in self.superblock_state_dict[quantity]:
                    self.superblock_state_dict[quantity][non_parity_key] = {}

                # Iterate over keys for superblock_parity_switches
                for superblock_parity_key in self.blockdict[quantity][non_parity_key].keys():
                    ysipm_list = self.blockdict[quantity][non_parity_key][superblock_parity_key]
                    ysipm_mean = np.mean(ysipm_list, axis=0, keepdims=True)

                    # Store averaged result in superblock_state_dict
                    self.superblock_state_dict[quantity][non_parity_key][superblock_parity_key] = ysipm_mean

        # Now handle 'zsipm' and 'zsipm_summary' by averaging the arrays
        for quantity in ['zsipm', 'zsipm_summary']:
            if quantity not in self.superblock_state_dict:
                self.superblock_state_dict[quantity] = {}

            # Iterate over keys for non_parity_switches
            for non_parity_key in self.blockdict[quantity].keys():
                if non_parity_key not in self.superblock_state_dict[quantity]:
                    self.superblock_state_dict[quantity][non_parity_key] = {}

                # Iterate over keys for superblock_parity_switches
                for superblock_parity_key in self.blockdict[quantity][non_parity_key].keys():
                    zsipm_list = self.blockdict[quantity][non_parity_key][superblock_parity_key]
                    zsipm_mean = np.mean(zsipm_list, axis=0, keepdims=True)

                    # Store averaged result in superblock_state_dict
                    self.superblock_state_dict[quantity][non_parity_key][superblock_parity_key] = zsipm_mean

        for quantity in ['yzsipm', 'yzsipm_summary']:
            if quantity not in self.superblock_state_dict:
                self.superblock_state_dict[quantity] = {}

            # Iterate over keys for non_parity_switches
            for non_parity_key in self.blockdict[quantity].keys():
                if non_parity_key not in self.superblock_state_dict[quantity]:
                    self.superblock_state_dict[quantity][non_parity_key] = {}

                # Iterate over keys for superblock_parity_switches
                for superblock_parity_key in self.blockdict[quantity][non_parity_key].keys():
                    yzsipm_list = self.blockdict[quantity][non_parity_key][superblock_parity_key]
                    yzsipm_mean = np.mean(yzsipm_list, axis=0, keepdims=True)

                    # Store averaged result in superblock_state_dict
                    self.superblock_state_dict[quantity][non_parity_key][superblock_parity_key] = yzsipm_mean

    def _create_sequence_results(self):
        # Initialize sequence_result and sequence_result_summary dictionaries
        self.sequence_result = {}
        self.sequence_result_summary = {}
        self.final_result = {}

        # Explicitly list the quantities for sequence_result and sequence_result_summary
        quantities_for_value_transformation = ['C', 'phi', 'omega', 'tau', 'N', 'A', 'xsipm', 'ysipm', 'zsipm', 'yzsipm','Ap', 'Am']
        quantities_for_value_transformation_summary = ['C_summary', 'phi_summary', 'omega_summary', 'tau_summary', 'N_summary','A_summary', 'xsipm_summary', 'ysipm_summary', 'zsipm_summary' , 'yzsipm_summary','Ap_summary', 'Am_summary']


        quantities_for_variance_transformation = ['dC2', 'dphi2', 'domega2', 'dtau2', 'dA2', 'dAp2', 'dAm2']
        quantities_for_variance_transformation_summary = ['dC2_summary', 'dphi2_summary', 'domega2_summary', 'dtau2_summary', 'dA2_summary','dAp2_summary', 'dAm2_summary']


        failed_non_parity_keys = []

        # Process the superblock_state_dict for each quantity
        for quantity in self.superblock_state_dict:
            # Determine whether the quantity belongs to sequence_result or sequence_result_summary
            if quantity in quantities_for_value_transformation_summary:
                result_dict = self.sequence_result_summary
            elif quantity in quantities_for_variance_transformation_summary:
                result_dict = self.sequence_result_summary
            elif quantity in quantities_for_value_transformation:
                result_dict = self.sequence_result
            elif quantity in quantities_for_variance_transformation:
                result_dict = self.sequence_result
            else:
                continue  # Skip if the quantity is not listed explicitly

            # Initialize the dictionary for this quantity in the result_dict
            if quantity not in result_dict:
                result_dict[quantity] = {}

            # Iterate over non_parity_switches keys
            for non_parity_key in self.superblock_state_dict[quantity].keys():
                # Get the superblock parity switch states as a sorted list of tuples in descending order
                superblock_keys_sorted = sorted(self.superblock_state_dict[quantity][non_parity_key].keys(), reverse=True)

                # Stack all matrices corresponding to these superblock states along a new axis (axis 0)
                stacked_matrix = np.stack(
                    [self.superblock_state_dict[quantity][non_parity_key][superblock_key] for superblock_key in superblock_keys_sorted],
                    axis=0
                )

                # Apply the necessary transformations based on the quantity
                if quantity in quantities_for_value_transformation or quantity in quantities_for_value_transformation_summary:
                    try:
                        transformed_matrix = self.parity_transformation_for_value(stacked_matrix)
                    except Exception as e:
                        failed_non_parity_keys.append(non_parity_key)
                        continue
                elif quantity in quantities_for_variance_transformation or quantity in quantities_for_variance_transformation_summary:
                    try:
                        transformed_matrix = self.parity_transformation_for_variance(stacked_matrix)
                    except Exception as e:
                        failed_non_parity_keys.append(non_parity_key)
                        continue
                else:
                    transformed_matrix = stacked_matrix  # No transformation needed for other quantities

                # Store the transformed matrix in the result dictionary
                result_dict[quantity][non_parity_key] = transformed_matrix

        # Remove failed non_parity_keys from the sequence_result and sequence_result_summary dictionaries
        for non_parity_key in failed_non_parity_keys:
            for quantity in self.sequence_result:
                if non_parity_key in self.sequence_result[quantity]:
                    del self.sequence_result[quantity][non_parity_key]
            for quantity in self.sequence_result_summary:
                if non_parity_key in self.sequence_result_summary[quantity]:
                    del self.sequence_result_summary[quantity][non_parity_key]

        # Now generate the final result
        for quantity_value, quantity_variance in [
            ('C_summary', 'dC2_summary'),
            ('phi_summary', 'dphi2_summary'),
            ('omega_summary', 'domega2_summary'),
            ('tau_summary', 'dtau2_summary'),
            ('A_summary', 'dA2_summary'),
            ('Ap_summary', 'dAp2_summary'),
            ('Am_summary', 'dAm2_summary')
        ]:
            if quantity_value in self.sequence_result_summary and quantity_variance in self.sequence_result_summary:
                for non_parity_key in self.sequence_result_summary[quantity_value]:
                    A_summary = self.sequence_result_summary[quantity_value][non_parity_key]
                    dA2_summary = self.sequence_result_summary[quantity_variance][non_parity_key]

                    # Apply error propagation on axis -2
                    A_mean, dA_mean_square_unscaled, _, red_chi_square = sequenceCalculator._propagate_error_bar(A_summary, dA2_summary, axis_to_take_average=-2, nan=False)

                    # Store the results in final_result, removing the '_summary' suffix
                    quantity = quantity_value.replace('_summary', '')
                    variance_quantity = quantity_variance.replace('_summary', '')
                    self.final_result[quantity] = self.final_result.get(quantity, {})
                    self.final_result[variance_quantity] = self.final_result.get(variance_quantity, {})
                    self.final_result[f'sipm_red_chi_square_{quantity}'] = self.final_result.get(f'sipm_red_chi_square_{quantity}', {})

                    self.final_result[quantity][non_parity_key] = A_mean
                    self.final_result[variance_quantity][non_parity_key] = dA_mean_square_unscaled
                    self.final_result[f'sipm_red_chi_square_{quantity}'][non_parity_key] = red_chi_square

        # Sum 'N_summary' along axis -2 and store in final_result as 'N'
        if 'N_summary' in self.sequence_result_summary:
            for non_parity_key in self.sequence_result_summary['N_summary']:
                N_summary = self.sequence_result_summary['N_summary'][non_parity_key]
                N_summed = np.sum(N_summary, axis=-2, keepdims = True)
                self.final_result['N'] = self.final_result.get('N', {})
                self.final_result['N'][non_parity_key] = N_summed
                
        # store xsipm, ysipm, zsipm _ summary in final result:
        for quantity in ['xsipm_summary', 'ysipm_summary', 'zsipm_summary', 'yzsipm_summary']:
            if quantity in self.sequence_result_summary:
                for non_parity_key in self.sequence_result_summary[quantity]:
                    self.final_result[quantity] = self.final_result.get(quantity, {})
                    self.final_result[quantity][non_parity_key] = self.sequence_result_summary[quantity][non_parity_key]

    def _deep_copy_results(self):
        # Deep copy sequence_result, sequence_result_summary, and superblock_state_dict into __unblinded and blinded
        self.sequenceresult.blinded.result = copy.deepcopy(self.sequence_result)
        self.sequenceresult.blinded.result_summary = copy.deepcopy(self.sequence_result_summary)
        self.sequenceresult.blinded.superblock_state_result = copy.deepcopy(self.superblock_state_dict)
        self.sequenceresult.blinded.final_result = copy.deepcopy(self.final_result)

        # Access the protected __unblinded attribute correctly
        self.sequenceresult._SequenceResults__unblinded.result = copy.deepcopy(self.sequence_result)
        self.sequenceresult._SequenceResults__unblinded.result_summary = copy.deepcopy(self.sequence_result_summary)
        self.sequenceresult._SequenceResults__unblinded.superblock_state_result = copy.deepcopy(self.superblock_state_dict)
        self.sequenceresult._SequenceResults__unblinded.final_result = copy.deepcopy(self.final_result)

    def _apply_blinding(self):
        # Blinding for superblock_state_result (handling both quantity and quantity_summary)
        for quantity in ['omega', 'phi', 'omega_summary', 'phi_summary', 'A', 'A_summary']:
            if quantity in self.sequenceresult.blinded.superblock_state_result:
                for non_parity_key in self.sequenceresult.blinded.superblock_state_result[quantity].keys():
                    for superblock_key in self.sequenceresult.blinded.superblock_state_result[quantity][non_parity_key]:
                        matrix = self.sequenceresult.blinded.superblock_state_result[quantity][non_parity_key][superblock_key]

                        if 'omega' == quantity:
                            # Shift omega and omega_summary[:, 4, ...] by blind value
                            matrix[:, 4, ...] += self.blind._blind_value_in_rad_s
                        elif 'phi' == quantity:
                            # Shift phi and phi_summary[:, 4, ...] by tau[:, 0, ...] * blind value
                            tau_matrix = self.sequenceresult.blinded.superblock_state_result['tau'][non_parity_key][superblock_key]
                            matrix[:, 4, ...] += tau_matrix[:, 0, ...] * self.blind._blind_value_in_rad_s
                        elif 'A' == quantity:
                            # Shift phi and phi_summary[:, 4, ...] by tau[:, 0, ...] * blind value
                            tau_matrix = self.sequenceresult.blinded.superblock_state_result['tau'][non_parity_key][superblock_key]
                            matrix[:, 4, ...] += tau_matrix[:, 0, ...] * self.blind._blind_value_in_rad_s * 2
                        elif 'omega_summary' == quantity:
                            matrix[:, 4, ...] += self.blind._blind_value_in_rad_s
                        elif 'phi_summary' == quantity:
                            tau_matrix = self.sequenceresult.blinded.superblock_state_result['tau_summary'][non_parity_key][superblock_key]
                            matrix[:, 4, ...] += tau_matrix[:, 0, ...] * self.blind._blind_value_in_rad_s
                        elif 'A_summary' == quantity:
                            tau_matrix = self.sequenceresult.blinded.superblock_state_result['tau_summary'][non_parity_key][superblock_key]
                            matrix[:, 4, ...] += tau_matrix[:, 0, ...] * self.blind._blind_value_in_rad_s * 2


        # Blinding for result and result_summary
        for quantity in ['omega', 'phi', 'A']:
            if quantity in self.sequenceresult.blinded.result:
                for non_parity_key in self.sequenceresult.blinded.result[quantity].keys():
                    matrix = self.sequenceresult.blinded.result[quantity][non_parity_key]

                    if quantity == 'omega':
                        # Shift omega[0, :, 4, ...] by blind value
                        matrix[0, :, 4, ...] += self.blind._blind_value_in_rad_s
                    elif quantity == 'phi':
                        # Shift phi[0, :, 4, ...] by tau[0, :, 0, ...] * blind value
                        tau_matrix = self.sequenceresult.blinded.result['tau'][non_parity_key]
                        matrix[0, :, 4, ...] += tau_matrix[0, :, 0, ...] * self.blind._blind_value_in_rad_s
                    elif quantity == 'A':
                        # Shift phi[0, :, 4, ...] by tau[0, :, 0, ...] * blind value
                        tau_matrix = self.sequenceresult.blinded.result['tau'][non_parity_key]
                        matrix[0, :, 4, ...] += tau_matrix[0, :, 0, ...] * self.blind._blind_value_in_rad_s * 2

        # Blinding for result_summary (same as result)
        for quantity in ['omega_summary', 'phi_summary', 'A_summary']:
            if quantity in self.sequenceresult.blinded.result_summary:
                for non_parity_key in self.sequenceresult.blinded.result_summary[quantity].keys():
                    matrix = self.sequenceresult.blinded.result_summary[quantity][non_parity_key]

                    if quantity == 'omega_summary':
                        # Shift omega_summary[0, :, 4, ...] by blind value
                        matrix[0, :, 4, ...] += self.blind._blind_value_in_rad_s
                    elif quantity == 'phi_summary':
                        # Shift phi_summary[0, :, 4, ...] by tau_summary[0, :, 0, ...] * blind value
                        tau_matrix = self.sequenceresult.blinded.result_summary['tau_summary'][non_parity_key]
                        matrix[0, :, 4, ...] += tau_matrix[0, :, 0, ...] * self.blind._blind_value_in_rad_s
                    elif quantity == 'A_summary':
                        # Shift phi_summary[0, :, 4, ...] by tau_summary[0, :, 0, ...] * blind value
                        tau_matrix = self.sequenceresult.blinded.result_summary['tau_summary'][non_parity_key]
                        matrix[0, :, 4, ...] += tau_matrix[0, :, 0, ...] * self.blind._blind_value_in_rad_s * 2

        # Blinding for final_result
        for quantity in ['omega', 'phi', 'A']:
            if quantity in self.sequenceresult.blinded.final_result:
                for non_parity_key in self.sequenceresult.blinded.final_result[quantity]:
                    matrix = self.sequenceresult.blinded.final_result[quantity][non_parity_key]

                    if quantity == 'omega':
                        # Shift omega[:, 4, ...] by blind value
                        matrix[0, :, 4, ...] += self.blind._blind_value_in_rad_s
                    elif quantity == 'phi':
                        # Shift phi[:, 4, ...] by tau[:, 0, ...] * blind value
                        tau_matrix = self.sequenceresult.blinded.final_result['tau'][non_parity_key]
                        matrix[0 ,:, 4, ...] += tau_matrix[0, :, 0, ...] * self.blind._blind_value_in_rad_s
                    elif quantity == 'A':
                        # Shift phi[:, 4, ...] by tau[:, 0, ...] * blind value
                        tau_matrix = self.sequenceresult.blinded.final_result['tau'][non_parity_key]
                        matrix[0 ,:, 4, ...] += tau_matrix[0, :, 0, ...] * self.blind._blind_value_in_rad_s * 2

    def _block_df_combine_sipm(superblock_parity_switches, non_parity_switches, df):
        # Define the quantities to average and channels to process
        quantity_for_average = ["C", "phi", "omega", "tau", "A", "Ap", "Am"]
        channels = ["nr", "N", "E", "B", "NE", "NB", "EB", "NEB"]

        # Function to calculate inverse-variance weighted average
        def weighted_average_with_uncertainty(df, quantity, channel):
            sipm_columns = [f'block_{quantity}_{channel}_sipm{i}' for i in range(8) if f'block_{quantity}_{channel}_sipm{i}' in df.columns]
            uncertainty_columns = [f'blockuncertainty_{quantity}_{channel}_sipm{i}' for i in range(8) if f'blockuncertainty_{quantity}_{channel}_sipm{i}' in df.columns]
            
            values = df[sipm_columns].values
            uncertainties = df[uncertainty_columns].values
            
            # Inverse of variance (uncertainty squared)
            weights = 1 / (uncertainties ** 2)
            
            # Weighted average
            weighted_avg = np.sum(values * weights, axis=1) / np.sum(weights, axis=1)
            
            # New uncertainty (harmonic sum)
            new_uncertainty = np.sqrt(1 / np.sum(weights, axis=1))
            
            # Add the new columns back to the original DataFrame
            df[f'block_{quantity}_{channel}'] = weighted_avg
            df[f'blockuncertainty_{quantity}_{channel}'] = new_uncertainty

        # Function to sum the block_N values for a channel
        def sum_N_values(df, channel):
            sipm_columns = [f'block_N_{channel}_sipm{i}' for i in range(8) if f'block_N_{channel}_sipm{i}' in df.columns]
            
            # Sum values across SiPMs
            summed_values = df[sipm_columns].sum(axis=1)
            
            # Add the new column back to the original DataFrame
            df[f'block_N_{channel}'] = summed_values

        def average_zsipm_values(df, channel):
            sipm_columns = [f'block_zsipm_{channel}_sipm{i}' for i in range(8) if f'block_zsipm_{channel}_sipm{i}' in df.columns]
            averaged_values = df[sipm_columns].mean(axis=1)
            df[f'block_zsipm_{channel}'] = averaged_values
        
        def average_yzsipm_values(df, channel):
            sipm_columns = [f'block_yzsipm_{channel}_sipm{i}' for i in range(8) if f'block_yzsipm_{channel}_sipm{i}' in df.columns]
            averaged_values = df[sipm_columns].mean(axis=1)
            df[f'block_yzsipm_{channel}'] = averaged_values

        def average_ysipm_values(df, channel):
            sipm_columns = [f'block_ysipm_{channel}_sipm{i}' for i in range(8) if f'block_ysipm_{channel}_sipm{i}' in df.columns]
            averaged_values = df[sipm_columns].mean(axis=1)
            df[f'block_ysipm_{channel}'] = averaged_values

        def average_xsipm_values(df, channel):
            sipm_columns = [f'block_xsipm_{channel}_sipm{i}' for i in range(8) if f'block_xsipm_{channel}_sipm{i}' in df.columns]
            averaged_values = df[sipm_columns].mean(axis=1)
            df[f'block_xsipm_{channel}'] = averaged_values

        # Process all quantities except 'N' quantities for all channels
        for quantity in quantity_for_average:
            for channel in channels:
                if quantity == 'tau' and channel != 'nr':  # Exception for 'tau'
                    continue
                if quantity != 'N' and quantity != 'zsipm' and quantity != 'ysipm' and quantity != 'xsipm' and quantity != 'yzsipm':  # Avoid processing 'N' , 'zsipm', 'xsipm', 'ysipm'quantities
                    weighted_average_with_uncertainty(df, quantity, channel)

        # Process 'N' quantities by summing values without uncertainties
        for channel in channels:
            sum_N_values(df, channel)
        
        for channel in channels:
            average_zsipm_values(df, channel)
            average_ysipm_values(df, channel)
            average_xsipm_values(df, channel)
            average_yzsipm_values(df, channel)

    def _sequence_result_convert_to_dataframe(a):
        # Step 1: Extract non-parity switches and prepare the initial column for them
        non_parity_switches = a.non_parity_switches  # Example: ['Delta_NE', 'Enr']
        
        # Initialize an empty list for rows to build the DataFrame
        rows = []
        all_columns = non_parity_switches.copy()  # Columns will be added dynamically based on the data
        
        # Extract the final_result dictionary and labels from object `a`
        final_result = a.blinded.final_result
        labels = a.labels
        # Step 2: Process each quantity and its variance/chi-square
        for switch_values in next(iter(final_result.values())).keys():  # Iterate through the second-level keys
            row = list(switch_values)  # Start the row with non-parity switch values
            
            # Dynamically create columns and append data based on actual matrix size
            for quantity in ['C', 'phi', 'omega', 'tau', 'A','Ap', 'Am']:
                if quantity in final_result:
                    matrix = final_result[quantity][switch_values]
                    variance_matrix = final_result[f'd{quantity}2'][switch_values]
                    chi_square_matrix = final_result[f'sipm_red_chi_square_{quantity}'][switch_values]
                    
                    # Iterate over the matrix and populate values for the DataFrame
                    for i in range(matrix.shape[0]):
                        for j in range(matrix.shape[2]):
                            label = labels[i][j] if i < len(labels) and j < len(labels[i]) else f"unknown_{i}_{j}"
                            
                            # Quantity value
                            row.append(matrix[i, 0, j, 0, 0, 0, 0])
                            column_name = f'{quantity}_{label}'
                            if column_name not in all_columns:
                                all_columns.append(column_name)
                            
                            # Uncertainty and scaled uncertainty
                            uncertainty = np.sqrt(variance_matrix[i, 0, j, 0, 0, 0, 0])
                            scaled_uncertainty = np.sqrt(variance_matrix[i, 0, j, 0, 0, 0, 0] * chi_square_matrix[i, 0, j, 0, 0, 0, 0])
                            
                            row.append(uncertainty)
                            uncertainty_column_name = f'uncertainty_{quantity}_{label}'
                            if uncertainty_column_name not in all_columns:
                                all_columns.append(uncertainty_column_name)
                            
                            row.append(scaled_uncertainty)
                            scaled_uncertainty_column_name = f'scaleduncertainty_{quantity}_{label}'
                            if scaled_uncertainty_column_name not in all_columns:
                                all_columns.append(scaled_uncertainty_column_name)
            
            # Special case for 'N' (no variance or chi-square)
            if 'N' in final_result:
                matrix_N = final_result['N'][switch_values]
                for i in range(matrix_N.shape[0]):
                    for j in range(matrix_N.shape[2]):
                        label = labels[i][j] if i < len(labels) and j < len(labels[i]) else f"unknown_{i}_{j}"
                        
                        row.append(matrix_N[i, 0, j, 0, 0, 0, 0])
                        column_name = f'N_{label}'
                        if column_name not in all_columns:
                            all_columns.append(column_name)

            # Special case for 'xsipm' (no variance or chi-square)
            if 'xsipm_summary' in final_result:
                matrix_xsipm = final_result['xsipm_summary'][switch_values]
                for i in range(matrix_xsipm.shape[0]):
                    for j in range(matrix_xsipm.shape[2]):
                        label = labels[i][j] if i < len(labels) and j < len(labels[i]) else f"unknown_{i}_{j}"
                        
                        row.append(matrix_xsipm[i, 0, j, 0, 0, 0, 0])
                        column_name = f'xsipm_{label}'
                        if column_name not in all_columns:
                            all_columns.append(column_name)
            
            # Special case for 'ysipm' (no variance or chi-square)
            if 'ysipm_summary' in final_result:
                matrix_ysipm = final_result['ysipm_summary'][switch_values]
                for i in range(matrix_ysipm.shape[0]):
                    for j in range(matrix_ysipm.shape[2]):
                        label = labels[i][j] if i < len(labels) and j < len(labels[i]) else f"unknown_{i}_{j}"
                        
                        row.append(matrix_ysipm[i, 0, j, 0, 0, 0, 0])
                        column_name = f'ysipm_{label}'
                        if column_name not in all_columns:
                            all_columns.append(column_name)
            
            # Special case for 'zsipm' (no variance or chi-square)
            if 'zsipm_summary' in final_result:
                matrix_zsipm = final_result['zsipm_summary'][switch_values]
                for i in range(matrix_zsipm.shape[0]):
                    for j in range(matrix_zsipm.shape[2]):
                        label = labels[i][j] if i < len(labels) and j < len(labels[i]) else f"unknown_{i}_{j}"
                        
                        row.append(matrix_zsipm[i, 0, j, 0, 0, 0, 0])
                        column_name = f'zsipm_{label}'
                        if column_name not in all_columns:
                            all_columns.append(column_name)
                            
            if 'yzsipm_summary' in final_result:
                matrix_yzsipm = final_result['yzsipm_summary'][switch_values]
                for i in range(matrix_yzsipm.shape[0]):
                    for j in range(matrix_yzsipm.shape[2]):
                        label = labels[i][j] if i < len(labels) and j < len(labels[i]) else f"unknown_{i}_{j}"
                        
                        row.append(matrix_yzsipm[i, 0, j, 0, 0, 0, 0])
                        column_name = f'yzsipm_{label}'
                        if column_name not in all_columns:
                            all_columns.append(column_name)

            # Append the row to the list of rows
            rows.append(row)
        
        # Step 3: Convert rows into a DataFrame with dynamically created columns
        df = pd.DataFrame(rows, columns=all_columns)
        
        return df

    def _sequence_result_convert_to_sipm_dataframe(a):
        """
        Build a wide DataFrame from the nested '..._summary' dictionaries in 
        a.sequenceresult.blinded.result_summary, with the logic described.
        """
        
        # --- 1) Extract all _summary keys from result_summary ---
        result_summary = a.sequenceresult.blinded.result_summary
        summary_keys = [k for k in result_summary.keys() if k.endswith("_summary")]

        # --- 2) Parse the keys into base quantities and their corresponding 'dX2' keys ---
        #     e.g. "C_summary" -> base="C"; "dC2_summary" -> base="C"
        #     We store them in a dict: quantities[base] = dict(value_key="C_summary", error_key="dC2_summary")
        
        def parse_base(key):
            """Returns (base_str, is_d2) from a key like 'C_summary', 'dC2_summary', etc."""
            if key.startswith("d") and key.endswith("2_summary"):
                # example: dC2_summary
                base_str = key[1:-len("2_summary")]  # remove leading 'd' and trailing '2_summary'
                return base_str, True
            else:
                # example: C_summary or phi_summary, etc.
                base_str = key[:-len("_summary")]    # remove trailing '_summary'
                return base_str, False

        quantities = {}
        for key in summary_keys:
            base, is_d2 = parse_base(key)
            if base not in quantities:
                quantities[base] = {"value_key": None, "error_key": None}
            if is_d2:
                quantities[base]["error_key"] = key
            else:
                quantities[base]["value_key"] = key

        # --- 3) Collect all possible tuple-keys across all relevant dictionaries ---
        all_tuple_keys = set()
        for base, dct in quantities.items():
            vkey = dct["value_key"]
            if vkey in result_summary:
                all_tuple_keys.update(result_summary[vkey].keys())
            ekey = dct["error_key"]
            if ekey in result_summary and ekey is not None:
                all_tuple_keys.update(result_summary[ekey].keys())
        all_tuple_keys = sorted(all_tuple_keys, key=lambda x: x if isinstance(x, tuple) else (x,))

        # For convenience, get an easy reference to the channel strings:
        # combine_switches(...) returns (some_matrix, channel_matrix),
        # so we want channel_matrix = combine_switches(...)[1]
        # Then channel_matrix[a_idx][b_idx] is the channel name.
        superblock_labels = a.superblock_parity_labels
        block_labels = a.block_parity_labels
        _, channel_matrix = combine_switches(superblock_labels, block_labels)

        # We'll build data in a "dict of columns" style. 
        # First: columns for each of a.non_parity_switches
        columns_data = {}
        non_parity_switches = a.non_parity_switches

        # Initialize columns for the non-parity-switches
        for switch_name in non_parity_switches:
            columns_data[switch_name] = []

        # Because we'll loop through each row in 'all_tuple_keys', 
        # we also need to gather the derived columns for each row.
        
        # A small helper to do a 1/σ^2 weighted average or sum (if no errors)
        def weighted_or_summed_average(values, variances):
            """
            If variances is not None, compute the 1/σ^2 weighted average.
            Otherwise, sum the values (and return None for the 'uncertainty').
            
            Returns (aggregated_value, aggregated_uncertainty)
            If 'variances' is None, aggregated_uncertainty is None.
            """
            values = np.array(values, dtype=float)
            if variances is None:
                # Sum them
                return values.sum(), None
            else:
                # Weighted average
                var_arr = np.array(variances, dtype=float)
                inv_var = 1.0 / var_arr
                wavg = np.sum(values * inv_var) / np.sum(inv_var)
                wvar = 1.0 / np.sum(inv_var)
                return wavg, np.sqrt(wvar)

        # Next, we want to figure out the shape (A, 1, B, 1, 1, C, 1) for each quantity
        # to know how many a_idx, b_idx, c_idx we have.  We'll do that by peeking at
        # one typical row (i.e. the first tuple key) if it exists.  Different base quantities
        # may have different shapes, so we handle that inside each base quantity loop.

        # We'll accumulate rows in a list of dicts, then build a df at the end.
        rows = []

        for tuple_key in all_tuple_keys:
            # Prepare a dictionary for this row
            row_dict = {}

            # Fill in the non-parity-switches columns from the tuple_key
            # The assumption: len(tuple_key) == len(a.non_parity_switches)
            for i, switch_name in enumerate(non_parity_switches):
                if isinstance(tuple_key, tuple):
                    row_dict[switch_name] = tuple_key[i]
                else:
                    # If for some reason the dictionary is keyed by a single non-tuple
                    row_dict[switch_name] = tuple_key

            # Now loop over each recognized base quantity
            for base, dct in quantities.items():
                vkey = dct["value_key"]
                ekey = dct["error_key"]
                if vkey is None:
                    # if there's no real 'value' dictionary for this base, skip
                    continue
                if vkey not in result_summary:
                    # the dictionary doesn't exist, skip
                    continue

                # Does the error dict exist for this tuple_key?
                has_error = (ekey is not None 
                            and ekey in result_summary 
                            and tuple_key in result_summary[ekey])

                # Retrieve the array for the value
                if tuple_key not in result_summary[vkey]:
                    # This tuple_key doesn't exist for this vkey, skip
                    continue

                value_array = result_summary[vkey][tuple_key]
                # shape is presumably (A,1,B,1,1,C,1)
                # Extract shape
                shape = value_array.shape
                # Typically: A = shape[0], B = shape[2], C = shape[5]
                # We'll check them carefully:
                A = shape[0]
                B = shape[2]
                C = shape[5]

                # If we have errors, retrieve the error array:
                if has_error:
                    error_array = result_summary[ekey][tuple_key]
                else:
                    error_array = None

                # We'll gather all the sipm values in a triple nested loop, 
                # then compute sipmavg, zplus, zminus, etc. for each (a_idx,b_idx).

                for a_idx in range(A):
                    for b_idx in range(B):
                        channel_name = channel_matrix[a_idx][b_idx]

                        # Gather the sipm values (and uncertainties if available)
                        sipm_vals = []
                        sipm_vars = [] if has_error else None

                        for c_idx in range(C):
                            val = value_array[a_idx, 0, b_idx, 0, 0, c_idx, 0]
                            sipm_vals.append(val)
                            if has_error:
                                var = error_array[a_idx, 0, b_idx, 0, 0, c_idx, 0]
                                sipm_vars.append(var)

                        # If C=1 => store only _sipmavg
                        # Otherwise store _sipm0,1,... if C>1
                        # If C=8 => also store zplus,zminus,xplus,xminus,yplus,yminus

                        # 1) Possibly store each sipm individually
                        if C > 1:
                            for c_idx, val in enumerate(sipm_vals):
                                col_name = f"{base}_{channel_name}_sipm{c_idx}"
                                row_dict[col_name] = val

                                # store the uncertainty if we have it
                                if has_error:
                                    unc_name = f"uncertainty_{base}_{channel_name}_sipm{c_idx}"
                                    row_dict[unc_name] = np.sqrt(sipm_vars[c_idx])

                        # 2) Compute the "average" or "sum" if no uncertainty
                        #    Weighted average if we do have an uncertainty
                        avg_val, avg_unc = weighted_or_summed_average(sipm_vals, sipm_vars)
                        row_dict[f"{base}_{channel_name}_sipmavg"] = avg_val
                        if has_error and avg_unc is not None:
                            row_dict[f"uncertainty_{base}_{channel_name}_sipmavg"] = avg_unc

                        # 3) If C=8, compute zplus,zminus,xplus,xminus,yplus,yminus
                        #    Indices for zplus = [0,1,2,3], zminus = [4,5,6,7]
                        #    etc. Weighted or summed the same way
                        if C == 8:
                            # Build dictionary of name -> indices
                            group_map = {
                                'zplus':   [0,1,2,3],
                                'zminus':  [4,5,6,7],
                                'yplus':   [0,1,4,5],
                                'yminus':  [2,3,6,7],
                                'xplus':   [0,2,4,6],
                                'xminus':  [1,3,5,7],
                            }
                            for grp_name, idx_list in group_map.items():
                                grp_vals = [sipm_vals[i] for i in idx_list]
                                grp_vars = ([sipm_vars[i] for i in idx_list] 
                                            if has_error else None)
                                grp_val, grp_unc = weighted_or_summed_average(grp_vals, grp_vars)
                                col_name = f"{base}_{channel_name}_{grp_name}"
                                row_dict[col_name] = grp_val
                                if has_error and grp_unc is not None:
                                    unc_col_name = f"uncertainty_{base}_{channel_name}_{grp_name}"
                                    row_dict[unc_col_name] = grp_unc

                        # If C=1, we do not create _sipm0,...  We'll rely on the above logic 
                        # which only created `base_{channel}_sipmavg`.

            # Done building columns for this particular row
            rows.append(row_dict)

        # Convert the list-of-dicts 'rows' into a DataFrame
        df = pd.DataFrame(rows)

        # Because different rows might not have exactly the same columns (some base might not exist),
        # pandas will fill missing columns with NaN as needed. That is usually fine.

        # It's often nice to reorder columns so that the non_parity_switches come first:
        # We'll do that by making a list of non-parity-switch columns plus the rest in sorted order.
        existing_cols = list(df.columns)
        front_cols = [c for c in non_parity_switches if c in existing_cols]
        other_cols = [c for c in existing_cols if c not in front_cols]
        df = df[front_cols + other_cols]

        return df

    def saveresults(self):
        self.sequenceresult.sequencedf = sequenceCalculator._sequence_result_convert_to_dataframe(self.sequenceresult)
        self.sequenceresult.sequencesipmdf = sequenceCalculator._sequence_result_convert_to_sipm_dataframe(self)
        sequenceCalculator._block_df_combine_sipm(self.superblock_parity_switches, self.non_parity_switches, self.sequenceresult.blockdf)

        # Save pickle
        with open(os.path.join(self.sequence_result_folder_path, f'sequenceresult_{self.sequenceresult.sequence_name}.pkl'), 'wb') as f:
            pickle.dump(self.sequenceresult, f)

        # Save CSVs in result folder
        self.sequenceresult.sequencesipmdf.to_csv(
            os.path.join(self.sequence_result_folder_path, f'sequencesipmdf_{self.sequenceresult.sequence_name}.csv'),
            index=False
        )
        self.sequenceresult.blockdf.to_csv(
            os.path.join(self.sequence_result_folder_path, f'sequenceblocks_{self.sequenceresult.sequence_name}.csv'),
            index=False
        )
        self.sequenceresult.sequencedf.to_csv(
            os.path.join(self.sequence_result_folder_path, f'sequencedf_{self.sequenceresult.sequence_name}.csv'),
            index=False
        )

        # Limit filename to 50 characters (excluding the path)
        filename_part = self.sequenceresult.sequence_string[1:]
        if len(filename_part) > 50:
            filename_part = filename_part[:50]

        # Save to multiple_results folder with truncated filename
        self.sequenceresult.blockdf.to_csv(
            os.path.join(r"C:\ACME_analysis\multiple_results\sequenceblock_results", f"{filename_part}.csv"),
            index=False
        )
        self.sequenceresult.sequencedf.to_csv(
            os.path.join(r"C:\ACME_analysis\multiple_results\sequencedf_result", f"{filename_part}.csv"),
            index=False
        )
        self.sequenceresult.sequencesipmdf.to_csv(
            os.path.join(r"C:\ACME_analysis\multiple_results\sequencesipm_results", f"{filename_part}.csv"),
            index=False
        )

    def output_standard_format(self):
        stand_dict = {'blind 5':{}}
        r_dict= stand_dict['blind 5']
        if not os.path.exists(r"C:\standard_format"):
            os.makedirs(r"C:\standard_format")
        for i in range(len(self.labels)):
            for j in range(len(self.labels[0])):
                for (newname, quantity, variance) in [("Contrast", "C", "dC2"), ("Phase_blinded", "phi", "dphi2"), ("Omega_blinded", "omega", "domega2")]:
                    if newname not in r_dict:
                        r_dict[newname] = {}
                    if self.labels[i][j] not in r_dict[newname]:
                        r_dict[newname][self.labels[i][j]] = {}
                    r_dict[newname][self.labels[i][j]]["Mean"] = self.sequenceresult.blinded.final_result[quantity][()][i,0,j,0,0,0,0]
                    r_dict[newname][self.labels[i][j]]["Sigma"] = np.sqrt(self.sequenceresult.blinded.final_result[variance][()][i,0,j,0,0,0,0])
        with open(r"C:\standard_format\standard" + self.sequenceresult.sequence_string +".json", 'w') as f:
            json.dump(stand_dict, f)