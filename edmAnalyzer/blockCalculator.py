from .parityStateTransfrom import *
from .headerHandler import *

import re
import numpy as np
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import warnings
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import json
import pickle

class blockCalculator:
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
            self.full_waveplate_dither_range_in_deg= 2
            self.error_propagation_method= "simple"

        def _load_parameters_from_json(self, parameter_file_path):
            try:
                with open(parameter_file_path, 'r') as f:
                    param_dict = json.load(f)
                
                for key, value in param_dict.items():
                    if hasattr(self, key) and value is not None:
                        setattr(self, key, value)
            except FileNotFoundError:
                print(f"Block Parameter file {parameter_file_path} not found, using default values.")
            except json.JSONDecodeError:
                print(f"Block: Error decoding JSON file {parameter_file_path}, using default values.")
            except Exception as e:
                print(f"Block: Unexpected error: {e}")

    class BlockResults:

        class Blinded:
            def __init__(self):
                self.blind_id = None
                self.result = {}
                self.result_sipmsum = {}    
                self.result_summary = {}

        class Unblinded:
            def __init__(self):
                self.result = {}
                self.result_sipmsum = {}
                self.result_summary = {}

        def __init__(self):
            self.blinded = self.Blinded()
            self.__unblinded = self.Unblinded()
            self.block_string = None
            self.blockcut_left = None
            self.blockcut_right = None
            self.parity_labels = None
            self.state_labels = None
            self.red_chi_square_trace_shot_A = None

    def __init__(self, blockpara_json_path = None, binresults_path_list = None, bincutresult_path_list = None, blockresult_path = None, df = None, phi_B_over_tau = None, blind_path = None):
        self.P, self.parity_labels, self.switch_labels = parityStateTransform(channelName= ['N','E','B'])
        self.blind = None
        self.read_blind(blind_path)
        self.parameters = self.Parameters()
        self.parameters._load_parameters_from_json(blockpara_json_path)
        self.blockresult = self.BlockResults()
        self.blockresult.blinded.blind_id = self.blind.blind_id
        self.blockresult.parity_labels = self.parity_labels
        self.blockresult.state_labels = self.switch_labels
        self.binresults_path_list = binresults_path_list
        self.bincutresult_path_list = bincutresult_path_list
        self.blockresult_path = blockresult_path
        self.df = df
        self.blockcut_mask = None
        self.blockcut_left = None
        self.blockcut_right = None
        self.t = -180/2/np.pi/self.parameters.full_waveplate_dither_range_in_deg
        self.phi_B_over_tau = phi_B_over_tau
        self.aggregated_traces = {}
        self._load_binresults()
        self._load_bincutresults()
        self._pipeline_simple()
        self.save_result()

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
        self.blind =  blockCalculator.blind_object(blind_value_in_rad_s, blind_id)

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

    def _calculate_center_of_mass(arr):
        # Get the shape of the array except the last dimension
        indices = np.arange(arr.shape[-1])  # Create an array of indices for the last dimension
        
        # Multiply the values in the array by the index along the last dimension
        weighted_values = arr * indices
        
        # Sum the weighted values and the original values along the last dimension
        sum_weighted = np.sum(weighted_values, axis=-1, keepdims=True)  # keepdims=True to maintain the last dimension
        sum_values = np.sum(arr, axis=-1, keepdims=True)  # keepdims=True to maintain the last dimension
        
        # Calculate the center of mass
        center_of_mass = sum_weighted / (sum_values + 1e-10)  # To avoid division by zero
        
        return center_of_mass

    def _sipm_vector(arr):
        sipm_matrix = np.array([[ 1,  1,  1,  1],
                  [-1,  1,  1, 1],
                  [ 1, -1,  1, 1],
                  [-1, -1,  1, -1],
                  [ 1,  1, -1, 0],
                  [-1,  1, -1, -1],
                  [ 1, -1, -1, 0],
                  [-1, -1, -1, -1]])
        sipm_matrix = sipm_matrix[0:arr.shape[-2]]
        summed_vector  = np.swapaxes(np.tensordot(arr, sipm_matrix, axes = (-2,0)),-2,-1)
        summed_signal = arr.sum(axis = -2, keepdims=True)
        result = np.nan_to_num(np.divide(summed_vector,summed_signal))
        return result[...,[0],:], result[...,[1],:], result[...,[2],:], result[...,[3],:]

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
        
    def _load_bincutresults(self):
        try:
            grand_mask_list = []
            grand_left_list = []
            grand_right_list = []

            # Iterate over the paths in bincutresult_path_list
            for path in self.bincutresult_path_list:
                # Load the .pkl file
                with open(path, 'rb') as file:
                    bincutresult = pickle.load(file)
                    
                    # Append grand_mask, grand_left, grand_right from each pkl file to their respective lists
                    grand_mask_list.append(bincutresult.grand_mask)
                    grand_left_list.append(bincutresult.grand_left)
                    grand_right_list.append(bincutresult.grand_right)
            
            # Calculate element-wise product for grand_mask
            self.blockcut_mask = np.prod(grand_mask_list, axis=0)

            # Calculate the maximum for grand_left
            self.blockcut_left = np.max(grand_left_list, axis=0)
            self.blockresult.blockcut_left = self.blockcut_left

            # Calculate the minimum for grand_right
            self.blockcut_right = np.min(grand_right_list, axis=0)
            self.blockresult.blockcut_right = self.blockcut_right
            
        except FileNotFoundError as e:
            print(f"Error loading bincut result files: {e}")
        except Exception as e:
            print(f"Unexpected error in _load_bincutresults: {e}")

    def _load_binresults(self):

        # Sort the binresults_path_list lexicographically
        self.binresults_path_list.sort()

        # Regex to extract run, sequence, block, and trace_offset from file names
        pattern = r"binresult_(\d{4})\.(\d{4})\.(\d{4})\.(\d{4})\.pkl"

        # Initialize aggregated_traces if not already done
        if 'A' not in self.aggregated_traces:
            self.aggregated_traces['A'] = {}
        if 'dA2' not in self.aggregated_traces:
            self.aggregated_traces['dA2'] = {}
        if 'N' not in self.aggregated_traces:
            self.aggregated_traces['N'] = {}
        if 'F' not in self.aggregated_traces:
            self.aggregated_traces['F'] = {}

        # Iterate over the sorted binresults_path_list
        for path in self.binresults_path_list:
            match = re.search(pattern, path)
            if not match:
                print(f"Filename format error in {path}")
                continue
            
            run, sequence, block, trace_offset = map(int, match.groups())

            # Load the binresult from the file
            with open(path, 'rb') as file:
                binresult = pickle.load(file)

                # Extract arrays
                A_array = binresult.A  # Assume A is an np array
                dA2_array = binresult.dA2_from_photon  # Assume this is an np array
                N_array = binresult.N  # Assume this is an np array

                # Iterate over the traces in A_array (axis=0)
                for idx in range(A_array.shape[0]):
                    trace = trace_offset + idx
                    
                    # Look up the corresponding row in the dataframe based on run, sequence, block, and trace
                    df_row = self.df[(self.df['run'] == run) & 
                                    (self.df['sequence'] == sequence) & 
                                    (self.df['block'] == block) & 
                                    (self.df['trace'] == trace)]
                    
                    if df_row.empty:
                        print(f"No matching data for run={run}, sequence={sequence}, block={block}, trace={trace}")
                        continue
                    
                    # Extract the N, E, B, theta, phi_B_over_tau columns
                    N_val = df_row['N'].values[0]
                    E_val = df_row['E'].values[0]
                    B_val = df_row['B'].values[0]
                    theta_val = df_row['theta'].values[0]
                    
                    state_tuple = (N_val, E_val, B_val, theta_val)

                    # Initialize dictionary entries if not yet present
                    if state_tuple not in self.aggregated_traces['A']:
                        self.aggregated_traces['A'][state_tuple] = []
                    if state_tuple not in self.aggregated_traces['dA2']:
                        self.aggregated_traces['dA2'][state_tuple] = []
                    if state_tuple not in self.aggregated_traces['N']:
                        self.aggregated_traces['N'][state_tuple] = []
                    if state_tuple not in self.aggregated_traces['F']:
                        self.aggregated_traces['F'][state_tuple] = []

                    # Append data to the respective state in aggregated_traces
                    self.aggregated_traces['A'][state_tuple].append(A_array[idx])
                    self.aggregated_traces['dA2'][state_tuple].append(dA2_array[idx])
                    self.aggregated_traces['N'][state_tuple].append(N_array[idx])
                    self.aggregated_traces['F'][state_tuple].append(binresult.F[idx])

        # Now, stack the lists into arrays along axis=0 for each state in aggregated_traces
        for quantity in ['A', 'dA2', 'N', 'F']:
            for state_tuple, data_list in self.aggregated_traces[quantity].items():
                self.aggregated_traces[quantity][state_tuple] = np.stack(data_list, axis=0)

        for quantity in ['A', 'dA2', 'N', 'F']:
            sorted_states = sorted(self.aggregated_traces[quantity].keys(), reverse=True)
            sorted_arrays = [self.aggregated_traces[quantity][state] for state in sorted_states]
            self.aggregated_traces[quantity] = np.stack(sorted_arrays, axis=0)
        self.blockresult.block_string = str(run).zfill(4)+'.'+str(sequence).zfill(4)+'.'+str(block).zfill(4)

    def _pipeline_simple(self):
        """ In this calculation pipeline, no correlation between anything is considered."""
        A_raw, dA2_raw, _, self.blockresult.red_chi_square_trace_shot_A  = blockCalculator._propagate_error_bar(self.aggregated_traces['A'], self.aggregated_traces['dA2'], axis_to_take_average=(1,2))
        N_raw= self.aggregated_traces['N'].sum(axis = (1,2),  keepdims=True)
        F_raw = self.aggregated_traces['F'].sum(axis = (1,2),  keepdims=True)
        # Calculate phi_state and C_state

        A_raw = A_raw.reshape(-1,2,*A_raw.shape[1:])
        dA2_raw = dA2_raw.reshape(-1,2,*dA2_raw.shape[1:])
        N_raw = N_raw.reshape(-1,2,*N_raw.shape[1:])
        F_raw = F_raw.reshape(-1,2,*F_raw.shape[1:])

        N_state = N_raw[:,0,...] + N_raw[:,1,...]

        N_state_summary = N_state[..., self.blockcut_left:self.blockcut_right].sum(axis = -1, keepdims=True)
        xsipm_state_summary,ysipm_state_summary,zsipm_state_summary, yzsipm_state_summary = blockCalculator._sipm_vector(N_state_summary)


        envelop_phi_state_all_sipm = (np.diff(N_state.sum(axis = -2, keepdims = True), axis = -1)/N_state.sum(axis = -2, keepdims = True)[...,1:])/20/2/2/2
        envelop_phi_state = (np.diff(N_state, axis = -1)/N_state[...,1:])/20/2/2/2
        center_of_mass_state = blockCalculator._calculate_center_of_mass(N_state)
        xsipm_state,ysipm_state,zsipm_state, yzsipm_state = blockCalculator._sipm_vector(N_state)
        sipmratio_state = N_state/N_state.sum(axis = -2, keepdims=True)

        N = np.tensordot(self.P, N_state, axes = (1,0))
        envelop_phi_all_sipm = np.tensordot(self.P, envelop_phi_state_all_sipm, axes = (1,0))
        envelop_phi = np.tensordot(self.P, envelop_phi_state, axes = (1,0))

        xsipm = np.tensordot(self.P, xsipm_state, axes = (1,0))
        ysipm = np.tensordot(self.P, ysipm_state, axes = (1,0))
        zsipm = np.tensordot(self.P, zsipm_state, axes = (1,0))
        yzsipm = np.tensordot(self.P, yzsipm_state, axes = (1,0))
        sipmratio = np.tensordot(self.P, sipmratio_state, axes = (1,0))    


        center_of_mass = np.tensordot(self.P, center_of_mass_state, axes = (1,0))

        C_state = self.t / 2 * (A_raw[:,0,...] - A_raw[:,1,...])
        s = np.sign(C_state)
        A_state = s / 2 * (A_raw[:,0,...] + A_raw[:,1,...])
        Ap_state = 1/2 * (A_raw[:,0,...] + A_raw[:,1,...])

        Am_state = 1/2 * (A_raw[:,0,...] - A_raw[:,1,...]) 
        FA_state = s/2 * (F_raw[:,0,...] + F_raw[:,1,...])
        FC_state = self.t / 2 * (F_raw[:,0,...] - F_raw[:,1,...])

        dAp2_state = 1.0/4.0 * (dA2_raw[:,0,...] + dA2_raw[:,1,...])
        dAm2_state = 1.0/4.0 * (dA2_raw[:,0,...] + dA2_raw[:,1,...])    
        
        dA2_state = (s / 2) ** 2 * (dA2_raw[:,0,...] + dA2_raw[:,1,...])
        dC2_state = (self.t / 2) ** 2 * (dA2_raw[:,0,...] + dA2_raw[:,1,...])
        covAC_state = s * self.t/ 4 *  (dA2_raw[:,0,...] - dA2_raw[:,1,...])
        #phi_state = 1 / 2 * A_state / np.abs(C_state) * (1 + d_C_state_2 / C_state**2) - 1 / 2 * 4 * covA_state_C_state * s / C_state**2
        phi_state = 1 / 2 * A_state / np.abs(C_state) * 1
        #dphi2_state = phi_state**2 * (d_C_state_2 / C_state**2 + d_A_state_2 / A_state**2) - phi_state * 8 * s * covA_state_C_state / C_state**2
        dphi2_state = phi_state**2 * (dC2_state / C_state**2 + dA2_state / A_state**2)

        # Calculate parity transformation
        dphi2_covariance = np.tensordot(self.P, np.moveaxis(np.tensordot(blockCalculator._diag_along_axis(dphi2_state,0), self.P.T, axes=([1], [0])),-1,1), axes=([1], [0]))
        dphi2 = blockCalculator._extract_diagonal_along_axes(np.tensordot(self.P, np.moveaxis(np.tensordot(blockCalculator._diag_along_axis(dphi2_state,0), self.P.T, axes=([1], [0])),-1,1), axes=([1], [0])), 0)
        dC2 = blockCalculator._extract_diagonal_along_axes(np.tensordot(self.P, np.moveaxis(np.tensordot(blockCalculator._diag_along_axis(dC2_state,0), self.P.T, axes=([1], [0])),-1,1), axes=([1], [0])), 0)

        phi = np.tensordot(self.P, phi_state, axes = (1,0))
        C = np.tensordot(self.P, C_state, axes = (1,0))
        A = np.tensordot(self.P, A_state, axes = (1,0))
        Ap = np.tensordot(self.P, Ap_state, axes = (1,0))

        Am = np.tensordot(self.P, Am_state, axes = (1,0))
        FA = np.tensordot(self.P, FA_state, axes = (1,0))
        FC = np.tensordot(self.P, FC_state, axes = (1,0))
        dA2 = blockCalculator._extract_diagonal_along_axes(np.tensordot(self.P, np.moveaxis(np.tensordot(blockCalculator._diag_along_axis(dA2_state,0), self.P.T, axes=([1], [0])),-1,1), axes=([1], [0])), 0)
        # Calculate the tau
        dAp2 = blockCalculator._extract_diagonal_along_axes(np.tensordot(self.P, np.moveaxis(np.tensordot(blockCalculator._diag_along_axis(dAp2_state,0), self.P.T, axes=([1], [0])),-1,1), axes=([1], [0])), 0)
        dAm2 = blockCalculator._extract_diagonal_along_axes(np.tensordot(self.P, np.moveaxis(np.tensordot(blockCalculator._diag_along_axis(dAm2_state,0), self.P.T, axes=([1], [0])),-1,1), axes=([1], [0])), 0)
        tau = phi[3].reshape(-1,*phi[3].shape[0:])/self.phi_B_over_tau
        dtau2 = dphi2[3].reshape(-1,*dphi2[3].shape[0:])/self.phi_B_over_tau**2

        # Calculate the omega 

        omega = phi/tau
        domega2 = omega**2 *(dphi2/phi**2 + dtau2/tau**2)

        self.blockresult._BlockResults__unblinded.result = {'N': N, 
                                                            'C': C, 
                                                            'A': A,
                                                            'dA2': dA2,
                                                            'dC2': dC2,
                                                            'phi': phi, 
                                                            'dphi2': dphi2, 
                                                            'tau': tau, 
                                                            'dtau2': dtau2, 
                                                            'omega': omega, 
                                                            'domega2': domega2,
                                                            'xsipm': xsipm,
                                                            'ysipm': ysipm,
                                                            'zsipm': zsipm,
                                                            'yzsipm': yzsipm,
                                                            'sipmratio': sipmratio,
                                                            'envelop_phi_all_sipm': envelop_phi_all_sipm,
                                                            'envelop_phi': envelop_phi,
                                                            'FA': FA,
                                                            'FC': FC,

                                                            'Ap': Ap,

                                                            'Am': Am,

                                                            'dAp2': dAp2,

                                                            'dAm2': dAm2}
                                                            
                                                            
        
        N_sipmsum = N.sum(axis = -2, keepdims=True)
        FA_sipmsum = FA.sum(axis = -2, keepdims=True)
        FC_sipmsum = FC.sum(axis = -2, keepdims=True)
        C_sipmsum, dC2_sipmsum, _, redchigroupC_sipmsum = blockCalculator._propagate_error_bar(C, dC2, axis_to_take_average = -2)
        A_sipmsum, dA2_sipmsum, _, redchigroupA_sipmsum = blockCalculator._propagate_error_bar(A, dA2, axis_to_take_average = -2)
        phi_sipmsum, dphi2_sipmsum, _, redchigroupphi_sipmsum = blockCalculator._propagate_error_bar(phi, dphi2, axis_to_take_average = -2)
        tau_sipmsum, dtau2_sipmsum, _, redchigrouptau_sipmsum = blockCalculator._propagate_error_bar(tau, dtau2, axis_to_take_average = -2)
        omega_sipmsum, domega2_sipmsum, _, redchigroupomega_sipmsum = blockCalculator._propagate_error_bar(omega, domega2, axis_to_take_average = -2)
        Ap_sipmsum, dAp2_sipmsum, _, _ = blockCalculator._propagate_error_bar(Ap, dAp2, axis_to_take_average = -2)
        Am_sipmsum, dAm2_sipmsum, _, _ = blockCalculator._propagate_error_bar(Am, dAm2, axis_to_take_average = -2)

        self.blockresult._BlockResults__unblinded.result_sipmsum = {'N_sipmsum': N_sipmsum,
                                                                    'C_sipmsum': C_sipmsum,
                                                                    'A_sipmsum': A_sipmsum,
                                                                    'dA2_sipmsum': dA2_sipmsum,
                                                                    'redchigroupA_sipmsum': redchigroupA_sipmsum,
                                                                    'dC2_sipmsum': dC2_sipmsum,
                                                                    'phi_sipmsum': phi_sipmsum,
                                                                    'dphi2_sipmsum': dphi2_sipmsum,
                                                                    'tau_sipmsum': tau_sipmsum,
                                                                    'dtau2_sipmsum': dtau2_sipmsum,
                                                                    'omega_sipmsum': omega_sipmsum,
                                                                    'domega2_sipmsum': domega2_sipmsum,
                                                                    'redchigroupC_sipmsum': redchigroupC_sipmsum,
                                                                    'redchigroupphi_sipmsum': redchigroupphi_sipmsum,
                                                                    'redchigrouptau_sipmsum': redchigrouptau_sipmsum,
                                                                    'redchigroupomega_sipmsum': redchigroupomega_sipmsum,
                                                                    'FA_sipmsum': FA_sipmsum,
                                                                    'FC_sipmsum': FC_sipmsum,
                                                                    'Ap_sipmsum': Ap_sipmsum,
                                                                    'dAp2_sipmsum': dAp2_sipmsum,
                                                                    'Am_sipmsum': Am_sipmsum,
                                                                    'dAm2_sipmsum': dAm2_sipmsum}

        N_summary = N[...,self.blockcut_left:self.blockcut_right].sum(axis = -1, keepdims=True)
        FA_summary = FA[...,self.blockcut_left:self.blockcut_right].sum(axis = -1, keepdims=True)
        FC_summary = FC[...,self.blockcut_left:self.blockcut_right].sum(axis = -1, keepdims=True)
        C_summary, dC2_summary, _, redchigroupC_summary = blockCalculator._propagate_error_bar(C[...,self.blockcut_left:self.blockcut_right], dC2[...,self.blockcut_left:self.blockcut_right], axis_to_take_average = -1)
        phi_summary, dphi2_summary, _, redchigroupphi_summary = blockCalculator._propagate_error_bar(phi[...,self.blockcut_left:self.blockcut_right], dphi2[...,self.blockcut_left:self.blockcut_right], axis_to_take_average = -1)
        tau_summary, dtau2_summary, _, redchigrouptau_summary = blockCalculator._propagate_error_bar(tau[...,self.blockcut_left:self.blockcut_right], dtau2[...,self.blockcut_left:self.blockcut_right], axis_to_take_average = -1)
        omega_summary, domega2_summary, _, redchigroupomega_summary = blockCalculator._propagate_error_bar(omega[...,self.blockcut_left:self.blockcut_right], domega2[...,self.blockcut_left:self.blockcut_right], axis_to_take_average = -1)
        A_summary, dA2_summary, _, redchigroupA_summary = blockCalculator._propagate_error_bar(A[...,self.blockcut_left:self.blockcut_right], dA2[...,self.blockcut_left:self.blockcut_right], axis_to_take_average = -1)
        Ap_summary, dAp2_summary, _, _ = blockCalculator._propagate_error_bar(Ap[...,self.blockcut_left:self.blockcut_right], dAp2[...,self.blockcut_left:self.blockcut_right], axis_to_take_average = -1)
        Am_summary, dAm2_summary, _, _ = blockCalculator._propagate_error_bar(Am[...,self.blockcut_left:self.blockcut_right], dAm2[...,self.blockcut_left:self.blockcut_right], axis_to_take_average = -1)

        xsipm_summary = np.tensordot(self.P, xsipm_state_summary, axes = (1,0))
        ysipm_summary = np.tensordot(self.P, ysipm_state_summary, axes = (1,0))
        zsipm_summary = np.tensordot(self.P, zsipm_state_summary, axes = (1,0))
        yzsipm_summary = np.tensordot(self.P, yzsipm_state_summary, axes = (1,0))

        self.blockresult._BlockResults__unblinded.result_summary = {'N_summary': N_summary, 
                                               'C_summary': C_summary, 
                                               'dC2_summary': dC2_summary, 
                                               'phi_summary': phi_summary, 
                                               'dphi2_summary': dphi2_summary, 
                                               'tau_summary': tau_summary, 
                                               'dtau2_summary': dtau2_summary, 
                                               'omega_summary': omega_summary, 
                                               'domega2_summary': domega2_summary,
                                               'centerofmass': center_of_mass,
                                               'xsipm_summary': xsipm_summary,
                                               'ysipm_summary': ysipm_summary,
                                               'zsipm_summary': zsipm_summary,
                                               'yzsipm_summary': yzsipm_summary,
                                               'redchigroupC_summary': redchigroupC_summary,
                                                'redchigroupphi_summary': redchigroupphi_summary,
                                                'redchigrouptau_summary': redchigrouptau_summary,
                                                'redchigroupomega_summary': redchigroupomega_summary,
                                                'A_summary': A_summary,
                                                'dA2_summary': dA2_summary,
                                                'redchigroupA_summary': redchigroupA_summary,
                                                'FA_summary': FA_summary,
                                                'FC_summary': FC_summary,
                                                'Ap_summary': Ap_summary,
                                                'dAp2_summary': dAp2_summary,
                                                'Am_summary': Am_summary,
                                                'dAm2_summary': dAm2_summary}
        
        self.blind_result()
        
    def blind_result(self):
        self.blockresult.blinded.result['N'] = self.blockresult._BlockResults__unblinded.result['N'].copy()
        self.blockresult.blinded.result['FA'] = self.blockresult._BlockResults__unblinded.result['FA'].copy()
        self.blockresult.blinded.result['FC'] = self.blockresult._BlockResults__unblinded.result['FC'].copy()
        self.blockresult.blinded.result['C'] = self.blockresult._BlockResults__unblinded.result['C'].copy()
        self.blockresult.blinded.result['dC2'] = self.blockresult._BlockResults__unblinded.result['dC2'].copy()

        self.blockresult.blinded.result['A'] = self.blockresult._BlockResults__unblinded.result['A'].copy()
        self.blockresult.blinded.result['A'][4] = self.blockresult.blinded.result['A'][4] + self.blind._blind_value_in_rad_s * 0.005 * 2

        self.blockresult.blinded.result['dA2'] = self.blockresult._BlockResults__unblinded.result['dA2'].copy()
        
        self.blockresult.blinded.result['Ap'] = self.blockresult._BlockResults__unblinded.result['Ap'].copy()
        self.blockresult.blinded.result['dAp2'] = self.blockresult._BlockResults__unblinded.result['dAp2'].copy()

        self.blockresult.blinded.result['Am'] = self.blockresult._BlockResults__unblinded.result['Am'].copy()
        self.blockresult.blinded.result['dAm2'] = self.blockresult._BlockResults__unblinded.result['dAm2'].copy()

        self.blockresult.blinded.result['phi'] = self.blockresult._BlockResults__unblinded.result['phi'].copy()
        self.blockresult.blinded.result['phi'][4] = self.blockresult.blinded.result['phi'][4] + self.blind._blind_value_in_rad_s * self.blockresult._BlockResults__unblinded.result['tau'][0]

        self.blockresult.blinded.result['dphi2'] = self.blockresult._BlockResults__unblinded.result['dphi2'].copy()

        self.blockresult.blinded.result['tau'] = self.blockresult._BlockResults__unblinded.result['tau'].copy()
        self.blockresult.blinded.result['dtau2'] = self.blockresult._BlockResults__unblinded.result['dtau2'].copy()

        self.blockresult.blinded.result['omega'] = self.blockresult._BlockResults__unblinded.result['omega'].copy()
        self.blockresult.blinded.result['omega'][4] = self.blockresult.blinded.result['omega'][4] + self.blind._blind_value_in_rad_s

        self.blockresult.blinded.result['domega2'] = self.blockresult._BlockResults__unblinded.result['domega2'].copy()

        self.blockresult.blinded.result['xsipm'] = self.blockresult._BlockResults__unblinded.result['xsipm'].copy()
        self.blockresult.blinded.result['ysipm'] = self.blockresult._BlockResults__unblinded.result['ysipm'].copy()
        self.blockresult.blinded.result['zsipm'] = self.blockresult._BlockResults__unblinded.result['zsipm'].copy()
        self.blockresult.blinded.result['yzsipm'] = self.blockresult._BlockResults__unblinded.result['yzsipm'].copy()
        self.blockresult.blinded.result['sipmratio'] = self.blockresult._BlockResults__unblinded.result['sipmratio'].copy()

        self.blockresult.blinded.result['envelop_phi_all_sipm'] = self.blockresult._BlockResults__unblinded.result['envelop_phi_all_sipm'].copy()
        self.blockresult.blinded.result['envelop_phi'] = self.blockresult._BlockResults__unblinded.result['envelop_phi'].copy()

        self.blockresult.blinded.result_summary['N_summary'] = self.blockresult._BlockResults__unblinded.result_summary['N_summary'].copy()
        self.blockresult.blinded.result_summary['A_summary'] = self.blockresult._BlockResults__unblinded.result_summary['A_summary'].copy()
        self.blockresult.blinded.result_summary['A_summary'][4] = self.blockresult.blinded.result_summary['A_summary'][4] + self.blind._blind_value_in_rad_s * 0.005 * 2
        self.blockresult.blinded.result_summary['dA2_summary'] = self.blockresult._BlockResults__unblinded.result_summary['dA2_summary'].copy()

        self.blockresult.blinded.result_summary['Ap_summary'] = self.blockresult._BlockResults__unblinded.result_summary['Ap_summary'].copy()
        self.blockresult.blinded.result_summary['dAp2_summary'] = self.blockresult._BlockResults__unblinded.result_summary['dAp2_summary'].copy()

        self.blockresult.blinded.result_summary['Am_summary'] = self.blockresult._BlockResults__unblinded.result_summary['Am_summary'].copy()
        self.blockresult.blinded.result_summary['dAm2_summary'] = self.blockresult._BlockResults__unblinded.result_summary['dAm2_summary'].copy()

        self.blockresult.blinded.result_summary['FC_summary'] = self.blockresult._BlockResults__unblinded.result_summary['FC_summary'].copy()
        self.blockresult.blinded.result_summary['FA_summary'] = self.blockresult._BlockResults__unblinded.result_summary['FA_summary'].copy()

        self.blockresult.blinded.result_summary['C_summary'] = self.blockresult._BlockResults__unblinded.result_summary['C_summary'].copy()
        self.blockresult.blinded.result_summary['dC2_summary'] = self.blockresult._BlockResults__unblinded.result_summary['dC2_summary'].copy()

        self.blockresult.blinded.result_summary['phi_summary'] = self.blockresult._BlockResults__unblinded.result_summary['phi_summary'].copy()
        self.blockresult.blinded.result_summary['phi_summary'][4] = self.blockresult.blinded.result_summary['phi_summary'][4] + self.blind._blind_value_in_rad_s * self.blockresult._BlockResults__unblinded.result_summary['tau_summary'][0]

        self.blockresult.blinded.result_summary['dphi2_summary'] = self.blockresult._BlockResults__unblinded.result_summary['dphi2_summary'].copy()

        self.blockresult.blinded.result_summary['tau_summary'] = self.blockresult._BlockResults__unblinded.result_summary['tau_summary'].copy()
        self.blockresult.blinded.result_summary['dtau2_summary'] = self.blockresult._BlockResults__unblinded.result_summary['dtau2_summary'].copy()

        self.blockresult.blinded.result_summary['omega_summary'] = self.blockresult._BlockResults__unblinded.result_summary['omega_summary'].copy()
        self.blockresult.blinded.result_summary['omega_summary'][4] = self.blockresult.blinded.result_summary['omega_summary'][4] + self.blind._blind_value_in_rad_s

        self.blockresult.blinded.result_summary['domega2_summary'] = self.blockresult._BlockResults__unblinded.result_summary['domega2_summary'].copy()

        self.blockresult.blinded.result_summary['centerofmass'] = self.blockresult._BlockResults__unblinded.result_summary['centerofmass'].copy()  

        self.blockresult.blinded.result_summary['xsipm_summary'] = self.blockresult._BlockResults__unblinded.result_summary['xsipm_summary'].copy()
        self.blockresult.blinded.result_summary['ysipm_summary'] = self.blockresult._BlockResults__unblinded.result_summary['ysipm_summary'].copy()
        self.blockresult.blinded.result_summary['zsipm_summary'] = self.blockresult._BlockResults__unblinded.result_summary['zsipm_summary'].copy()
        self.blockresult.blinded.result_summary['yzsipm_summary'] = self.blockresult._BlockResults__unblinded.result_summary['yzsipm_summary'].copy()
        
        self.blockresult.blinded.result_summary['redchigroupC_summary'] = self.blockresult._BlockResults__unblinded.result_summary['redchigroupC_summary'].copy()
        self.blockresult.blinded.result_summary['redchigroupphi_summary'] = self.blockresult._BlockResults__unblinded.result_summary['redchigroupphi_summary'].copy()
        self.blockresult.blinded.result_summary['redchigrouptau_summary'] = self.blockresult._BlockResults__unblinded.result_summary['redchigrouptau_summary'].copy()
        self.blockresult.blinded.result_summary['redchigroupomega_summary'] = self.blockresult._BlockResults__unblinded.result_summary['redchigroupomega_summary'].copy()

        self.blockresult.blinded.result_sipmsum['N_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['N_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['A_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['A_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['A_sipmsum'][4] = self.blockresult.blinded.result_sipmsum['A_sipmsum'][4] + self.blind._blind_value_in_rad_s * 0.005 * 2
        self.blockresult.blinded.result_sipmsum['dA2_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['dA2_sipmsum'].copy()

        self.blockresult.blinded.result_sipmsum['Ap_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['Ap_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['dAp2_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['dAp2_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['Am_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['Am_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['dAm2_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['dAm2_sipmsum'].copy()

        self.blockresult.blinded.result_sipmsum['C_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['C_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['dC2_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['dC2_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['phi_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['phi_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['phi_sipmsum'][4] = self.blockresult.blinded.result_sipmsum['phi_sipmsum'][4] + self.blind._blind_value_in_rad_s * self.blockresult._BlockResults__unblinded.result_sipmsum['tau_sipmsum'][0]
        self.blockresult.blinded.result_sipmsum['dphi2_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['dphi2_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['tau_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['tau_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['dtau2_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['dtau2_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['omega_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['omega_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['omega_sipmsum'][4] = self.blockresult.blinded.result_sipmsum['omega_sipmsum'][4] + self.blind._blind_value_in_rad_s
        self.blockresult.blinded.result_sipmsum['domega2_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['domega2_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['FA_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['FA_sipmsum'].copy()
        self.blockresult.blinded.result_sipmsum['FC_sipmsum'] = self.blockresult._BlockResults__unblinded.result_sipmsum['FC_sipmsum'].copy()

    def save_result(self):
        with open(os.path.join(self.blockresult_path, "blockresult_" + self.blockresult.block_string)+ '.pkl', 'wb') as f:
            pickle.dump(self.blockresult, f)
    