import json
import os
import pickle
import numpy as np
from numba import jit, prange
import math

class binCalculator:
    class Parameters:
        def __init__(self, parent):
            self.name = None
            self.parent = parent
            self.bin_mask = (0, 0, 0, 0)
            self.bin_size = 50
            self.bin_offset = 0
            self.background_begin_time_ms = 0
            self.background_end_time_ms = 0.5
            self.background_average_level = '4trace'
            self.background_D_mode = 'equal'
            self.noisechannel_substraction = False
            self.noisechannel = []
            self.group_size = 20
            self.sum_shot = True
            self.sum_channel = False
            self.ACMEIII_header_purge = True
            self.file_dtype = 'int16'
            self.trace_already_summed = True
            self.channel_weight =[-0.03963429, -0.03918644, -0.03918644, -0.04009249, -0.03986207,
       -0.03853333, -0.0387486 , -0.0387486]
            self.total_timestamps = 175000
            self.info_header_length = 4

        def _load_parameters_from_json(self, parameter_file_path):
            try:
                with open(parameter_file_path, 'r') as f:
                    param_dict = json.load(f)
                
                for key, value in param_dict.items():
                    if hasattr(self, key) and value is not None:
                        setattr(self, key, value)
            except FileNotFoundError:
                print(f"Parameter file {parameter_file_path} not found, using default values.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON file {parameter_file_path}, using default values.")
            except Exception as e:
                print(f"Unexpected error: {e}")

    class BinResults:
        def __init__(self):
            self.name = None
            self.A = None
            self.N = None
            self.dA2_from_photon = None
            self.dA2_from_fitting = None
            self.chi_square = None
            self.red_chi_square = None
            self.N0 = None
            self.A0 = None
            self.N1 = None
            self.A1 = None
            self.shot_yield = None
            self.e_A = None
            self.e_N = None
            self.bin_pair_inspection_data = None
            self.x_mask = None  # pre swap x_mask
            self.y_mask = None # pre swap y_mask
            self.swap_xy = None
            self.F = None

    def __init__(self, binary_file_path, parameter_file_path, df_slice):
        # Initialization
        self.raw_data = None
        self.data = None
        self.name = None
        self.file_name = None
        self.background_subtracted_data = None
        self.ungrouped_Fx = None
        self.ungrouped_Fy = None
        self.ungrouped_N_total = None
        self.ungrouped_asymmetry = None
        self.grouped_Fx = None
        self.grouped_Fy =  None
        self.grouped_F = None  # Fx - Fy
        self.grouped_N_total = None
        self.grouped_asymmetry = None
        self.ungrouped_Fx_before_substraction = None
        self.ungrouped_Fy_before_substraction = None
        self.df = df_slice
        self.binary_file_path = binary_file_path
        self.parameter_file_path = parameter_file_path
        self.binresult = self.BinResults()
        self.parameter = self.Parameters(self)
        self.parameter._load_parameters_from_json(self.parameter_file_path)
        try:
            self.swap_xy = False if self.df['Polarization Switching XY Swapped'].dropna().mode()[0] == 0 else True
        except:
            self.swap_xy = False
        self.binresult.swap_xy = self.swap_xy
        # print('swap_xy', self.swap_xy)
    
    def default_pipeline(self):
        # 1. Load the binary data
        self._load_binary_data(time_total_timestamps=self.parameter.total_timestamps)

        self._photon_conversion()

        self._apply_binning(time_total_timestamps=self.parameter.total_timestamps)

        self._calculate_Fx_Fy()

        self._bg_substraction()

        self._calculate_asymmetry()

        self._group_asymmetry()

        self._group_linear_fit()

        self._chi_square()

        self._calculate_shot_yield()

    def smooth_data_step(data, step_size=5):
        smoothed_data = np.array([
            np.mean(data[max(0, i - step_size // 2):min(len(data), i + step_size // 2 + 1)]) 
            for i in range(len(data))
        ])
        return smoothed_data

    def _load_binary_data(self, time_total_timestamps = 175000):
        """ load in the binary file as unbinned data (before scaling to photon number) into self.raw_data """
        def _header_purging(data,info_header_length = 4,channel_total_number = 8, time_total_timestamps= time_total_timestamps):
            reshaped_data = data.reshape(-1,info_header_length+channel_total_number*time_total_timestamps)
            truncated_segments = reshaped_data[:, info_header_length:]
            purged_data = truncated_segments.flatten()
            return purged_data

        def _fluorescent_to_data(data, purge = True, trace_already_summed = True, info_header_length = 4, time_total_timestamps = time_total_timestamps):
            channel_total_number = 8
            time_total_timestamps = time_total_timestamps
            if purge == True:
                shot_number = int(len(data)/(channel_total_number*time_total_timestamps+ info_header_length))
                purged_data = _header_purging(data, info_header_length=info_header_length, time_total_timestamps=time_total_timestamps)
            else:
                shot_number = int(len(data)/(channel_total_number*time_total_timestamps))
                purged_data = data
            if trace_already_summed is False:
                trace_number = shot_number//25
                mat = purged_data.reshape(trace_number, 25 , channel_total_number,-1)
            else:
                trace_number = shot_number
                mat = purged_data.reshape(trace_number, 1 , channel_total_number, -1)
            return mat

        if self.parameter.file_dtype == 'int16':
            dtype = np.int16
        elif self.parameter.file_dtype == 'int32':
            dtype = np.int32
        elif self.parameter.file_dtype == 'float32':
            dtype = np.float32
        elif self.parameter.file_dtype == 'float64':
            dtype = np.float64

        with open(self.binary_file_path, "rb") as file:
            self.data = np.fromfile(self.binary_file_path, dtype=dtype)
        self.file_name = os.path.basename(self.binary_file_path)
        self.name, _ = os.path.splitext(self.file_name)
        self.binresult.name = self.name
        self.raw_data = _fluorescent_to_data(self.data, self.parameter.ACMEIII_header_purge, self.parameter.trace_already_summed, self.parameter.info_header_length, time_total_timestamps=self.parameter.total_timestamps)
        self.raw_data = self.raw_data.astype(np.float64)

    def _photon_conversion(self):
        self.data = self.raw_data*(np.array(self.parameter.channel_weight).reshape(1,1,8,1))

    def _apply_binning(self, time_total_timestamps = 175000):
        bin_offset = self.parameter.bin_offset
        while bin_offset<0:
            bin_offset = bin_offset + self.parameter.bin_size
        while bin_offset>=self.parameter.bin_size:
            bin_offset = bin_offset - self.parameter.bin_size
        adjusted_size = time_total_timestamps - (time_total_timestamps - bin_offset) % self.parameter.bin_size
        if self.parameter.trace_already_summed is False:
            self.binned_data = self.data[:,:,:,bin_offset:adjusted_size].reshape(self.data.shape[0],25,8,-1,self.parameter.bin_size)
        else:
            self.binned_data = self.data[:,:,:,bin_offset:adjusted_size].reshape(self.data.shape[0],1,8,-1,self.parameter.bin_size)

    def _calculate_Fx_Fy(self):
        def _generate_mask(x_gap_left=0,x_gap_right=0,y_gap_left=0,y_gap_right=0,bin_size=50, swap_xy = False):
            x_mask = np.concatenate([np.zeros(x_gap_left),np.ones(bin_size//2-x_gap_left-x_gap_right),np.zeros(x_gap_right),np.zeros(bin_size//2)])
            y_mask = np.concatenate([np.zeros(bin_size//2),np.zeros(y_gap_left),np.ones(bin_size//2-y_gap_left-y_gap_right),np.zeros(y_gap_right)])
            if swap_xy:
                return y_mask,x_mask
            return x_mask,y_mask

        # Use the bin_mask to separate Fx and Fy from data
        x_mask, y_mask = _generate_mask(*self.parameter.bin_mask, self.parameter.bin_size, self.swap_xy)
        self.binresult.x_mask = x_mask
        self.binresult.y_mask = y_mask
        ungrouped_Fx = np.tensordot(self.binned_data, x_mask, axes=([4], [0]))
        ungrouped_Fy = np.tensordot(self.binned_data, y_mask, axes=([4], [0]))

        if self.parameter.sum_shot:
            ungrouped_Fx = ungrouped_Fx.sum(axis = (1),keepdims = True)
            ungrouped_Fy = ungrouped_Fy.sum(axis = (1),keepdims = True)
        if self.parameter.sum_channel:
            ungrouped_Fx = ungrouped_Fx.sum(axis = (2),keepdims = True)
            ungrouped_Fy = ungrouped_Fy.sum(axis = (2),keepdims = True)

        self.ungrouped_Fx_before_substraction = ungrouped_Fx
        self.ungrouped_Fy_before_substraction = ungrouped_Fy
        
        if self.parameter.noisechannel_substraction:
            if len(self.parameter.noisechannel) > 0:
                background_channel = self.parameter.noisechannel
                remaining_channel = [i for i in range(8) if i not in background_channel]
                predicted_bg = np.zeros(self.ungrouped_Fy_before_substraction.mean(axis = 2, keepdims = True).shape)
                for trace in range(self.ungrouped_Fy_before_substraction.shape[0]):
                    for shot in range(self.ungrouped_Fy_before_substraction.shape[1]):
                        if len(background_channel) > 1:
                            predicted_bg[trace, shot, 0 , :] = (binCalculator.smooth_data_step(self.ungrouped_Fy_before_substraction[trace, shot, background_channel, :].mean(axis = 0), 300) + binCalculator.smooth_data_step(self.ungrouped_Fx_before_substraction[trace, shot, background_channel, :].mean(axis = 0), 300))/2.0
                        else:
                            predicted_bg[trace, shot, 0 , :] = (binCalculator.smooth_data_step(self.ungrouped_Fy_before_substraction[trace, shot, background_channel[0], :], 300) + binCalculator.smooth_data_step(self.ungrouped_Fx_before_substraction[trace, shot, background_channel[0], :], 300))/2.0
                self.ungrouped_Fx_before_substraction = self.ungrouped_Fx_before_substraction[:,:,remaining_channel,:] - predicted_bg
                self.ungrouped_Fy_before_substraction = self.ungrouped_Fy_before_substraction[:,:,remaining_channel,:] - predicted_bg
        else:
            if len(self.parameter.noisechannel) > 0:
                background_channel = self.parameter.noisechannel
                remaining_channel = [i for i in range(8) if i not in background_channel]
                self.ungrouped_Fx_before_substraction = self.ungrouped_Fx_before_substraction[:,:,remaining_channel,:]
                self.ungrouped_Fy_before_substraction = self.ungrouped_Fy_before_substraction[:,:,remaining_channel,:]
            
    def _bg_substraction(self):
        background_begin_bin_index = int(math.floor((self.parameter.background_begin_time_ms*1000000//80)//self.parameter.bin_size))
        background_end_bin_index = int(math.floor((self.parameter.background_end_time_ms*1000000//80)//self.parameter.bin_size))

        if self.parameter.background_average_level == 'shot':
            bgx = self.ungrouped_Fx_before_substraction[:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = 3, keepdims = True).repeat(self.ungrouped_Fx_before_substraction.shape[3],axis = 3)
            bgy = self.ungrouped_Fy_before_substraction[:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = 3, keepdims = True).repeat(self.ungrouped_Fy_before_substraction.shape[3],axis = 3)
        if self.parameter.background_average_level == 'trace':
            bgx = self.ungrouped_Fx_before_substraction[:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (1,3), keepdims = True).repeat(self.ungrouped_Fx_before_substraction.shape[1],axis = 1).repeat(self.ungrouped_Fx_before_substraction.shape[3],axis = 3)
            bgy = self.ungrouped_Fy_before_substraction[:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (1,3), keepdims = True).repeat(self.ungrouped_Fy_before_substraction.shape[1],axis = 1).repeat(self.ungrouped_Fy_before_substraction.shape[3],axis = 3)
        if self.parameter.background_average_level == 'block':
            bgx = self.ungrouped_Fx_before_substraction[:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (0,1,3), keepdims = True).repeat(self.ungrouped_Fx_before_substraction.shape[0],axis = 0).repeat(self.ungrouped_Fx_before_substraction.shape[1],axis = 1).repeat(self.ungrouped_Fx_before_substraction.shape[3],axis = 3)
            bgy = self.ungrouped_Fy_before_substraction[:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (0,1,3), keepdims = True).repeat(self.ungrouped_Fy_before_substraction.shape[0],axis = 0).repeat(self.ungrouped_Fy_before_substraction.shape[1],axis = 1).repeat(self.ungrouped_Fy_before_substraction.shape[3],axis = 3)
        if self.parameter.background_average_level == 'none':
            bgx = 0
            bgy = 0
        if self.parameter.background_average_level == '2trace':
            bgx = self.ungrouped_Fx_before_substraction.reshape(-1,2,*self.ungrouped_Fx_before_substraction.shape[1:])[:,:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (1,2,4), keepdims = True).repeat(2,axis = 1).repeat(self.ungrouped_Fx_before_substraction.shape[1],axis = 2).repeat(self.ungrouped_Fx_before_substraction.shape[3],axis = 4).reshape(self.ungrouped_Fx_before_substraction.shape)
            bgy = self.ungrouped_Fy_before_substraction.reshape(-1,2,*self.ungrouped_Fy_before_substraction.shape[1:])[:,:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (1,2,4), keepdims = True).repeat(2,axis = 1).repeat(self.ungrouped_Fy_before_substraction.shape[1],axis = 2).repeat(self.ungrouped_Fy_before_substraction.shape[3],axis = 4).reshape(self.ungrouped_Fy_before_substraction.shape)
        if self.parameter.background_average_level == '4trace':
            bgx = self.ungrouped_Fx_before_substraction.reshape(-1,4,*self.ungrouped_Fx_before_substraction.shape[1:])[:,:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (1,2,4), keepdims = True).repeat(4,axis = 1).repeat(self.ungrouped_Fx_before_substraction.shape[1],axis = 2).repeat(self.ungrouped_Fx_before_substraction.shape[3],axis = 4).reshape(self.ungrouped_Fx_before_substraction.shape)
            bgy = self.ungrouped_Fy_before_substraction.reshape(-1,4,*self.ungrouped_Fy_before_substraction.shape[1:])[:,:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (1,2,4), keepdims = True).repeat(4,axis = 1).repeat(self.ungrouped_Fy_before_substraction.shape[1],axis = 2).repeat(self.ungrouped_Fy_before_substraction.shape[3],axis = 4).reshape(self.ungrouped_Fy_before_substraction.shape)
        if self.parameter.background_average_level == '8trace':
            bgx = self.ungrouped_Fx_before_substraction.reshape(-1,8,*self.ungrouped_Fx_before_substraction.shape[1:])[:,:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (1,2,4), keepdims = True).repeat(8,axis = 1).repeat(self.ungrouped_Fx_before_substraction.shape[1],axis = 2).repeat(self.ungrouped_Fx_before_substraction.shape[3],axis = 4).reshape(self.ungrouped_Fx_before_substraction.shape)
            bgy = self.ungrouped_Fy_before_substraction.reshape(-1,8,*self.ungrouped_Fy_before_substraction.shape[1:])[:,:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (1,2,4), keepdims = True).repeat(8,axis = 1).repeat(self.ungrouped_Fy_before_substraction.shape[1],axis = 2).repeat(self.ungrouped_Fy_before_substraction.shape[3],axis = 4).reshape(self.ungrouped_Fy_before_substraction.shape)
        if self.parameter.background_average_level == '16trace':
            bgx = self.ungrouped_Fx_before_substraction.reshape(-1,16,*self.ungrouped_Fx_before_substraction.shape[1:])[:,:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (1,2,4), keepdims = True).repeat(16,axis = 1).repeat(self.ungrouped_Fx_before_substraction.shape[1],axis = 2).repeat(self.ungrouped_Fx_before_substraction.shape[3],axis = 4).reshape(self.ungrouped_Fx_before_substraction.shape)
            bgy = self.ungrouped_Fy_before_substraction.reshape(-1,16,*self.ungrouped_Fy_before_substraction.shape[1:])[:,:,:,:,background_begin_bin_index:background_end_bin_index].mean(axis = (1,2,4), keepdims = True).repeat(16,axis = 1).repeat(self.ungrouped_Fy_before_substraction.shape[1],axis = 2).repeat(self.ungrouped_Fy_before_substraction.shape[3],axis = 4).reshape(self.ungrouped_Fy_before_substraction.shape)
            
        if self.parameter.background_D_mode == 'equal':
            bg = (bgx + bgy) / 2.0
            bgx = bg
            bgy = bg

        self.bgx = bgx
        self.bgy = bgy
        self.ungrouped_Fx = self.ungrouped_Fx_before_substraction - bgx
        self.ungrouped_Fy = self.ungrouped_Fy_before_substraction - bgy

        # for bin_pair_inspection_slice
        bgz = self.binned_data[:,:,:,background_begin_bin_index:background_end_bin_index,:].mean(axis = (1,3), keepdims = True).repeat(self.binned_data.shape[1],axis = 1).repeat(self.binned_data.shape[3],axis = 3)
        self.binresult.bin_pair_inspection_data = (self.binned_data - bgz).mean(axis = (0,1,2))

    def _calculate_asymmetry(self):
        self.ungrouped_N_total = self.ungrouped_Fx + self.ungrouped_Fy
        self.ungrouped_asymmetry = (self.ungrouped_Fx - self.ungrouped_Fy) / self.ungrouped_N_total

    def _group_asymmetry(self):
        group_size = self.parameter.group_size

        truncated_bin_size = self.ungrouped_N_total.shape[-1] - (self.ungrouped_N_total.shape[-1] )%group_size
        number_of_group = truncated_bin_size//group_size
        new_dimension = (*self.ungrouped_N_total.shape[:-1], number_of_group, group_size)
        self.grouped_Fx = self.ungrouped_Fx[..., :truncated_bin_size].reshape(new_dimension)
        self.grouped_Fy =  self.ungrouped_Fy[..., :truncated_bin_size].reshape(new_dimension)
        self.grouped_F = self.grouped_Fx - self.grouped_Fy
        self.binresult.F = self.grouped_F.sum(axis = -1)
        self.grouped_N_total =  self.ungrouped_N_total[..., :truncated_bin_size].reshape(new_dimension)
        self.grouped_asymmetry =  self.ungrouped_asymmetry[..., :truncated_bin_size].reshape(new_dimension)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _linear_fit_5d_numba(data,x):
        traces, shots, channels, groups, withingroups = data.shape
        B0 = np.zeros((traces, shots, channels, groups))
        B1 = np.zeros((traces, shots, channels, groups))
        err = np.zeros((traces, shots, channels, groups, withingroups))
        for trace in prange(traces):
            for shot in range(shots):
                for channel in range(channels):
                    for group in range(groups):
                        y = data[trace, shot, channel, group, :]
                        # Calculating sums needed for the coefficients
                        Sxx = np.sum(x * x)
                        Sxy = np.sum(x * y)
                        Sx = np.sum(x)
                        Sy = np.sum(y)
                        # Slope and intercept calculations
                        denominator = withingroups * Sxx - Sx ** 2
                        if denominator != 0:
                            slope = (withingroups * Sxy - Sx * Sy) / denominator
                            intercept = (Sy - slope * Sx) / withingroups
                        else:
                            slope = 0.0
                            intercept = np.mean(y)  # or any fallback logic
                        # Store results
                        B0[trace, shot, channel, group] = intercept
                        B1[trace, shot, channel, group] = slope
                        # Calculate residuals
                        predicted_y = intercept + slope * x
                        err[trace, shot, channel, group, :] = y - predicted_y
        return B0, B1, err

    def _group_linear_fit(self):
        t = np.arange(self.parameter.group_size)
        self.binresult.A0, self.binresult.A1, self.binresult.e_A = binCalculator._linear_fit_5d_numba(self.grouped_asymmetry, t)
        self.binresult.N0, self.binresult.N1, self.binresult.e_N = binCalculator._linear_fit_5d_numba(self.grouped_N_total, t)

    def _chi_square(self):
        self.binresult.A = np.zeros(self.binresult.A0.shape)
        self.binresult.N = np.zeros(self.binresult.N0.shape)
        self.binresult.dA2_from_fitting = np.zeros(self.binresult.A0.shape)
        self.binresult.dA2_from_photon = np.zeros(self.binresult.A0.shape)

        for trace in range(self.binresult.N0.shape[0]):
            for shot in range(self.binresult.N0.shape[1]):
                for channel in range(self.binresult.N0.shape[2]):
                    for group in range(self.binresult.N0.shape[3]):
                        self.binresult.N[trace, shot, channel, group] = self.parameter.group_size* (self.binresult.N0[trace, shot, channel, group] + self.binresult.N1[trace, shot, channel, group] * (self.parameter.group_size - 1)/2)
                        self.binresult.A[trace, shot, channel, group]  = self.binresult.A0[trace, shot, channel, group] + self.binresult.A1[trace, shot, channel, group] * (self.parameter.group_size - 1)/2
                        self.binresult.dA2_from_fitting[trace, shot, channel, group] = 1/(self.parameter.group_size)*1/(self.parameter.group_size -2)*np.sum((self.binresult.e_A[trace, shot, channel, group, :])**2)
                        self.binresult.dA2_from_photon[trace, shot, channel, group] = (1 - (self.binresult.A0[trace, shot, channel, group] + self.binresult.A1[trace, shot, channel, group]*(self.parameter.group_size-1)/2) ** 2)/(self.parameter.group_size*(self.binresult.N0[trace, shot, channel, group]+self.binresult.N1[trace, shot, channel, group]*((self.parameter.group_size-1)/2)))
        self.binresult.red_chi_square = self.binresult.dA2_from_fitting / self.binresult.dA2_from_photon
        self.binresult.chi_square = self.binresult.red_chi_square * (self.parameter.group_size - 2)

    def _calculate_shot_yield(self):
        self.binresult.shot_yield = self.ungrouped_N_total.sum(axis = (2,3))

    def saveBinResults(self, folder_path):
        # Ensure the directory exists before writing the file
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)  # Create the directory if it does not exist
            except OSError as e:
                print(f"Bin Calculator: Error creating directory {folder_path}: {e}")
                return

        # Generate a file name based on your logic (example: timestamp + .pkl)
        file_name = "binresult_" + self.binresult.name +".pkl"
        file_path = os.path.join(folder_path, file_name)

        try:
            with open(file_path, 'wb') as f:
                # Save only the bin results without additional dictionary layers
                pickle.dump(self.binresult, f)
            # print(f"Bin result saved: {file_path}")
        except Exception as e:
            print(f"Bin Calculator: Error saving results: {e}")
