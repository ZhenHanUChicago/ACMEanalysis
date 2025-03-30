from .binCalculator import binCalculator


import pickle
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
import os

class binCutter:
    class binCutRules:
        def __init__(self):
            self.frac_threshold = 0.25
            self.absgrouptrace_threshold = 100
        def _load_bincut_from_json(self, bincut_file_path):
            try:
                with open(bincut_file_path, 'r') as f:
                    bincut_dict = json.load(f)
                
                for key, value in bincut_dict.items():
                    if hasattr(self, key) and value is not None:
                        setattr(self, key, value)
            except FileNotFoundError:
                print(f"Parameter file {bincut_file_path} not found, using default values.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON file {bincut_file_path}, using default values.")
            except Exception as e:
                print(f"Unexpected error: {e}")

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

    class binCutResults:
        def __init__(self):
            self.frac_mask = None
            self.frac_left = None
            self.frac_right = None
            self.abs_mask = None
            self.abs_left = None
            self.abs_right = None
            self.grand_mask = None
            self.grand_left = None
            self.grand_right = None

    def __init__(self, binresult_file_path = None, bincut_file_path = None, bincutresult_folder_path = None):
        self.binresult_file_path = binresult_file_path
        self.binresult = binCutter._load_results(binresult_file_path)
        self.bincutrule = binCutter.binCutRules()
        self.bincutrule._load_bincut_from_json(bincut_file_path)
        self.bincutresult = binCutter.binCutResults()
        self._frac_cut()
        self._abs_cut()
        self._union_cuts()
        self.saveBinCutResult(bincutresult_folder_path)
    
    def longest_consecutive_ones(arr):
        # Find the longest consecutive 1s sequence
        max_len = 0
        current_len = 0
        begin = -1
        end = -1
        current_begin = -1
        
        for i, val in enumerate(arr):
            if val == 1:
                if current_len == 0:
                    current_begin = i  # Mark the start of the new sequence of 1s
                current_len += 1
            else:
                if current_len > max_len:
                    max_len = current_len
                    begin = current_begin
                    end = i  # Exclusive end
                current_len = 0  # Reset the current sequence length
        
        # Check one more time in case the longest sequence ends at the last index
        if current_len > max_len:
            begin = current_begin
            end = len(arr)
        
        # Create a new array with 1s only in the longest consecutive 1s sequence interval
        result = np.zeros_like(arr)
        if begin != -1:
            result[begin:end] = 1
        
        return result, (begin, end)

    def _frac_cut(self):
        trace_mask = np.ones(self.binresult.N.shape[-1])
        for i in range(self.binresult.N.shape[0]):
            data_slice = self.binresult.N[i].sum(axis = (0,1))
            trace_mask = trace_mask*(data_slice > self.bincutrule.frac_threshold * np.abs(data_slice.max())).astype(int)
        self.bincutresult.frac_mask = trace_mask
        self.bincutresult.frac_mask, (self.bincutresult.frac_left, self.bincutresult.frac_right) = binCutter.longest_consecutive_ones(self.bincutresult.frac_mask)

    def _abs_cut(self):
        trace_mask = np.ones(self.binresult.N.shape[-1])
        for i in range(self.binresult.N.shape[0]):
            data_slice = self.binresult.N[i].sum(axis = (0,1))
            trace_mask = trace_mask*(data_slice > self.bincutrule.absgrouptrace_threshold).astype(int)
        self.bincutresult.abs_mask = trace_mask
        self.bincutresult.abs_mask, (self.bincutresult.abs_left, self.bincutresult.abs_right) = binCutter.longest_consecutive_ones(self.bincutresult.abs_mask)

    def _union_cuts(self):
        self.bincutresult.grand_mask = self.bincutresult.frac_mask * self.bincutresult.abs_mask
        self.bincutresult.grand_left, self.bincutresult.grand_right = max(self.bincutresult.frac_left, self.bincutresult.abs_left), min(self.bincutresult.frac_right, self.bincutresult.abs_right)

    def saveBinCutResult(self, folder_path):
        # Ensure the directory exists before writing the file
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)  # Create the directory if it does not exist
            except OSError as e:
                print(f"Bin Cutter: Error creating directory {folder_path}: {e}")
                return

        # Generate a file name based on your logic (example: timestamp + .pkl)
        file_name = "bincutresult_" + self.binresult.name +".pkl"
        file_path = os.path.join(folder_path, file_name)

        try:
            with open(file_path, 'wb') as f:
                # Save only the bin results without additional dictionary layers
                pickle.dump(self.bincutresult, f)
            #print(f"Bin Cutter: Results saved successfully to {folder_path}")
        except Exception as e:
            print(f"Bin Calculator: Error saving results: {e}")