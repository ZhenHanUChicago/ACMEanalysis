from .blockCalculator import blockCalculator


import pickle
import json
import numpy as np
import os

class blockCutter:
    class BlockCutRules:
        def __init__(self):
            self.min_number_of_good_groups = 20
            self.min_C = 0.6
            self.max_C = 1.4

        def _load_blockcut_rules_from_json(self, blockcut_file_path):
            try:
                with open(blockcut_file_path, 'r') as f:
                    blockcut_dict = json.load(f)
                
                for key, value in blockcut_dict.items():
                    if hasattr(self, key) and value is not None:
                        setattr(self, key, value)
            except FileNotFoundError:
                print(f"Parameter file {blockcut_file_path} not found, using default values.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON file {blockcut_file_path}, using default values.")
            except Exception as e:
                print(f"Unexpected error: {e}")

    class BlockCutResults:
        def __init__(self):
            self.does_this_block_stay = True
            self.cause_of_removal = []

    def __init__(self, blockresult_path = None, blockcutresult_folder_path = None, block_cut_rule_json_path = None, df = None):
        self.blockresult_path = blockresult_path
        self.df = df
        self.blockcutresult_folder_path = blockcutresult_folder_path
        self.blockresult = self._load_results(self.blockresult_path)
        self.blockcutrule = self.BlockCutRules()
        self.blockcutrule._load_blockcut_rules_from_json(block_cut_rule_json_path)
        self.blockcutresult = self.BlockCutResults()
        self.cut_blocks()
        self.saveBlockCutResult(self.blockcutresult_folder_path)

    def cut_blocks(self):
        """This function performs the actual cutting"""
        if self.blockresult.blockcut_right - self.blockresult.blockcut_left < self.blockcutrule.min_number_of_good_groups:
            self.blockcutresult.does_this_block_stay = False
            self.blockcutresult.cause_of_removal.append("goodgrouptoofew")
            return None
        
        if self.blockresult.blockcut_left<self.blockresult.blockcut_right:
            if np.abs(self.blockresult._BlockResults__unblinded.result['C'][0,..., self.blockresult.blockcut_left:self.blockresult.blockcut_right]).min() < self.blockcutrule.min_C:
                self.blockcutresult.does_this_block_stay = False
                self.blockcutresult.cause_of_removal.append("Clow")
            
            if np.abs(self.blockresult._BlockResults__unblinded.result['C'][0,..., self.blockresult.blockcut_left:self.blockresult.blockcut_right]).max() > self.blockcutrule.max_C:
                self.blockcutresult.does_this_block_stay = False
                self.blockcutresult.cause_of_removal.append("Chigh")
  
    def saveBlockCutResult(self, folder_path):
        # Ensure the directory exists before writing the file
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)  # Create the directory if it does not exist
            except OSError as e:
                print(f"Block Cutter: Error creating directory {folder_path}: {e}")
                return

        # Generate a file name based on your logic (example: timestamp + .pkl)
        file_name = "blockcutresult_" + self.blockresult.block_string +".pkl"
        file_path = os.path.join(folder_path, file_name)

        try:
            with open(file_path, 'wb') as f:
                # Save only the bin results without additional dictionary layers
                pickle.dump(self.blockcutresult, f)
            # print(f"Block Cutter: Results saved successfully to {folder_path}")
        except Exception as e:
            print(f"Block Cutter: Error saving results: {e}")

    @staticmethod
    def _load_results(result_file_path):
        """Load the pickled results from the file."""
        try:
            with open(result_file_path, 'rb') as f:
                blockresult = pickle.load(f)
            return blockresult
        except Exception as e:
            print(f"Error loading results: {e}")
            return None
    

