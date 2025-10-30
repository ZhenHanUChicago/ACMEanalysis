
from .binCalculator import binCalculator
from .binCutter import binCutter
from .binVisualizer import binVisualizer
from .blockCalculator import blockCalculator
from .blockCutter import blockCutter
from .blockVisualizer import blockVisualizer
from .headerHandler import hh
from .parityStateTransfrom import parityStateTransform, combine_switches
from .sequenceCalculator import sequenceCalculator
from .sequenceVisualizer import sequenceVisualizer



import os
import pandas as pd
from itertools import product, combinations
import json
import glob
import gc
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import random
import shutil
from tqdm import tqdm
import subprocess
import warnings
from IPython.display import clear_output
from joblib import Parallel, delayed


warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class runHandler:
    def __init__(self):
        self.run_folder = None
        self.binpara_folder_path = None
        self.bincut_folder_path = None
        self.blockpara_folder_path = None
        self.blockcut_folder_path = None
        self.config_folder_path = None
        self.sequence_folder_path = None
        self.blinding_folder_path = None
        self.run_results_data_path = None
        self.run_results_figures_path = None
        self.path_df = None
        self.binary_data_folder_paths = []  # List of folders containing .bin files
        self.run_sequence_range = []  # List of [run, sequence] pairs
        self.sequence_assignment = None  # Sequence assignment
        self.df = None
        self.purged_df = None
        self.block_df = None

    def old_pipeline(self, name = "noise", sequence = 1013):
        self.load_run_folder(r"C:\ACME_analysis" + "\\" + name + str(sequence))
        self.load_sequence_assignment()
        self.load_aggregated_header()
        self.initialize_run_folder()
        self.set_binary_data_folder_paths([r"C:\ACMEdata\data"+str(sequence)])
        self.touch()
    
    def _sequence_instruction_maker(self, sequence_type):
        """
        Processes a sequence_type input (string, list, or dictionary) and creates JSON files
        in the folder:
            C:\\ACMEcode\\ACMEanalysis\\templates\\example_folder\\Analysis Parameters\\sequence

        Returns:
            List of filenames (without .json) created or directly appended.
        """
        
        # Define output folder
        output_folder = r"C:\ACMEcode\ACMEanalysis\templates\example_folder\Analysis Parameters\sequence"
        os.makedirs(output_folder, exist_ok=True)
        
        # Normalize input into a list
        if isinstance(sequence_type, (str, dict)):
            sequence_list = [sequence_type]
        elif isinstance(sequence_type, list):
            sequence_list = sequence_type
        else:
            raise TypeError("sequence_type must be a string, list, or dictionary.")
        
        result = []

        for element in sequence_list:
            if isinstance(element, str):
                # Append directly if string
                result.append(element)
                continue
            
            if not isinstance(element, dict):
                continue  # skip invalid elements

            # Check required keys
            if "non_parity_switches" not in element or "superblock_parity_switches" not in element:
                continue

            # Extract lists
            non_parity = element.get("non_parity_switches", [])
            superblock_parity = element.get("superblock_parity_switches", [])

            # Build filename
            name_parts = []
            superblock_str = ''.join(superblock_parity)
            non_parity_str = ''.join(non_parity)
            
            if superblock_str and non_parity_str:
                filename = f"{superblock_str}~{non_parity_str}"
            elif superblock_str:
                filename = superblock_str
            elif non_parity_str:
                filename = non_parity_str
            else:
                continue  # skip if both empty

            # Sanitize filename
            filename = filename.replace("/", "").replace("\\", "")
            filename = filename.replace("-", "m").replace("+", "p").replace("_", "")
            json_filename = filename + ".json"

            # Write JSON file
            json_path = os.path.join(output_folder, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(element, f, indent=4)

            result.append(filename)

        return result

    def new_pipeline(self, name = "noise", run = 9 , sequence = 1210, sequence_type = None, bincut = None, binpara = None, blockcut = None, blockpara = None, config = None):
        sequence_type_text = self._sequence_instruction_maker(sequence_type)
        work_folder = runHandler.create_work_folder(name, run, sequence, sequence_type = sequence_type_text, bincut=bincut, binpara=binpara, blockcut=blockcut, blockpara=blockpara, config=config)
        data_folder, header_folder = runHandler.grab_data(run = run, sequence = sequence)
        self.load_run_folder(work_folder)
        self.load_sequence_assignment()
        self.create_aggregated_header([header_folder])
        self.load_aggregated_header()
        self.initialize_run_folder()
        self.set_binary_data_folder_paths([data_folder])
        self.touch()

    def load_pipeline(self, name = "noise", run = 9, sequence = 1210):
        if isinstance(sequence, int):
            sequence = [sequence]
        sequence = list(sequence)
        work_folder = r"C:\ACME_analysis" + "\\" + name + str(run).zfill(4) + "."+ str(sequence[0]).zfill(4)
        self.load_run_folder(work_folder)
        self.load_sequence_assignment()
        self.load_aggregated_header()
        self.initialize_run_folder()
        self.set_binary_data_folder_paths([r"C:\ACMEdata\data"+ str(run).zfill(4) + "."+ str(sequence[0]).zfill(4)])
        self.touch()

    def calculation_pipeline(self, parallel = True, overwrite = False):
        self.calculate_bin_result(parallel=parallel, overwrite = overwrite)
        self.cut_bin()
        self.calculate_block_result()
        self.cut_block()
        self.calculate_sequence_result()

    def create_work_folder(name, run, sequence, 
                        parent_folder_path=r"C:\\ACME_analysis", 
                        example_folder_path=r"C:\ACMEcode\ACMEanalysis\templates\example_folder",
                        sequence_type=None, bincut=None, binpara=None, blockcut=None, blockpara=None, config=None):
        # Ensure sequence is iterable
        if isinstance(sequence, int):
            sequence = [sequence]

        # Prepare folder name
        run_str = str(run).zfill(4)
        seq_str = str(sequence[0]).zfill(4)
        new_folder_name = f"{name}{run_str}.{seq_str}"
        new_folder_path = os.path.join(parent_folder_path, new_folder_name)

        # --- Modified folder creation logic ---
        if not os.path.exists(new_folder_path):
            print(f"Creating folder {new_folder_name} and copying contents from example folder.")
            shutil.copytree(example_folder_path, new_folder_path)
        else:
            print(f"Folder {new_folder_name} already exists. Proceeding without overwriting folder content.")
            # Ensure subfolders exist (but do not overwrite existing ones)
            for root, dirs, _ in os.walk(example_folder_path):
                rel_path = os.path.relpath(root, example_folder_path)
                dest_path = os.path.join(new_folder_path, rel_path)
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path, exist_ok=True)

        # --------------------------------------------------------------
        # -------------------- BIN CUT HANDLING -------------------------
        # --------------------------------------------------------------
        path_to_bincut = os.path.join(new_folder_path, "Analysis Parameters", "bincut")
        path_to_bincut_template = r"C:\ACMEcode\ACMEanalysis\templates\bincut"
        if bincut is not None:
            for file in os.listdir(path_to_bincut):
                if file.endswith(".json"):
                    os.remove(os.path.join(path_to_bincut, file))
            if not isinstance(bincut, list):
                bincut = [bincut]
            for bincut_instance in bincut:
                if isinstance(bincut_instance, int):
                    shutil.copy(os.path.join(path_to_bincut_template, f"frac{str(bincut_instance).zfill(2)}.json"),
                                os.path.join(path_to_bincut, f"frac{str(bincut_instance).zfill(2)}.json"))
                elif isinstance(bincut_instance, str):
                    shutil.copy(os.path.join(path_to_bincut_template, f"{bincut_instance}.json"),
                                os.path.join(path_to_bincut, f"{bincut_instance}.json"))

        # --------------------------------------------------------------
        # -------------------- BIN PARA HANDLING ------------------------
        # --------------------------------------------------------------
        path_to_binpara = os.path.join(new_folder_path, "Analysis Parameters", "binpara")
        path_to_binpara_template = r"C:\ACMEcode\ACMEanalysis\templates\binpara"

        # Handle defaults for None or []: take offsettrace19.json
        if binpara is None or (isinstance(binpara, list) and len(binpara) == 0):
            shutil.copy(os.path.join(path_to_binpara_template, "offsettrace19.json"),
                        os.path.join(path_to_binpara, "offsettrace19.json"))
        else:
            # Delete existing jsons
            for file in os.listdir(path_to_binpara):
                if file.endswith(".json"):
                    os.remove(os.path.join(path_to_binpara, file))

            # Normalize to list
            if not isinstance(binpara, list):
                binpara = [binpara]

            # Load baseline once
            with open(os.path.join(path_to_binpara_template, "offsettrace19.json"), "r") as _f:
                _baseline_dict = json.load(_f)

            first = binpara[0] if len(binpara) > 0 else None

            # Case 2c: first is list -> expect [string, dict] pairs with custom names
            if isinstance(first, list):
                for item in binpara:
                    if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], dict):
                        fname = item[0] if item[0].lower().endswith(".json") else item[0] + ".json"
                        merged = dict(_baseline_dict); merged.update(item[1] or {})
                        with open(os.path.join(path_to_binpara, fname), "w") as f:
                            json.dump(merged, f, indent=2)
                    elif isinstance(item, int):
                        shutil.copy(os.path.join(path_to_binpara_template, f"offsettrace{item}.json"),
                                    os.path.join(path_to_binpara, f"offsettrace{item}.json"))
                    elif isinstance(item, str):
                        shutil.copy(os.path.join(path_to_binpara_template, f"{item}.json"),
                                    os.path.join(path_to_binpara, f"{item}.json"))

            # Case 2d: first is dict -> auto-name customizedbinpara1,2,3...
            elif isinstance(first, dict):
                idx = 1
                for item in binpara:
                    if isinstance(item, dict):
                        merged = dict(_baseline_dict); merged.update(item or {})
                        with open(os.path.join(path_to_binpara, f"customizedbinpara{idx}.json"), "w") as f:
                            json.dump(merged, f, indent=2)
                        idx += 1
                    elif isinstance(item, int):
                        shutil.copy(os.path.join(path_to_binpara_template, f"offsettrace{item}.json"),
                                    os.path.join(path_to_binpara, f"offsettrace{item}.json"))
                    elif isinstance(item, str):
                        shutil.copy(os.path.join(path_to_binpara_template, f"{item}.json"),
                                    os.path.join(path_to_binpara, f"{item}.json"))

            # Case 2a: first is int -> treat list as ints (but tolerate mixed)
            elif isinstance(first, int):
                idx = 1
                for item in binpara:
                    if isinstance(item, int):
                        shutil.copy(os.path.join(path_to_binpara_template, f"offsettrace{item}.json"),
                                    os.path.join(path_to_binpara, f"offsettrace{item}.json"))
                    elif isinstance(item, str):
                        shutil.copy(os.path.join(path_to_binpara_template, f"{item}.json"),
                                    os.path.join(path_to_binpara, f"{item}.json"))
                    elif isinstance(item, dict):
                        merged = dict(_baseline_dict); merged.update(item or {})
                        with open(os.path.join(path_to_binpara, f"customizedbinpara{idx}.json"), "w") as f:
                            json.dump(merged, f, indent=2)
                        idx += 1

            # Case 2b: first is string -> treat list as strings (but tolerate mixed)
            elif isinstance(first, str):
                idx = 1
                for item in binpara:
                    if isinstance(item, str):
                        shutil.copy(os.path.join(path_to_binpara_template, f"{item}.json"),
                                    os.path.join(path_to_binpara, f"{item}.json"))
                    elif isinstance(item, int):
                        shutil.copy(os.path.join(path_to_binpara_template, f"offsettrace{item}.json"),
                                    os.path.join(path_to_binpara, f"offsettrace{item}.json"))
                    elif isinstance(item, dict):
                        merged = dict(_baseline_dict); merged.update(item or {})
                        with open(os.path.join(path_to_binpara, f"customizedbinpara{idx}.json"), "w") as f:
                            json.dump(merged, f, indent=2)
                        idx += 1

            # Case 1: binpara was a single non-list (already normalized to list of len 1)
            elif len(binpara) == 1:
                item = binpara[0]
                if isinstance(item, int):
                    shutil.copy(os.path.join(path_to_binpara_template, f"offsettrace{item}.json"),
                                os.path.join(path_to_binpara, f"offsettrace{item}.json"))
                elif isinstance(item, str):
                    shutil.copy(os.path.join(path_to_binpara_template, f"{item}.json"),
                                os.path.join(path_to_binpara, f"{item}.json"))
                elif isinstance(item, dict):
                    merged = dict(_baseline_dict); merged.update(item or {})
                    with open(os.path.join(path_to_binpara, "customizedbinpara1.json"), "w") as f:
                        json.dump(merged, f, indent=2)
                else:
                    shutil.copy(os.path.join(path_to_binpara_template, "offsettrace19.json"),
                                os.path.join(path_to_binpara, "offsettrace19.json"))
            else:
                shutil.copy(os.path.join(path_to_binpara_template, "offsettrace19.json"),
                            os.path.join(path_to_binpara, "offsettrace19.json"))

        # --------------------------------------------------------------
        # ---------------- BLOCKCUT / BLOCKPARA / CONFIG ----------------
        # --------------------------------------------------------------
        path_to_blockcut = os.path.join(new_folder_path, "Analysis Parameters", "blockcut")
        path_to_blockcut_template = r"C:\ACMEcode\ACMEanalysis\templates\blockcut"
        if blockcut is not None:
            for file in os.listdir(path_to_blockcut):
                if file.endswith(".json"):
                    os.remove(os.path.join(path_to_blockcut, file))
            if not isinstance(blockcut, list):
                blockcut = [blockcut]
            for blockcut_instance in blockcut:
                if isinstance(blockcut_instance, str):
                    shutil.copy(os.path.join(path_to_blockcut_template, f"{blockcut_instance}.json"),
                                os.path.join(path_to_blockcut, f"{blockcut_instance}.json"))

        path_to_blockpara = os.path.join(new_folder_path, "Analysis Parameters", "blockpara")
        path_to_blockpara_template = r"C:\ACMEcode\ACMEanalysis\templates\blockpara"
        if blockpara is not None:
            for file in os.listdir(path_to_blockpara):
                if file.endswith(".json"):
                    os.remove(os.path.join(path_to_blockpara, file))
            if not isinstance(blockpara, list):
                blockpara = [blockpara]
            for blockpara_instance in blockpara:
                if isinstance(blockpara_instance, str):
                    shutil.copy(os.path.join(path_to_blockpara_template, f"{blockpara_instance}.json"),
                                os.path.join(path_to_blockpara, f"{blockpara_instance}.json"))

        path_to_config = os.path.join(new_folder_path, "Analysis Parameters", "config")
        path_to_config_template = r"C:\ACMEcode\ACMEanalysis\templates\config"
        if config is not None:
            for file in os.listdir(path_to_config):
                if file.endswith(".json"):
                    os.remove(os.path.join(path_to_config, file))
            if not isinstance(config, list):
                config = [config]
            for config_instance in config:
                if isinstance(config_instance, str):
                    shutil.copy(os.path.join(path_to_config_template, f"{config_instance}.json"),
                                os.path.join(path_to_config, f"{config_instance}.json"))

        # --------------------------------------------------------------
        # ------------------ SEQUENCE ASSIGNMENT ------------------------
        # --------------------------------------------------------------
        sequence_assignment_file = os.path.join(new_folder_path, "sequence assignment.json")
        if sequence_type is None:
            sequence_data = [
                ["PR", [[run, seq] for seq in sequence]],
                ["PR~D", [[run, seq] for seq in sequence]]
            ]
        elif isinstance(sequence_type, str):
            sequence_data = [[sequence_type, [[run, seq] for seq in sequence]]]
        else:
            sequence_data = [
                [sequence_type[i], [[run, seq] for seq in sequence]]
                for i in range(len(sequence_type))
            ]

        with open(sequence_assignment_file, "w") as json_file:
            json.dump(sequence_data, json_file, indent=4)

        print(f"Folder {new_folder_name} set up successfully and 'sequence assignment.json' updated.")
        return new_folder_path

    def grab_data(data_source_folder_path="X:\\ACME 3 EDM Data",
                header_source_folder_path="X:\\ACME 3 EDM Data",
                data_header_destination_path="C:\\ACMEdata",
                run=9,
                sequence=None):
        if run >= 13:
            data_source_folder_path = "Z:\\ACME 3 EDM Data"
            header_source_folder_path = "Z:\\ACME 3 EDM Data"

        # Validate sequence
        if sequence is None:
            print("No sequences provided. Exiting.")
            return

        # Ensure sequence is iterable
        if isinstance(sequence, int):
            sequence = [sequence]
        sequence = list(sequence)

        # Convert run to zero-padded string
        run_str = str(run).zfill(4)

        # Paths to data and header folders
        data_source_path = os.path.join(data_source_folder_path, run_str, "all_detectors")
        header_source_path = os.path.join(header_source_folder_path, run_str, "header_data")

        # Check source directories exist
        if not os.path.exists(data_source_path) or not os.path.exists(header_source_path):
            print(f"Source folders do not exist for run {run_str}. Exiting.")
            return

        seq_str = str(sequence[0]).zfill(4)
        data_dest_folder = os.path.join(data_header_destination_path, f"data{run_str}.{seq_str}")
        header_dest_folder = os.path.join(data_header_destination_path, f"header{run_str}.{seq_str}")
        os.makedirs(data_dest_folder, exist_ok=True)
        os.makedirs(header_dest_folder, exist_ok=True)

        total_skipped_data = 0
        total_skipped_header = 0

        # Helper function: check if file or its .0000-variant exists and has same size
        def file_exists_and_same_size(src_file, dest_folder):
            base_name = os.path.basename(src_file)
            name, ext = os.path.splitext(base_name)
            alt_name = f"{name}.0000{ext}"  # variant with ".0000" appended before extension

            dest_file_normal = os.path.join(dest_folder, base_name)
            dest_file_alt = os.path.join(dest_folder, alt_name)

            for dest_file in [dest_file_normal, dest_file_alt]:
                if os.path.exists(dest_file):
                    try:
                        src_size = os.path.getsize(src_file)
                        dest_size = os.path.getsize(dest_file)
                        if src_size == dest_size:
                            return True  # Same size — skip copying
                        else:
                            # Different size — replace old incomplete file
                            os.remove(dest_file)
                            return False
                    except OSError:
                        return False
            return False

        # Iterate over sequences
        for seq in sequence:
            seq_str = str(seq).zfill(4)

            # Copy files for data
            data_files = [f for f in os.listdir(data_source_path) if f.startswith(f"{run_str}.{seq_str}.")]
            print(f"Copying {len(data_files)} files to {data_dest_folder}...")
            for file in tqdm(data_files, desc=f"Copying data files for sequence {seq_str}"):
                src_file = os.path.join(data_source_path, file)
                if not file_exists_and_same_size(src_file, data_dest_folder):
                    shutil.copy(src_file, os.path.join(data_dest_folder, file))
                else:
                    total_skipped_data += 1

            # Copy files for header
            header_files = [f for f in os.listdir(header_source_path) if f.startswith(f"{run_str}.{seq_str}.")]
            print(f"Copying {len(header_files)} files to {header_dest_folder}...")
            for file in tqdm(header_files, desc=f"Copying header files for sequence {seq_str}"):
                src_file = os.path.join(header_source_path, file)
                if not file_exists_and_same_size(src_file, header_dest_folder):
                    shutil.copy(src_file, os.path.join(header_dest_folder, file))
                else:
                    total_skipped_header += 1

        print("Processing complete.")
        if total_skipped_data > 0 or total_skipped_header > 0:
            print(f"Warning: {total_skipped_data} data files and {total_skipped_header} header files were skipped because identical files already exist.")

        return data_dest_folder, header_dest_folder

    def _process_header_df(self, df, folder_path):
        
        def process_group(group):
            # For 'Lock Status' columns: set 0 if all zeros, else 1
            for col in group.columns:
                if 'Lock Status' in col:
                    group[col] = 0 if (group[col] == 0).all() else 1

            # For 'Ablation X' and 'Ablation Y', take the mode
            if 'Ablation X' in group.columns:
                group['Ablation X'] = group['Ablation X'].mode()[0] if not group['Ablation X'].mode().empty else None
            if 'Ablation Y' in group.columns:
                group['Ablation Y'] = group['Ablation Y'].mode()[0] if not group['Ablation Y'].mode().empty else None

            # For the rest of the columns, calculate the mean
            for col in group.columns:
                if col not in ['run', 'sequence', 'block', 'Ablation X', 'Ablation Y'] and 'Lock Status' not in col:
                    group[col] = group[col].mean()

            return group.iloc[0]  # Return the reduced single row


        # Step 1: Fill missing values
        # Fill 'Lock Status' columns with 0 if NaN, fill others with nearest value
        for col in df.columns:
            if 'Lock Status' in col:
                df[col].fillna(0, inplace=True)
            elif col.endswith('Current'):
                # Fill with same run, sequence, block value or nearest
                df[col] = df.groupby(['run', 'sequence', 'block'])[col].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
            else:
                df[col].fillna(method='ffill', inplace=True)
                df[col].fillna(method='bfill', inplace=True)

        drop_list = [
        "Start Time", "End Time", "total switch time", "E-fields", "B-fields", 
        "Freqs/Powers", "Waveplates", "dt", "Records per Trace", 
        "DAQ Voltage Range", "Number of Channels", "Conversion Factor Individual", 
        "Conversion Factor Summed", "Acquisition Rate", 
        "Polarization Switching Frequency", "Polarization Switching Deadtime", 
        "Polarization Switching Extra XY Delay", "Polarization Switching XY Swapped", 
        "Scope Trigger Offset", "Current Sequence Code ID",'Ablation Mirror Position X', 'Ablation Mirror Position Y', "dBzdx Main (mA)", "dBzdx Sub (mA)", "Bx-1 up (mA)", "Bx-1 down (mA)", 
        "Bx-2 up (mA)", "Bx-2 down (mA)", "Bx-3 up (mA)", "Bx-3 down (mA)", 
        "Bx-4 up (mA)", "Bx-4 down (mA)", "By center top current (mA)", 
        "By center bottom current (mA)", "dBydx up top current (mA)", 
        "dBydx up bottom current (mA)", "dBydx down top current (mA)", 
        "dBydx down bottom current (mA)", "By lattice (+++) (mA)", 
        "By lattice (++-) (mA)", "By lattice (+-+) (mA)", "By lattice (+--) (mA)", 
        "By lattice (-++) (mA)", "By lattice (-+-) (mA)", "By lattice (--+) (mA)", 
        "By lattice (---) (mA)","Bz Left (+z) current (mA)","Bz Right (-z) current (mA)","field plate voltage V_1",	"field plate voltage V_2","guard ring voltage V_1","guard ring voltage V_2"
    ]
        # Step 2: Drop unwanted columns
        contained_drop_list = [col for col in drop_list if col in df.columns]
        df.drop(columns=contained_drop_list, inplace=True)

        # Save the filled DataFrame to 'purged_aggregated_header.csv'
        purged_aggregated_header_path = os.path.join(folder_path, 'purged_aggregated_header.csv')
        df.to_csv(purged_aggregated_header_path, index=False)
        parity_transformed_df = self._logging_channels_parity_transform(df, exemption_list=[], nr_only_list=[])
        parity_transformed_df.to_csv(os.path.join(folder_path, 'parity_transformed_block_df.csv'), index=False)
        # Step 3: Reduce by removing 'trace' axis
        grouped_df = df.groupby(['run', 'sequence', 'block']).apply(lambda group: process_group(group)).reset_index(drop=True)

        # Save the grouped DataFrame to 'block_header.csv'
        block_header_path = os.path.join(folder_path, 'block_header.csv')
        grouped_df.to_csv(block_header_path, index=False)

    def _logging_channels_parity_transform(self, df, exemption_list = [], nr_only_list = [], nochange_list = []):
        """
        Generate a new DataFrame with parity-transformed columns based on N, E, theta, B.

        Parameters:
        df : pd.DataFrame
            The original dataframe with columns: run, sequence, block, trace, N, E, theta, B, and others.
        exemption_list : list
            List of column names to be excluded from parity transformation.
        nr_only_list : list, optional
            List of column names for which only non-reversing (nr) averaging is performed.
        nochange_list : list, optional
            List of column names to be included with simple averaging and keeping original names.

        Returns:
        pd.DataFrame
            New dataframe grouped by (run, sequence, block) with parity-transformed columns.
        """
        if nr_only_list is None:
            nr_only_list = []
        if nochange_list is None:
            nochange_list = []

        nochange_list_minimal = self._generate_nochange_list()
        nochange_list = list(set(nochange_list) | set(nochange_list_minimal))

        df = df.copy()
        if 'theta' in df.columns:
            df['Q'] = df['theta']

        available_labels = [label for label in ['N', 'E', 'Q', 'B'] if label in df.columns]

        transform_columns = [col for col in df.columns 
                            if col not in exemption_list and col not in nochange_list and col not in ['run', 'sequence', 'block', 'trace', 'N', 'E', 'theta', 'B']]

        group_keys = ['run', 'sequence', 'block']
        grouped = df.groupby(group_keys, sort=False)

        subsets = []
        subset_names = []
        for r in range(0, len(available_labels) + 1):
            for comb in combinations(available_labels, r):
                subsets.append(list(comb))
                if len(comb) == 0:
                    subset_names.append('nr')
                else:
                    subset_names.append(''.join(comb))

        def process_group(name_group):
            name, group = name_group
            base_row = {k: v for k, v in zip(group_keys, name)}
            for label in available_labels:
                base_row[label] = group[label].mean()

            group_label_products = {}
            for subset, subset_name in zip(subsets, subset_names):
                if subset_name == 'nr':
                    group_label_products[subset_name] = np.ones(len(group))
                else:
                    group_label_products[subset_name] = group[subset].prod(axis=1)

            for col in transform_columns:
                if col in nr_only_list or not available_labels:
                    weighted_avg = group[col].mean()
                    base_row[f"{col}_nr"] = weighted_avg
                else:
                    for subset_name in subset_names:
                        weighted_avg = (group[col] * group_label_products[subset_name]).mean()
                        base_row[f"{col}_{subset_name}"] = weighted_avg

            for col in nochange_list:
                if col in group.columns:
                    base_row[col] = group[col].mean()

            return base_row

        records = Parallel(n_jobs=-1)(delayed(process_group)(item) for item in grouped)

        result_df = pd.DataFrame.from_records(records)
        return result_df

    def _generate_nochange_list(self):
        """
        Generate nochange list from a master json file and corresponding sub-json files.

        Parameters:
        master_json_path : str
            Path to the master JSON file.
        folder_path : str
            Path to the folder containing sub JSON files.

        Returns:
        list
            Combined list of non_parity_switches and superblock_parity_switches.
        """
        master_json_path = os.path.join(self.run_folder, "sequence assignment.json")
        folder_path = os.path.join(self.run_folder, "Analysis Parameters", "sequence")
        with open(master_json_path, 'r') as f:
            master_data = json.load(f)

        collected_switches = set()

        for entry in master_data:
            subfile_name = entry[0]
            subfile_path = os.path.join(folder_path, f"{subfile_name}.json")
            if os.path.exists(subfile_path):
                with open(subfile_path, 'r') as subf:
                    sub_data = json.load(subf)
                    collected_switches.update(sub_data.get('non_parity_switches', []))
                    collected_switches.update(sub_data.get('superblock_parity_switches', []))

        return list(collected_switches)

    def create_run_folder(self, parent_folder = r"C:\ACME_analysis", run_name = r"default_name"):
        # Create the main run folder
        run_folder = os.path.join(parent_folder, run_name)
        
        if os.path.exists(run_folder):
            raise FileExistsError(f"The folder '{run_folder}' already exists. Aborting to avoid overwriting.")
        
        os.makedirs(run_folder)
        print(f"Created run folder: {run_folder}")
        
        # Create Run Results and subfolders
        run_results_path = os.path.join(run_folder, "Run Results")
        os.makedirs(os.path.join(run_results_path, "data"))
        os.makedirs(os.path.join(run_results_path, "figures"))
        
        # Create Analysis Parameters and subfolders
        analysis_params_path = os.path.join(run_folder, "Analysis Parameters")
        os.makedirs(os.path.join(analysis_params_path, "binpara"))
        os.makedirs(os.path.join(analysis_params_path, "bincut"))
        os.makedirs(os.path.join(analysis_params_path, "blockpara"))
        os.makedirs(os.path.join(analysis_params_path, "blockcut"))
        os.makedirs(os.path.join(analysis_params_path, "config"))
        os.makedirs(os.path.join(analysis_params_path, "sequence"))

        # Create Blinding Files folder
        blinding_files_path = os.path.join(run_folder, "Blinding Files")
        os.makedirs(blinding_files_path)
        
        # Store paths as object properties
        self.run_folder = run_folder
        self.run_results_data_path = os.path.join(run_results_path, "data")
        self.run_results_figures_path = os.path.join(run_results_path, "figures")
        self.binpara_folder_path = os.path.join(analysis_params_path, "binpara")
        self.bincut_folder_path = os.path.join(analysis_params_path, "bincut")
        self.blockpara_folder_path = os.path.join(analysis_params_path, "blockpara")
        self.blockcut_folder_path = os.path.join(analysis_params_path, "blockcut")
        self.config_folder_path = os.path.join(analysis_params_path, "config")
        self.sequence_folder_path = os.path.join(analysis_params_path, "sequence")
        self.blinding_folder_path = blinding_files_path

    def load_run_folder(self, run_folder):
        # Check for required subfolders
        expected_folders = [
            os.path.join(run_folder, "Run Results", "data"),
            os.path.join(run_folder, "Run Results", "figures"),
            os.path.join(run_folder, "Analysis Parameters", "binpara"),
            os.path.join(run_folder, "Analysis Parameters", "bincut"),
            os.path.join(run_folder, "Analysis Parameters", "blockpara"),
            os.path.join(run_folder, "Analysis Parameters", "blockcut"),
            os.path.join(run_folder, "Analysis Parameters", "config"),
            os.path.join(run_folder, "Analysis Parameters", "sequence"),
            os.path.join(run_folder, "Blinding Files")
        ]
        
        for folder in expected_folders:
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Required folder '{folder}' is missing.")
        
        # Set folder paths
        self.run_folder = run_folder
        self.run_results_data_path = os.path.join(run_folder, "Run Results", "data")
        self.run_results_figures_path = os.path.join(run_folder, "Run Results", "figures")
        analysis_params_path = os.path.join(run_folder, "Analysis Parameters")
        self.binpara_folder_path = os.path.join(analysis_params_path, "binpara")
        self.bincut_folder_path = os.path.join(analysis_params_path, "bincut")
        self.blockpara_folder_path = os.path.join(analysis_params_path, "blockpara")
        self.blockcut_folder_path = os.path.join(analysis_params_path, "blockcut")
        self.config_folder_path = os.path.join(analysis_params_path, "config")
        self.sequence_folder_path = os.path.join(analysis_params_path, "sequence")
        self.blinding_folder_path = os.path.join(run_folder, "Blinding Files")

    def initialize_run_folder(self):
        # Get all JSON files from relevant folders
        binpara_files = [f for f in os.listdir(self.binpara_folder_path) if f.endswith('.json')]
        bincut_files = [f for f in os.listdir(self.bincut_folder_path) if f.endswith('.json')]
        blockpara_files = [f for f in os.listdir(self.blockpara_folder_path) if f.endswith('.json')]
        blockcut_files = [f for f in os.listdir(self.blockcut_folder_path) if f.endswith('.json')]
        config_files = [f for f in os.listdir(self.config_folder_path) if f.endswith('.json')]

        # Generate all possible combinations of JSON files
        combinations = list(product(binpara_files, bincut_files, blockpara_files, blockcut_files, config_files))
        
        data = []
        for comb in combinations:
            binpara, bincut, blockpara, blockcut, config = comb
            sub_folder_name = f"binpara_{binpara.split('.')[0]}_bincut_{bincut.split('.')[0]}_blockpara_{blockpara.split('.')[0]}_blockcut_{blockcut.split('.')[0]}_config_{config.split('.')[0]}"
            sub_result_data_folder = os.path.join(self.run_results_data_path, sub_folder_name)
            sub_result_figure_folder = os.path.join(self.run_results_figures_path, sub_folder_name)
            
            # Create necessary subfolders if they don't exist
            os.makedirs(os.path.join(sub_result_data_folder, "Binary Results"), exist_ok=True)
            os.makedirs(os.path.join(sub_result_data_folder, "Block Results"), exist_ok=True)
            os.makedirs(os.path.join(sub_result_data_folder, "Sequence Results"), exist_ok=True)
            os.makedirs(os.path.join(sub_result_data_folder, "Run Results"), exist_ok=True)
            os.makedirs(os.path.join(sub_result_figure_folder, "Binary Results"), exist_ok=True)
            os.makedirs(os.path.join(sub_result_figure_folder, "Block Results"), exist_ok=True)
            os.makedirs(os.path.join(sub_result_figure_folder, "Sequence Results"), exist_ok=True)
            os.makedirs(os.path.join(sub_result_figure_folder, "Run Results"), exist_ok=True)
            
            # Collect relevant information for the DataFrame
            data.append({
                'binpara': binpara,
                'bincut': bincut,
                'blockpara': blockpara,
                'blockcut': blockcut,
                'config': config,
                'binpara_json_path': os.path.join(self.binpara_folder_path, binpara),
                'bincut_json_path': os.path.join(self.bincut_folder_path, bincut),
                'blockpara_json_path': os.path.join(self.blockpara_folder_path, blockpara),
                'blockcut_json_path': os.path.join(self.blockcut_folder_path, blockcut),
                'config_json_path': os.path.join(self.config_folder_path, config),
                'sub_result_data_folder_path': sub_result_data_folder,
                'sub_result_figure_folder_path': sub_result_figure_folder,
                'binresult_data_folder_path': os.path.join(sub_result_data_folder, "Binary Results"),
                'binresult_figure_folder_path': os.path.join(sub_result_figure_folder, "Binary Results"),
                'blockresult_data_folder_path': os.path.join(sub_result_data_folder, "Block Results"),
                'blockresult_figure_folder_path': os.path.join(sub_result_figure_folder, "Block Results"),
                'sequenceresult_data_folder_path': os.path.join(sub_result_data_folder, "Sequence Results"),
                'sequenceresult_figure_folder_path': os.path.join(sub_result_figure_folder, "Sequence Results"),
                'runresult_data_folder_path': os.path.join(sub_result_data_folder, "Run Results"),
                'runresult_figure_folder_path': os.path.join(sub_result_figure_folder, "Run Results")
            })
        
        # Create DataFrame and assign to path_df
        self.path_df = pd.DataFrame(data)

    def create_aggregated_header(self, folder_list):
        """
        This method runs the headerHandler, generates the DataFrame, 
        and saves it as 'aggregated header.csv' in the run folder.
        """
        # Run headerHandler on the provided folder list
        df = hh.headerHandler(folder_list)
        if "Q" in df.columns:
            df.rename(columns={"Q": "theta"}, inplace=True)
        # Save DataFrame as 'aggregated header.csv' in the run folder

        aggregated_header_path = os.path.join(self.run_folder, 'aggregated header.csv')
                # Input Omega B information
        """
        if 'Bz Left (+z) current (mA)' in df.columns and 'Bz Right (-z) current (mA)' in df.columns:
            df['phi_B_over_tau'] = 4.838052564 * np.abs(df['Bz Left (+z) current (mA)'] + df['Bz Right (-z) current (mA)'])
        else:
            df['phi_B_over_tau'] = 1
            print("WARNING: Omega B information not found in the header, setting phi_B_over_tau to 1.")
        """

        # Define 'x' as the sum of the relevant columns
        try:
            df['x_dontusethis_xxx'] = df['Bz Left (+z) current (mA)'] + df['Bz Right (-z) current (mA)']
            # Group by 'run', 'trace', 'block', and 'sequence' to calculate 'phi_B_over_tau' for each group
            phi_values = df.groupby(['run', 'block', 'sequence']).apply(
                lambda g: 4.838052564 * (g.loc[g['B'] == 1, 'x_dontusethis_xxx'].mean() - g.loc[g['B'] == -1, 'x_dontusethis_xxx'].mean())/2
            ).rename('phi_B_over_tau').reset_index()

            # Merge the calculated values back into the original DataFrame
            df = df.merge(phi_values, on=['run', 'block', 'sequence'])

            # Drop the temporary 'x' column if you no longer need it
            df = df.drop(columns=['x_dontusethis_xxx'])
        except:
            df['phi_B_over_tau'] = 51.75

        try:
            df['XY_power_imbalance'] = df['Readout X Power'] / df['Readout Y Channel']
            df['XY_power_sum'] = (df['Readout X Power'] + df['Readout Y Channel'])/200
        except:
            pass
        
        if 'D' in df.columns:
            df['Dflat'] = (
                df[['run', 'sequence', 'D']]
                .drop_duplicates()
                .sort_values(['run', 'sequence', 'D'])
                .reset_index(drop=True)
                .reset_index()
                .merge(df, on=['run', 'sequence', 'D'], how='right')
                .sort_index()  # restore original order
                ['index']
            )
        else:
            df['Dflat'] = (
                df[['run', 'sequence']]
                .drop_duplicates()
                .sort_values(['run', 'sequence'])
                .reset_index(drop=True)
                .reset_index()
                .merge(df, on=['run', 'sequence'], how='right')
                .sort_index()  # restore original order
                ['index']
            )


        df.to_csv(aggregated_header_path, index=False)
        print(f"Aggregated header saved at: {aggregated_header_path}")
        


        # Store the DataFrame in the runHandler instance as a property
        self.df = df
        
        self._process_header_df(df, self.run_folder)
        self.purged_df = pd.read_csv(os.path.join(self.run_folder, 'purged_aggregated_header.csv'))
        self.block_df = pd.read_csv(os.path.join(self.run_folder, 'block_header.csv'))

    def load_aggregated_header(self):
        """
        This method loads 'aggregated header.csv' from the run folder 
        into the 'self.df' property.
        """
        aggregated_header_path = os.path.join(self.run_folder, 'aggregated header.csv')
        
        if os.path.exists(aggregated_header_path):
            # Load the CSV file into a DataFrame and store it in self.df
            self.df = pd.read_csv(aggregated_header_path)
            print(f"Aggregated header loaded from: {aggregated_header_path}")
        else:
            raise FileNotFoundError(f"The file 'aggregated header.csv' does not exist at {aggregated_header_path}")
        
        if os.path.exists(os.path.join(self.run_folder, 'block_header.csv')):
            self.block_df = pd.read_csv(os.path.join(self.run_folder, 'block_header.csv'))
        else:
            raise FileNotFoundError(f"The file 'block_header.csv' does not exist at {self.run_folder}")
        
        if os.path.exists(os.path.join(self.run_folder, 'purged_aggregated_header.csv')):
            self.purged_df = pd.read_csv(os.path.join(self.run_folder, 'purged_aggregated_header.csv'))
        else:
            raise FileNotFoundError(f"The file 'purged_aggregated_header.csv' does not exist at {self.run_folder}")
        
    def create_sequence_assignment(self, sequence_assignment):
        """
        This method takes a sequence assignment list, verifies the existence
        of corresponding JSON files, saves the assignment to 'sequence assignment.json', 
        and generates 'run_sequence_range' with all run-sequence pairs.
        """
        # Ensure all sequence types have corresponding JSON files, if not then create
        for sequence_type, _ in sequence_assignment:
            json_file_path = os.path.join(self.sequence_folder_path, f"{sequence_type}.json")
            if not os.path.exists(json_file_path):
                print(f"Required JSON file '{json_file_path}' not found for sequence type '{sequence_type}'.")
                # create the json
                with open(json_file_path, 'w') as f:
                    json.dump({"non_parity_switches" : [],"superblock_parity_switches" : ["P", "L", "R"]}, f, indent=4)

        # Store the sequence assignment in a property
        self.sequence_assignment = sequence_assignment

        # Generate the run_sequence_range, which is a flat list of all [run, sequence] pairs
        self.run_sequence_range = []
        for _, run_sequence_pairs in sequence_assignment:
            self.run_sequence_range.extend(run_sequence_pairs)

        # Save the sequence assignment to 'sequence assignment.json' in the run folder
        sequence_assignment_path = os.path.join(self.run_folder, 'sequence assignment.json')
        with open(sequence_assignment_path, 'w') as f:
            json.dump(sequence_assignment, f, indent=4)
        print(f"Sequence assignment saved to: {sequence_assignment_path}")

    def load_sequence_assignment(self):
        """
        This method loads 'sequence assignment.json' from the run folder and 
        sets the 'run_sequence_range' property with all [run, sequence] pairs.
        """
        sequence_assignment_path = os.path.join(self.run_folder, 'sequence assignment.json')

        if os.path.exists(sequence_assignment_path):
            # Load the sequence assignment from the JSON file
            with open(sequence_assignment_path, 'r') as f:
                self.sequence_assignment = json.load(f)
            
            # Generate the run_sequence_range based on the loaded sequence assignment
            self.run_sequence_range = []
            for _, run_sequence_pairs in self.sequence_assignment:
                self.run_sequence_range.extend(run_sequence_pairs)
            
            print(f"Sequence assignment loaded from: {sequence_assignment_path}")
        else:
            raise FileNotFoundError(f"'sequence assignment.json' not found at {sequence_assignment_path}")
        
    def set_binary_data_folder_paths(self, folder_paths):
        """
        This method sets the self.binary_data_folder_paths to the provided list of folder paths.
        """
        self.binary_data_folder_paths = folder_paths
        print(f"Binary data folder paths set to: {self.binary_data_folder_paths}")

    def touch(self):
        """
        Rename files from xxxx.xxxx.xxxx.bin to xxxx.xxxx.xxxx.0000.bin
        """
        print("Touching files...")
        for folder in self.binary_data_folder_paths:
            for filename in os.listdir(folder):
                print(filename, end = "|")
                if filename.endswith('.bin') and len(filename) == 18 and filename.count('.') == 3:
                    # filename is of the form xxxx.xxxx.xxxx.bin
                    new_filename = filename.replace('.bin', '.0000.bin')
                    source_path = os.path.join(folder, filename)
                    target_path = os.path.join(folder, new_filename)

                    if not os.path.exists(target_path):
                        os.rename(source_path, target_path)
                    else:
                        print(f"Collision detected: {target_path} already exists, skipping.")
        print("", end = "\n")

    def untouch(self):
        """
        Rename files from xxxx.xxxx.xxxx.0000.bin back to xxxx.xxxx.xxxx.bin
        """
        for folder in self.binary_data_folder_paths:
            for filename in os.listdir(folder):
                if filename.endswith('.0000.bin') and len(filename) == 23 and filename.count('.') == 4:
                    # filename is of the form xxxx.xxxx.xxxx.0000.bin
                    new_filename = filename.replace('.0000.bin', '.bin')
                    source_path = os.path.join(folder, filename)
                    target_path = os.path.join(folder, new_filename)

                    if not os.path.exists(target_path):
                        os.rename(source_path, target_path)
                    else:
                        print(f"Collision detected: {target_path} already exists, skipping.")

    def calculate_bin_result(self, parallel=False, overwrite=False):
        print("Calculating bin result...")
        """
        This method calculates the bin result based on binpara, bincut, blockpara, blockcut, and config.
        It scans through the binary files in self.binary_data_folder_paths and filters files based on run_sequence_range.
        Results are calculated for different binpara and saved to different binresult_data_folder_path.

        If parallel=True, the processing of binary files is done in parallel using ThreadPoolExecutor.
        If overwrite=False, skips processing of bin files whose all output results already exist.
        Any errors in processing individual bin files will be caught and logged without interrupting the flow.
        """

        def process_bin_file(bin_file, binpara_json_path, binpara_group):
            """
            Function to process a single binary file, perform the calculation, and save the results.
            Skips processing if all expected output files already exist and overwrite=False.
            """
            file_name = os.path.basename(bin_file)
            run_num, seq_num, block_num, trace_offset = [int(x) for x in file_name.split('.')[:4]]

            # Check if the [run_num, seq_num] pair is in the run_sequence_range
            if [run_num, seq_num] not in self.run_sequence_range:
                return  # Skip if not in range

            # Determine expected output file name
            expected_output_name = f"binresult_{os.path.splitext(file_name)[0]}.pkl"

            # Check if all target output files already exist
            all_exist = True
            for _, row in binpara_group.iterrows():
                output_folder = row['binresult_data_folder_path']
                output_file = os.path.join(output_folder, expected_output_name)
                if not os.path.exists(output_file):
                    all_exist = False
                    break

            if all_exist and not overwrite:
                print(f"Skipping {file_name}: all outputs exist.")
                return  # Skip this bin file entirely

            # Slice self.df to get the DataFrame for the current run, sequence, and block
            df_slice = self.df[
                (self.df['run'] == run_num)
                & (self.df['sequence'] == seq_num)
                & (self.df['block'] == block_num)
            ]

            # Perform calculation using binCalculator
            calculator = binCalculator(bin_file, binpara_json_path, df_slice)
            calculator.default_pipeline()

            # Save to all target folders for this binpara
            for _, row in binpara_group.iterrows():
                binresult_data_folder_path = row['binresult_data_folder_path']
                os.makedirs(binresult_data_folder_path, exist_ok=True)
                calculator.saveBinResults(binresult_data_folder_path)

            # Cleanup
            del calculator
            gc.collect()

        # Loop through each binpara group
        for binpara in self.path_df['binpara'].unique():
            binpara_group = self.path_df[self.path_df['binpara'] == binpara]
            binpara_json_path = binpara_group['binpara_json_path'].iloc[0]

            for binary_folder in self.binary_data_folder_paths:
                bin_files = glob.glob(os.path.join(binary_folder, '*.bin'))

                if parallel:
                    # Parallel processing
                    with ThreadPoolExecutor() as executor:
                        futures = []
                        for bin_file in bin_files:
                            futures.append(executor.submit(process_bin_file, bin_file, binpara_json_path, binpara_group))

                        # Handle results and errors gracefully
                        for future in futures:
                            try:
                                future.result()  # Raises any exceptions inside threads
                            except Exception as e:
                                print(f"[Warning] Failed processing file in parallel: {e}")
                else:
                    # Sequential processing
                    for bin_file in tqdm(bin_files, desc=f"Processing binpara {binpara}"):
                        try:
                            process_bin_file(bin_file, binpara_json_path, binpara_group)
                        except Exception as e:
                            print(f"[Warning] Failed processing {os.path.basename(bin_file)}: {e}")

    def cut_bin(self):
        """
        This method processes pre-calculated bin results and generates cuts using binCutter.
        It loops through every binresult_data_folder_path, locates .pkl files, and applies bin cuts.
        """
        # Loop through each binpara, bincut, blockpara, blockcut, and config without grouping
        for idx, row in self.path_df.iterrows():
            binpara = row['binpara']
            bincut = row['bincut']
            blockpara = row['blockpara']
            blockcut = row['blockcut']
            config = row['config']
            
            bincut_json_path = row['bincut_json_path']
            binresult_data_folder_path = row['binresult_data_folder_path']
            
            # Locate all .pkl files in the binresult_data_folder_path
            pkl_files = glob.glob(os.path.join(binresult_data_folder_path, 'binresult_*.pkl'))
            
            for pkl_file in pkl_files:
                # Extract run and sequence numbers from the file name
                file_name = os.path.basename(pkl_file)
                run_num, seq_num, block_num, trace_offset = [int(x) for x in file_name.split('_')[1].split('.')[:4]]
                
                # Create binCutter and process the result
                cutter = binCutter(
                    binresult_file_path=pkl_file,
                    bincut_file_path=bincut_json_path,
                    bincutresult_folder_path=binresult_data_folder_path
                )
                
                # binCutter automatically calculates and saves the results
                del cutter
                gc.collect()

    def visualize_bin_result(self, only_one=False):
        """
        This method visualizes the bin result based on binpara, bincut, blockpara, blockcut, and config.
        It scans through the binary result files and visualizes them using the binVisualizer.
        Figures are saved to the corresponding 'binresult_figure_folder_path'.
        
        If only_one=True, only a single randomly selected file is visualized for each combination of binpara, bincut, blockpara, blockcut, and config.
        """
        # Loop through each combination of binpara, bincut, blockpara, blockcut, and config
        for idx, row in self.path_df.iterrows():
            binpara = row['binpara']
            binpara_json_path = row['binpara_json_path']
            bincut_json_path = row['bincut_json_path']
            binresult_data_folder_path = row['binresult_data_folder_path']
            binresult_figure_folder_path = row['binresult_figure_folder_path']

            # Locate all binresult_xxxx.xxxx.xxxx.xxxx.pkl files in the binresult_data_folder_path
            binresult_files = glob.glob(os.path.join(binresult_data_folder_path, 'binresult_*.pkl'))

            print(f"Bin Visualizer: Found {len(binresult_files)} .pkl files in {binresult_data_folder_path}")

            if only_one and binresult_files:
                # Select one random file if only_one is True
                binresult_files = [random.choice(binresult_files)]

            for binresult_file in tqdm(binresult_files):
                # Extract run, sequence, block, and trace_offset numbers from the file name
                file_name = os.path.basename(binresult_file)
                run_num, seq_num, block_num, trace_offset = [int(x) for x in file_name.split('_')[1].split('.')[:4]]
                
                # Expect a corresponding bincutresult_xxxx.xxxx.xxxx.xxxx.pkl in the same folder
                bincutresult_file_path = os.path.join(binresult_data_folder_path, f'bincutresult_{run_num:04}.{seq_num:04}.{block_num:04}.{trace_offset:04}.pkl')
                
                if not os.path.exists(bincutresult_file_path):
                    print(f"Warning: Corresponding bincut result file {bincutresult_file_path} not found. Skipping visualization for this result.")
                    continue

                # Create binVisualizer and process the visualization
                visualizer = binVisualizer(
                    parameter_file_path=binpara_json_path,
                    binresult_file_path=binresult_file,
                    bincut_file_path=bincut_json_path,
                    bincutresult_file_path=bincutresult_file_path,
                    figure_folder_path=binresult_figure_folder_path
                )
                visualizer.close_all_figures()
                del visualizer
                gc.collect()
                # print(f"Visualizing file: {binresult_file} with binpara: {binpara}")
                # The binVisualizer automatically creates and saves figures to the figure folder

            print(f"Finished visualizing all files for binpara: {binpara}, bincut: {row['bincut']}, blockpara: {row['blockpara']}, blockcut: {row['blockcut']}, config: {row['config']}")
    
    def calculate_block_result(self):
        """
        This method calculates block-level results using blockCalculator.
        It scans through binresult_xxxx.xxxx.xxxx.xxxx.pkl and bincutresult_xxxx.xxxx.xxxx.xxxx.pkl files,
        groups them by run, sequence, and block, and passes the appropriate lists to blockCalculator.
        Errors encountered during block calculations are caught and reported without stopping the pipeline.
        """
        blind_path = os.path.join(self.run_folder, 'Blinding Files')  # The blinding folder path

        # Loop through each row in path_df to process binresult_data_folder_path
        print("Calculating block result...")
        for idx, row in self.path_df.iterrows():
            blockpara_json_path = row['blockpara_json_path']
            binresult_data_folder_path = row['binresult_data_folder_path']
            blockresult_data_folder_path = row['blockresult_data_folder_path']  # Correct target folder for output

            # Locate all binresult and bincutresult files in the binresult_data_folder_path
            binresult_files = glob.glob(os.path.join(binresult_data_folder_path, 'binresult_*.pkl'))
            bincutresult_files = glob.glob(os.path.join(binresult_data_folder_path, 'bincutresult_*.pkl'))

            # Group files by run, sequence, and block
            grouped_files = {}
            for bin_file in binresult_files:
                # Extract run, sequence, block, and trace_offset
                file_name = os.path.basename(bin_file)
                run_num, seq_num, block_num, trace_offset = [int(x) for x in file_name.split('_')[1].split('.')[:4]]

                key = (run_num, seq_num, block_num)
                if key not in grouped_files:
                    grouped_files[key] = {'binresults': [], 'bincutresults': []}
                grouped_files[key]['binresults'].append(bin_file)

            # Do the same for bincutresult files
            for cut_file in bincutresult_files:
                file_name = os.path.basename(cut_file)
                run_num, seq_num, block_num, trace_offset = [int(x) for x in file_name.split('_')[1].split('.')[:4]]
                key = (run_num, seq_num, block_num)
                if key in grouped_files:
                    grouped_files[key]['bincutresults'].append(cut_file)

            # Process each (run, sequence, block) group
            for (run, sequence, block), files in tqdm(grouped_files.items()):
                # Sort files by trace_offset for both binresults and bincutresults
                files['binresults'].sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[3]))
                files['bincutresults'].sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[3]))

                # Filter self.df for the current run, sequence, and block
                df_block = self.df[
                    (self.df['run'] == run)
                    & (self.df['sequence'] == sequence)
                    & (self.df['block'] == block)
                ]
                
                if df_block.empty:
                    print(f"Warning: No corresponding data found in self.df for run {run}, sequence {sequence}, block {block}. Skipping.")
                    continue

                # Calculate phi_B_over_tau as the mean of df['phi_B_over_tau'] for the current group
                phi_B_over_tau = df_block['phi_B_over_tau'].mean()

                # Try executing block calculation safely
                try:
                    calculator = blockCalculator(
                        blockpara_json_path=blockpara_json_path,
                        binresults_path_list=files['binresults'],
                        bincutresult_path_list=files['bincutresults'],
                        blockresult_path=blockresult_data_folder_path,
                        df=df_block,
                        phi_B_over_tau=phi_B_over_tau,
                        blind_path=blind_path
                    )
                    # blockCalculator automatically writes its results to disk
                except Exception as e:
                    print(f"[Warning] Block calculation failed for run {run}, sequence {sequence}, block {block}: {e}")
                    continue  # Proceed with the next group

    def cut_block(self):
        """
        This method runs block-level cuts using blockCutter.
        It iterates over blockresult_xxxx.xxxx.xxxx.pkl files in the blockresult_folder_path and applies block cuts.
        """
        # Loop through each row in path_df to process blockresult_folder_path
        for idx, row in self.path_df.iterrows():
            blockresult_folder_path = row['blockresult_data_folder_path']  # Block result folder
            blockcut_json_path = row['blockcut_json_path']  # Block cut rule JSON
            blockcutresult_folder_path = blockresult_folder_path  # Output folder is the same as blockresult_folder_path

            # Locate all blockresult_xxxx.xxxx.xxxx.pkl files in the blockresult_folder_path
            blockresult_files = glob.glob(os.path.join(blockresult_folder_path, 'blockresult_*.pkl'))

            print(f"Block cutter: Found {len(blockresult_files)} blockresult files in {blockresult_folder_path}")

            # Process each blockresult file
            for blockresult_file in blockresult_files:
                # Extract run, sequence, and block numbers from the file name
                file_name = os.path.basename(blockresult_file)
                run_num, seq_num, block_num = [int(x) for x in file_name.split('_')[1].split('.')[:3]]

                # Filter self.df for the current run, sequence, and block
                df_block = self.df[(self.df['run'] == run_num) & (self.df['sequence'] == seq_num) & (self.df['block'] == block_num)]
                
                if df_block.empty:
                    print(f"Warning: No corresponding data found in self.df for run {run_num}, sequence {seq_num}, block {block_num}. Skipping.")
                    continue

                # Invoke blockCutter with the appropriate parameters
                # print(f"Applying block cut for run {run_num}, sequence {seq_num}, block {block_num}")
                
                cutter = blockCutter(
                    blockresult_path=blockresult_file,
                    blockcutresult_folder_path=blockcutresult_folder_path,
                    block_cut_rule_json_path=blockcut_json_path,
                    df=df_block
                )

                # The blockCutter automatically writes the cut result files
                # print(f"Block cut applied for run {run_num}, sequence {seq_num}, block {block_num}.")

    def visualize_block_result(self, only_first=None, block_range=None, visualizeF = False, visualizeblockparity = True, visualizeSipm = True, visualizeDegenTrace = True, visualizeN = True,visualizeenvlopphi = False):
        """
        This method visualizes block-level results using blockVisualizer.
        It scans through blockresult_xxxx.xxxx.xxxx.pkl files in the blockresult_data_folder_path
        and generates plots, saving them to blockresult_figure_folder_path.
        
        If only_first is specified as an integer, it will visualize only the first 'only_first' number of files.
        If block_range is specified, it will only visualize blocks in the given 'xxxx.xxxx.xxxx' format.
        If both only_first and block_range are None, all files are visualized.
        """
        # Loop through each row in path_df to process blockresult_data_folder_path and blockresult_figure_folder_path
        for idx, row in self.path_df.iterrows():
            blockresult_folder_path = row['blockresult_data_folder_path']
            blockresult_figure_folder_path = row['blockresult_figure_folder_path']

            # Locate all blockresult_xxxx.xxxx.xxxx.pkl files in the blockresult_folder_path
            blockresult_files = glob.glob(os.path.join(blockresult_folder_path, 'blockresult_*.pkl'))

            print(f"Block Visualizer: Found {len(blockresult_files)} blockresult files in {blockresult_folder_path}")

            # Filter files based on block_range if provided
            if block_range is not None:
                blockresult_files = [
                    f for f in blockresult_files
                    if '.'.join(os.path.basename(f).split('_')[1].split('.')[:3]) in block_range
                ]
            
            # If only_first is set and block_range is None, limit the number of files to process
            if only_first is not None and block_range is None:
                blockresult_files = blockresult_files[:only_first]

            print("Block Visualizer: Found " + str(len(blockresult_files)) + " files to visualize")
            # Process each blockresult file

            for blockresult_file in tqdm(blockresult_files):
                # Extract run, sequence, and block numbers from the file name
                file_name = os.path.basename(blockresult_file)
                run_num, seq_num, block_num = [int(x) for x in file_name.split('_')[1].split('.')[:3]]

                # Create blockVisualizer object to generate and save the plot
                # print(f"Visualizing block result for run {run_num}, sequence {seq_num}, block {block_num}")
                blockcutresult_file_path = os.path.join(blockresult_folder_path, f'blockcutresult_{run_num:04}.{seq_num:04}.{block_num:04}.pkl')
                visualizer = blockVisualizer(
                    blockresult_file,
                    blockresult_figure_folder_path,
                    blockcutresult_file_path, 
                    visualizeF = visualizeF, 
                    visualizeblockparity = visualizeblockparity, 
                    visualizeSipm = visualizeSipm, 
                    visualizeDegenTrace = visualizeDegenTrace,
                    visualizeN = visualizeN,
                    visualizeenvlopphi = visualizeenvlopphi
                )
                # Automatically generates and saves the plot

                # Cleanup the visualizer to free memory
                del visualizer
                gc.collect()

                # print(f"Block visualization completed for run {run_num}, sequence {seq_num}, block {block_num}.")

    def calculate_sequence_result(self):
        """
        This method calculates sequence-level results using sequenceCalculator.
        It iterates through path_df and processes each sequence assignment group from self.sequence_assignment.
        Block and blockcut result files are grouped and passed to sequenceCalculator for calculation.
        """
        blind_path = os.path.join(self.run_folder, 'Blinding Files')  # Blinding folder path

        # Loop through each sequence group in sequence_assignment
        for index, (sequence_type, sequence_ranges) in enumerate(self.sequence_assignment):
            # Sequence JSON file
            sequence_json_path = os.path.join(self.sequence_folder_path, f"{sequence_type}.json")

            # Create a subfolder for sequence results: str(index) + '_' + sequence_type
            for idx, row in self.path_df.iterrows():
                sequenceresult_data_folder_path = row['sequenceresult_data_folder_path']
                sequence_result_folder_path = os.path.join(sequenceresult_data_folder_path, f"{index}_{sequence_type}")
                os.makedirs(sequence_result_folder_path, exist_ok=True)

                blockresult_files = []
                blockcutresult_files = []
                df_for_sequence = pd.DataFrame()

                # Loop through each blockresult_data_folder_path to gather block and cut results
                blockresult_folder_path = row['blockresult_data_folder_path']
                blockresult_files_in_folder = glob.glob(os.path.join(blockresult_folder_path, 'blockresult_*.pkl'))
                blockcutresult_files_in_folder = glob.glob(os.path.join(blockresult_folder_path, 'blockcutresult_*.pkl'))

                # Process each sequence range (e.g., [[9,66], [9,67]])
                for run_num, seq_num in sequence_ranges:
                    # Filter and sort the blockresult files for the current sequence range
                    for block_file in blockresult_files_in_folder:
                        file_name = os.path.basename(block_file)
                        file_run_num, file_seq_num, _ = [int(x) for x in file_name.split('_')[1].split('.')[:3]]
                        if file_run_num == run_num and file_seq_num == seq_num:
                            blockresult_files.append(block_file)

                    # Filter and sort the blockcutresult files for the current sequence range
                    for cut_file in blockcutresult_files_in_folder:
                        file_name = os.path.basename(cut_file)
                        file_run_num, file_seq_num, _ = [int(x) for x in file_name.split('_')[1].split('.')[:3]]
                        if file_run_num == run_num and file_seq_num == seq_num:
                            blockcutresult_files.append(cut_file)

                    # Filter the relevant portion of the DataFrame (self.df)
                    df_for_sequence = pd.concat([df_for_sequence, self.df[
                        (self.df['run'] == run_num) & (self.df['sequence'] == seq_num)
                    ]])

                # Sort the blockresult and blockcutresult lists lexicographically
                blockresult_files.sort()
                blockcutresult_files.sort()

                # Invoke sequenceCalculator
                print(f"Calculating sequence result for sequence group {index}, type {sequence_type}")
                
                calculator = sequenceCalculator(
                    sequence_json_path=sequence_json_path,
                    blockresults_path_list=blockresult_files,
                    blockcutresults_path_list=blockcutresult_files,
                    sequence_result_folder_path=sequence_result_folder_path,
                    df=df_for_sequence,
                    blind_path=blind_path,
                    sequence_name=f"{index}_{sequence_type}"
                )

                # The sequenceCalculator automatically writes the result files
                # print(f"Sequence result calculated for sequence group {index}, type {sequence_type}")
                del calculator
                gc.collect()

    def visualize_sequence_result(self, degenerate_blocks=False, additional_columns = [], groups = False, parity = False, quantity_of_interest = ['phi','C','omega'], new1d = True, fitting = True, non_parity_sw = None, plot_2d = False, axes2d = None, baseline = None):
        """
        This method visualizes sequence-level results using sequenceVisualizer.
        It scans through subfolders in sequenceresult_data_folder_path, finds corresponding sequence results, and creates
        figure folders to save visualizations.
        """
        # Loop through each row in path_df to process sequenceresult_data_folder_path and sequenceresult_figure_folder_path
        for idx, row in self.path_df.iterrows():
            sequenceresult_data_folder_path = row['sequenceresult_data_folder_path']
            sequenceresult_figure_folder_path = row['sequenceresult_figure_folder_path']

            # Get all subfolders in sequenceresult_data_folder_path
            subfolders = [f.path for f in os.scandir(sequenceresult_data_folder_path) if f.is_dir()]

            # Process each subfolder and locate the corresponding sequenceresult pkl file
            for subfolder in subfolders:
                folder_name = os.path.basename(subfolder)
                sequenceresult_file_path = os.path.join(subfolder, f'sequenceresult_{folder_name}.pkl')

                # Check if the sequence result file exists
                if not os.path.exists(sequenceresult_file_path):
                    print(f"Warning: Sequence result file not found: {sequenceresult_file_path}. Skipping visualization for this folder.")
                    continue

                # Create the corresponding folder in the figure folder path
                figure_folder_path = os.path.join(sequenceresult_figure_folder_path, folder_name)
                os.makedirs(figure_folder_path, exist_ok=True)

                # Find the corresponding sequence json file
                sequence_type = folder_name.split('_')[1]  # Extract the sequence type from the folder name
                sequence_json_path = os.path.join(self.sequence_folder_path, f"{sequence_type}.json")

                # Invoke sequenceVisualizer to generate the figures
                print(f"Visualizing sequence result for folder {folder_name}")
                
                visualizer = sequenceVisualizer(
                    sequenceresult_file_path=sequenceresult_file_path,
                    figure_folder_path=figure_folder_path,
                    sequence_json_path=sequence_json_path
                )
                if parity:
                    visualizer.visualize_final_results_1d()
                if degenerate_blocks:
                    visualizer.visualize_degenerate_blocks(self.block_df, additional_columns)
                if groups:
                    visualizer.visualize_groups()
                if new1d:
                    visualizer.visualize_and_fit_final_results_1d(quantity_of_interest, non_parity_sw = non_parity_sw, linear_fit = fitting)
                if plot_2d:
                    visualizer.visualize_2d(quantity_of_interest, axes2d = axes2d, baseline = None)
                # The sequenceVisualizer automatically generates and saves the figures
                del visualizer
                gc.collect()

                # print(f"Visualization completed for sequence result: {sequenceresult_file_path}")

    def bin_misalignment_check(binary_path):

        directory, filename = os.path.split(binary_path)
        name, ext = os.path.splitext(filename)

        # Initialize an indicator variable
        file_renamed = False

        # Check if the extension is '.bin' and the name has 14 characters
        if ext == '.bin' and len(name) == 14:
            # Pad with ".0000" and create the new full path
            new_filename = name + ".0000" + ext
            new_path = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(binary_path, new_path)
            print(f"File renamed to: {new_path}")
            
            # Set the indicator to True
            file_renamed = True
        else:
            new_path = binary_path

        df = pd.DataFrame()
        df['Polarization Switching XY Swapped'] = [0,0,0,0,0]
        a = binCalculator(binary_file_path=new_path, parameter_file_path=r"C:\ACME_analysis\binmisalignment_bench\offsettrace.json", df_slice=df)
        a.default_pipeline()
        a.saveBinResults(r"C:\ACME_analysis\binmisalignment_bench")
        filename = a.file_name[0:19]
        bin_result_path = r"C:\ACME_analysis\binmisalignment_bench\binresult_"  +filename+ r".pkl"
        a = binCutter(binresult_file_path=bin_result_path, bincut_file_path=r"C:\ACME_analysis\binmisalignment_bench\frac15.json", bincutresult_folder_path=r"C:\ACME_analysis\binmisalignment_bench")
        bin_cut_result_path = r"C:\ACME_analysis\binmisalignment_bench\bincutresult_" + filename + r".pkl"
        a = binVisualizer(parameter_file_path=r"C:\ACME_analysis\binmisalignment_bench\offsettrace.json", 
                          binresult_file_path=bin_result_path, bincut_file_path=r"C:\ACME_analysis\binmisalignment_bench\frac15.json",
                            bincutresult_file_path=bin_cut_result_path, 
                            figure_folder_path=r"C:\ACME_analysis\binmisalignment_bench")
        a.close_all_figures()
        del a
        gc.collect()


        # After the operation, check the indicator and revert the filename
        if file_renamed:
            # Revert the filename back to the original
            os.rename(new_path, binary_path)
            print(f"File name reverted to: {binary_path}")
