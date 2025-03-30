import pandas as pd
import re
import glob
import os
from datetime import datetime

class hh:
    def parse_config_file(filepath):
        # Regex patterns to identify the different line types
        header_pattern = r"header\s+(\d+\.\d+\.\d+\.\d+\.\d+)"
        # Updated value pattern to robustly handle all numerical formats
        value_pattern = r"^(?!==)([^\t]+?)\t+.*?([\-\d\.]+(?:[Ee][+-]?\d+)?)(\s*[\w\/]*)$"
        
        # Read the file and split into paragraphs using 'header' as the starting point of a new paragraph
        with open(filepath, 'r') as file:
            content = file.read()
        
        # DataFrame to hold all the data
        df = pd.DataFrame()
        paragraphs = re.split(r'\n(?=header)', content)

        for paragraph in paragraphs:
            data_dict = {}
            lines = paragraph.split('\n')
            for line in lines:
                line = line.strip()
                header_match = re.match(header_pattern, line)
                value_match = re.match(value_pattern, line)
                
                if header_match:
                    run, sequence, block, trace, _ = header_match.group(1).split('.')
                    data_dict.update({'run': run, 'sequence': sequence, 'block': block, 'trace': trace})
                elif line.startswith("Start Time") or line.startswith("End Time"):
                    parts = line.split('\t')
                    if len(parts) > 1:
                        timestamp = parts[1].strip()
                        try:
                            dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
                            prefix = "start" if "Start Time" in line else "end"
                            data_dict[f"{prefix} year"] = dt.year
                            data_dict[f"{prefix} month"] = dt.month
                            data_dict[f"{prefix} day"] = dt.day
                            data_dict[f"{prefix} hour"] = dt.hour
                            data_dict[f"{prefix} minute"] = dt.minute
                            data_dict[f"{prefix} second"] = dt.second + dt.microsecond / 1e6
                        except ValueError:
                            pass  # Ignore invalid datetime values
                elif value_match:
                    key = value_match.group(1).strip()
                    value = value_match.group(2).strip()
                    try:
                        float_val = float(value)
                        if float_val.is_integer():
                            value = int(float_val)
                        else:
                            value = float_val
                    except ValueError:
                        continue  # If conversion fails, skip adding this entry
                    if key not in data_dict:
                        data_dict[key] = value
                    else:
                        pass
            
            # Convert the dictionary to a DataFrame row and append it to the main DataFrame
            row_df = pd.DataFrame([data_dict])
            df = pd.concat([df, row_df], ignore_index=True)

        # Ensure all missing values are filled with NaN and reorder columns placing identifiers first
        identifier_cols = ['run', 'sequence', 'block', 'trace']
        other_cols = [col for col in df.columns if col not in identifier_cols]
        final_cols = identifier_cols + other_cols
        df = df.reindex(columns=final_cols)
        
        return df
    
    def headerHandler(list_of_target_folders):
        all_dfs = []  # This list will store all the DataFrames to be concatenated.
        
        # Iterate over each directory in the list
        for folder in list_of_target_folders:
            # Use glob to find all .txt files in the current folder
            txt_files = glob.glob(os.path.join(folder, '*.txt'))
            
            # Process each file found
            for file_path in txt_files:
                try:
                    # Parse the configuration file to a DataFrame
                    df = hh.parse_config_file(file_path)
                    all_dfs.append(df)  # Add the resulting DataFrame to the list
                except Exception as e:
                    print(f"Failed to process {file_path}: {str(e)}")
        
        # Concatenate all DataFrames into one, if any are found
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
        else:
            final_df = pd.DataFrame()  # Return an empty DataFrame if no files were processed
                
        final_df['run'] = final_df['run'].astype(int)
        final_df['sequence'] = final_df['sequence'].astype(int)
        final_df['block'] = final_df['block'].astype(int)
        final_df['trace'] = final_df['trace'].astype(int)

        return final_df