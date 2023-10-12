
import os
import argparse
import pandas as pd

def consolidate_metrics(input_csv, output_csv_path, usecol):
# Read the input CSV file which contains the list of result CSV files to aggregate
    input_df = pd.read_csv(input_csv)
    
    # Assume that the column holding the model file names is named 'model_file'
    csv_files = input_df['FileName'].tolist()
    
    # Initialize an empty dictionary to hold DataFrames for each AI model
    model_data_dict = {}
    
    # Assume that the result CSV files are in the same directory as input_csv
    input_folder = os.path.dirname(input_csv)

    # Loop through each CSV file and read it into a DataFrame
    for i, csv_file in enumerate(csv_files):
        model_name = os.path.splitext(os.path.basename(csv_file))[0]
        csv_file_path = os.path.join(input_folder, csv_file)
        
        # Check if usecol can be converted to an integer, if not, assume it's a column name
        try:
            usecol_int = int(usecol) + 1  # if it's an index, convert to int and adjust
            cols_to_use = [0, usecol_int]  # using integer indices
        except ValueError:
            cols_to_use = ['class', usecol]  # using column names
        
        df = pd.read_csv(csv_file_path, usecols=cols_to_use)
        
        # Store the DataFrame in the dictionary with the short name as the key
        model_data_dict[model_name] = df
    
    # Initialize an empty DataFrame to store the metrics for each model
    consolidated_metrics_df = pd.DataFrame()
    
    # Loop through the model data and extract metrics for class "1"
    for model_name, df in model_data_dict.items():
        metrics = df.set_index('class')[usecol].to_dict()
        metrics['Model'] = model_name
        metrics_df = pd.DataFrame([metrics])  # Convert dict to DataFrame
        consolidated_metrics_df = pd.concat([consolidated_metrics_df, metrics_df], ignore_index=True)

    
    # Reorder the columns to have 'Model' as the first column
    cols = ['Model'] + [col for col in consolidated_metrics_df.columns if col != 'Model']
    consolidated_metrics_df = consolidated_metrics_df[cols]
    
    # Save the consolidated metrics to a CSV file
    consolidated_metrics_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Consolidate metrics from multiple CSV files.')
    parser.add_argument('--input_csv', type=str, help='Path to the input folder containing CSV files.')
    parser.add_argument('--output_csv_path', type=str, help='Path to the output consolidated CSV file.')
    parser.add_argument('--usecol', type=str, help='Column number to be used for metrics.')
    
    args = parser.parse_args()
    consolidate_metrics(args.input_csv, args.output_csv_path, args.usecol)
