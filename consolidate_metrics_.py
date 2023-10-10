
import os
import argparse
import pandas as pd

def consolidate_metrics(input_folder, output_csv_path):
    # Initialize an empty dictionary to hold DataFrames for each AI model
    model_data_dict = {}
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    # Loop through each CSV file and read it into a DataFrame
    for i, csv_file in enumerate(csv_files):
        model_name = f"Model_{i+1}"
        csv_file_path = os.path.join(input_folder, csv_file)
        
        df = pd.read_csv(csv_file_path)
        
        # Drop the second and last columns, keeping only the column corresponding to class "1"
        df = df.drop(df.columns[[1, -1]], axis=1)
        
        # Store the DataFrame in the dictionary with the short name as the key
        model_data_dict[model_name] = df
    
    # Initialize an empty DataFrame to store the metrics for each model
    consolidated_metrics_df = pd.DataFrame()
    
    # Loop through the model data and extract metrics for class "1"
    for model_name, df in model_data_dict.items():
        metrics = df.set_index('class')['1'].to_dict()
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
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing CSV files.')
    parser.add_argument('--output_csv_path', type=str, help='Path to the output consolidated CSV file.')
    
    args = parser.parse_args()
    consolidate_metrics(args.input_folder, args.output_csv_path)
