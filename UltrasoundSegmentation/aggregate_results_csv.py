import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-folder", type=str)
    parser.add_argument("--results-csv-filename", type=str, default="results.csv")
    parser.add_argument("--output-csv", type=str, default="results.csv")
    parser.add_argument("--metric_column_name", type=str, default="average")
    return parser.parse_args()


def main(args):
    # Get paths of results csv files
    model_names = next(os.walk(args.models_folder))[1]
    results_csv_paths = [os.path.join(args.models_folder, model_name, args.results_csv_filename) for model_name in model_names]

    # Read first csv file to create dataframe
    df = pd.read_csv(results_csv_paths[0])
    results_df = pd.DataFrame(columns=np.insert(df["class"].to_numpy(), 0, "model"))
    results_df = pd.DataFrame({"model": df["class"], f"{model_names[0]}": df[args.metric_column_name]})
    
    # Iterate through remaining csv files and add to dataframe
    for i in range(1, len(results_csv_paths)):
        df = pd.read_csv(results_csv_paths[i])
        results_df[f"{model_names[i]}"] = df[args.metric_column_name]

    # Transpose dataframe for better readability
    results_df = results_df.set_index("model").T

    # Save dataframe to csv
    output_path = os.path.join(args.models_folder, args.output_csv)
    results_df.to_csv(output_path, index_label="model")
    print(f"Saved results to {output_path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
