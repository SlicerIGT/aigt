import os
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-folder", type=str)
    parser.add_argument("--results-csv-filename", type=str, default="results.csv")
    parser.add_argument("--sample-images-folder", type=str, default="samples")
    parser.add_argument("--output-csv", type=str, default="models.csv")
    return parser.parse_args()


def main(args):
    model_names = next(os.walk(args.models_folder))[1]
    model_paths = [os.path.join(args.models_folder, model_name, "model_traced.pt") for model_name in model_names]
    results_csv_paths = [os.path.join(args.models_folder, model_name, args.results_csv_filename) for model_name in model_names]
    sample_images_paths = [os.path.join(args.models_folder, model_name, args.sample_images_folder) for model_name in model_names]
    model_df = pd.DataFrame({
        "model_path": model_paths,
        "output_csv_file": results_csv_paths,
        "output_dir": sample_images_paths
    })
    output_path = os.path.join(args.models_folder, args.output_csv)
    model_df.to_csv(output_path, index=False)
    print(f"Saved model paths to {output_path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
