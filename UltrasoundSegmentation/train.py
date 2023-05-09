"""
Train a u-net model on the ultrasound dataset.
For experiment tracking:
    - Save a copy of the configuration file and the trained model in the output folder
    - Log training metrics to a file or console
    - Log training metrics to Weights & Biases
"""

import argparse
import logging
import random
import torch
import os
import yaml
import wandb
import matplotlib.pyplot as plt

from datetime import datetime
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from metrics import soft_iou
from UltrasoundDataset import UltrasoundDataset
from UNet import UNet


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-folder", type=str)
    parser.add_argument("--val-data-folder", type=str)
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str)
    return parser.parse_args()


def main(args):
    # Make sure output folder exists and save a copy of the configuration file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set up logging into file or console
    if args.log_file is not None:
        log_file = os.path.join(args.output_dir, args.log_file)
        logging.basicConfig(filename=log_file, filemode="w", level=args.log_level)
        print(f"Logging to file {log_file}.")
    else:
        logging.basicConfig(level=args.log_level)  # Log to console

    # Read config file
    if args.config_file is None:
        args.config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "train_config.yaml")

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    with open(os.path.join(args.output_dir, "train_config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device {device}.")

    # Initialize Weights & Biases
    wandb.login()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{config['experiment_name']}_{timestamp}"
    run = wandb.init(
        # Set the project where this run will be logged
        project=config["wandb_project_name"],
        name=experiment_name,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": config["learning_rate"],
            "epochs": config["num_epochs"],
        })

    # Set seed for reproducibility
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device == "cuda":
        torch.cuda.manual_seed(config["seed"])

    # Create dataloaders using UltrasoundDataset
    resize_transform = transforms.Resize((config["image_size"], config["image_size"]), antialias=True)
    train_dataset = UltrasoundDataset(args.train_data_folder, transform=resize_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
    val_dataset = UltrasoundDataset(args.val_data_folder, transform=resize_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = UNet(in_channels=config["in_channels"], out_channels=config["out_channels"])
    model = model.to(device=device)

    # from torchinfo import summary
    # summary(model, input_size=(1, config["in_channels"], 128, 128))

    loss_function = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), config["learning_rate"])

    for epoch in range(config["num_epochs"]):
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        model.train()
        train_losses = []
        for batch in tqdm(train_dataloader):
            inputs = batch[0].to(device=device)
            labels = batch[1].to(device=device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses)
        logging.info(f"Training loss: {train_loss}")

        # Validation step
        model.eval()
        test_losses = []
        test_ious = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                inputs = batch[0].to(device=device)
                labels = batch[1].to(device=device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                test_losses.append(loss.item())
                test_ious.append(soft_iou(outputs, labels))
        test_loss = sum(test_losses) / len(test_losses)
        test_iou = sum(test_ious) / len(test_ious)
        logging.info(f"Val loss: {test_loss}")
        logging.info(f"Val IoU: {test_iou}")
        wandb.log({"train_loss": train_loss, "val_loss": test_loss, "val_iou": test_iou})

        # Save model after every Nth epoch as specified in the config file
        if config["save_frequency"]>0 and (epoch + 1) % config["save_frequency"] == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_{epoch+1}.pt"))

        # Log a random sample of 3 test images along with their ground truth and predictions
        sample = random.sample(range(len(val_dataset)), 3)

        inputs = torch.stack([val_dataset[i][0] for i in sample])
        labels = torch.stack([val_dataset[i][1] for i in sample])
        with torch.no_grad():
            logits = model(inputs.to(device=device))
        outputs = torch.sigmoid(logits)

        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for i in range(3):
            axes[i, 0].imshow(inputs[i, 0, :, :], cmap="gray")
            axes[i, 1].imshow(labels[i].squeeze(), cmap="gray")
            im = axes[i, 2].imshow(outputs[i].squeeze().cpu().detach().numpy(), vmin=0, vmax=1, cmap="viridis")
            
            # Create an additional axis for the colorbar
            cax = fig.add_axes([axes[i, 2].get_position().x1 + 0.01,
                                axes[i, 2].get_position().y0,
                                0.02,
                                axes[i, 2].get_position().height])
            fig.colorbar(im, cax=cax)
        wandb.log({"examples": wandb.Image(fig)})
        
    # Save final trained model in a self-contained way.
    # Generate a filename for the saved model that contains the run ID so that we can easily find the model corresponding to a given run.

    # model_filename = f"model_{run.id}.pt"
    # torch.save(model.state_dict(), os.path.join(args.output_dir, model_filename))

    # Save the final model also under the name "model.pt" so that we can easily find it later.
    # This is useful if we want to use the model for inference without having to specify the model filename.

    model_filename = "model.pt"
    torch.save(model.state_dict(), os.path.join(args.output_dir, model_filename))
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
