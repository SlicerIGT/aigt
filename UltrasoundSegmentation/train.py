"""
Train a u-net model on the ultrasound dataset.
For experiment tracking:
    - Save a copy of the configuration file and the trained model in the output folder
    - Log training metrics to a file or console
    - Log training metrics to Weights & Biases
"""

import argparse
import logging
import monai
import random
import traceback
import torch
import os
import sys
import yaml
import wandb
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from monai.data import DataLoader
from monai.data.utils import decollate_batch
from monai import transforms
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete
)
from monai.metrics import (
    DiceMetric, 
    MeanIoU,  
    ConfusionMatrixMetric
)

from UltrasoundDataset import UltrasoundDataset
from UNet import UNet


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-folder", type=str)
    parser.add_argument("--val-data-folder", type=str)
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--wandb-project-name", type=str, default="aigt_ultrasound_segmentation")
    parser.add_argument("--wandb-exp-name", type=str)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str)
    try:
        return parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)


def main(args):
    # Make sure output folder exists and save a copy of the configuration file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set up logging into file or console
    if args.log_file is not None:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        log_file = os.path.join(args.output_dir, args.log_file)
        logging.basicConfig(filename=log_file, filemode="w", level=args.log_level)
        print(f"Logging to file {log_file}.")
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Log to console

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
    if args.wandb_exp_name is not None:
        experiment_name = f"{args.wandb_exp_name}_{timestamp}"
    else:
        experiment_name = f"{config['model_name']}_{timestamp}"
    run = wandb.init(
        # Set the project where this run will be logged
        project=args.wandb_project_name,
        name=experiment_name,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": config["learning_rate"],
            "epochs": config["num_epochs"],
        })
    run.define_metric("val_loss", summary="min")
    run.define_metric("accuracy", summary="max")

    # Log values of the config dictionary on Weights & Biases
    wandb.config.update(config)

    # Set seed for reproducibility
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device == "cuda":
        torch.cuda.manual_seed(config["seed"])
    g = torch.Generator()
    g.manual_seed(config["seed"])

    # Create transforms
    train_transform_list = []
    val_transform_list = []
    if config["transforms"]["general"]:
        for tfm in config["transforms"]["general"]:
            try:
                train_transform_list.append(getattr(transforms, tfm["name"])(**tfm["params"]))
                val_transform_list.append(getattr(transforms, tfm["name"])(**tfm["params"]))
            except KeyError:  # Apply transform to both image and label by default
                train_transform_list.append(getattr(transforms, tfm["name"])(keys=["image", "label"]))
                val_transform_list.append(getattr(transforms, tfm["name"])(keys=["image", "label"]))
    if config["transforms"]["train"]:
        for tfm in config["transforms"]["train"]:
            try:
                train_transform_list.append(getattr(transforms, tfm["name"])(**tfm["params"]))
            except KeyError:
                train_transform_list.append(getattr(transforms, tfm["name"])(keys=["image", "label"]))
    train_transform = Compose(train_transform_list)
    val_transform = Compose(val_transform_list)

    # Create dataloaders using UltrasoundDataset
    train_dataset = UltrasoundDataset(args.train_data_folder, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, generator=g)
    val_dataset = UltrasoundDataset(args.val_data_folder, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, generator=g)

    # Construct model
    if config["model_name"] == "monai_unet":
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    else:
        model = UNet(in_channels=config["in_channels"], out_channels=config["out_channels"])
    
    # Construct loss function
    if config["loss_function"] == "monai_dice":
        loss_function = monai.losses.DiceLoss(sigmoid=True)
    else:
        loss_function = BCEWithLogitsLoss()

    model = model.to(device=device)

    # from torchinfo import summary
    # summary(model, input_size=(1, config["in_channels"], 128, 128))

    optimizer = Adam(model.parameters(), config["learning_rate"])

    # Set up learning rate decay
    try:
        learning_rate_decay_frequency = int(config["learning_rate_decay_frequency"])
    except ValueError:
        learning_rate_decay_frequency = 100
    try:
        learning_rate_decay_factor = float(config["learning_rate_decay_factor"])
    except ValueError:
        learning_rate_decay_factor = 1.0 # No decay
    logging.info(f"Learning rate decay frequency: {learning_rate_decay_frequency}")
    logging.info(f"Learning rate decay factor: {learning_rate_decay_factor}")

    # Metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    confusion_matrix_metric = ConfusionMatrixMetric(
        include_background=True, 
        metric_name=["accuracy", "precision", "sensitivity", "specificity", "f1_score"],
        reduction="mean"
    )
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Train model
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_frequency, gamma=learning_rate_decay_factor)
    for epoch in range(config["num_epochs"]):
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch in tqdm(train_dataloader):
            step += 1
            inputs = batch["image"].to(device=device)
            labels = batch["label"].to(device=device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        logging.info(f"Training loss: {epoch_loss}")
        scheduler.step()

        # Validation step
        model.eval()
        val_loss = 0
        val_step = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                val_step += 1
                val_inputs = batch["image"].to(device=device)
                val_labels = batch["label"].to(device=device)
                val_outputs = model(val_inputs)
                loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()
                
                # Compute metrics for current iteration
                val_post_preds = [post_pred(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_post_preds, y=val_labels)
                iou_metric(y_pred=val_post_preds, y=val_labels)
                confusion_matrix_metric(y_pred=val_post_preds, y=val_labels)

            val_loss /= val_step
            dice = dice_metric.aggregate().item()
            iou = iou_metric.aggregate().item()
            cm = confusion_matrix_metric.aggregate()

            # reset status for next validation round
            dice_metric.reset()
            iou_metric.reset()
            confusion_matrix_metric.reset()

            logging.info(f"Val loss: {val_loss}")
            logging.info(f"Dice: {dice}\n"
                        f"IoU: {iou}\n"
                        f"Accuracy: {(acc := cm[0].item())}\n"
                        f"Precision: {(pre := cm[1].item())}\n"
                        f"Sensitivity: {(sen := cm[2].item())}\n"
                        f"Specificity: {(spe := cm[3].item())}\n"
                        f"F1 score: {(f1 := cm[4].item())}")
        
        # Log a random sample of 3 test images along with their ground truth and predictions
        random.seed(config["seed"])
        sample = random.sample(range(len(val_dataset)), 5)

        inputs = torch.stack([val_dataset[i]["image"] for i in sample])
        labels = torch.stack([val_dataset[i]["label"] for i in sample])
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

        # Log metrics and examples together to maintain global step == epoch
        run.log({
            "train_loss": epoch_loss, 
            "val_loss": val_loss,
            "dice": dice,
            "iou": iou,
            "accuracy": acc,
            "precision": pre,
            "sensitivity": sen,
            "specificity": spe,
            "f1_score": f1,
            "examples": wandb.Image(fig)})

        plt.close(fig)

        # Log current learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            logging.info(f"Current learning rate: {current_lr}")

        # Save model after every Nth epoch as specified in the config file
        if config["save_frequency"]>0 and (epoch + 1) % config["save_frequency"] == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_{epoch+1}.pt"))

    # Log final metrics
    metric_table = wandb.Table(
        columns=["acc", "pre", "sen", "spe", "f1", "dice", "iou"], 
        data=[[acc, pre, sen, spe, f1, dice, iou]]
    )
    run.log({"metrics": metric_table})
        
    # Save final trained model in a self-contained way.
    # Generate a filename for the saved model that contains the run ID so that we can easily find the model corresponding to a given run.

    # model_filename = f"model_{run.id}.pt"
    # torch.save(model.state_dict(), os.path.join(args.output_dir, model_filename))

    # Save the final model also under the name "model.pt" so that we can easily find it later.
    # This is useful if we want to use the model for inference without having to specify the model filename.

    model_filename = "model.pt"
    torch.save(model.state_dict(), os.path.join(args.output_dir, model_filename))
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
