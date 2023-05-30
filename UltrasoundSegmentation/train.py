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
import json
import yaml
import wandb
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
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
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--config-file", type=str, default="train_config.yaml")
    parser.add_argument("--save-torchscript", action="store_true")
    parser.add_argument("--save-ckpt-freq", type=int, default=0)
    parser.add_argument("--wandb-project-name", type=str, default="aigt_ultrasound_segmentation")
    parser.add_argument("--wandb-exp-name", type=str)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--save-log", action="store_true")
    try:
        return parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)


def main(args):
    # Load config file
    # If config file is not given, use default config
    # If path is not specified, look for config file in the same folder as this script
    if args.config_file is None:
        args.config_file = os.path.join(os.path.dirname(__file__), "train_config.yaml")
    else:
        if not os.path.isabs(args.config_file):
            args.config_file = os.path.join(os.path.dirname(__file__), args.config_file)
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    
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

    # Create directory for run output
    run_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)

    # Remove loggers from other libraries
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up logging to console
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-8s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(args.log_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    # Log to file if specified
    if args.save_log:
        log_file = os.path.join(run_dir, "train.log")
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Logging to file {log_file}.")
    
    # Save copy of config file in run folder
    config_copy_path = os.path.join(run_dir, "train_config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f)
        logging.info(f"Saved config file to {config_copy_path}.")
    
    # Create checkpoints folder if needed
    if args.save_ckpt_freq > 0:
        ckpt_dir = os.path.join(run_dir, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device {device}.")

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
                train_transform_list.append(
                    getattr(transforms, tfm["name"])(**tfm["params"])
                )
                val_transform_list.append(
                    getattr(transforms, tfm["name"])(**tfm["params"])
                )
            except KeyError:  # Apply transform to both image and label by default
                train_transform_list.append(
                    getattr(transforms, tfm["name"])(keys=["image", "label"])
                )
                val_transform_list.append(
                    getattr(transforms, tfm["name"])(keys=["image", "label"])
                )
    if config["transforms"]["train"]:
        for tfm in config["transforms"]["train"]:
            try:
                train_transform_list.append(
                    getattr(transforms, tfm["name"])(**tfm["params"])
                )
            except KeyError:
                train_transform_list.append(
                    getattr(transforms, tfm["name"])(keys=["image", "label"])
                )
    train_transform = Compose(train_transform_list)
    val_transform = Compose(val_transform_list)

    # Create dataloaders using UltrasoundDataset
    train_dataset = UltrasoundDataset(args.train_data_folder, transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=config["shuffle"], 
        generator=g
    )
    val_dataset = UltrasoundDataset(args.val_data_folder, transform=val_transform)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        generator=g
    )

    # Construct model
    if config["model_name"] == "monai_unet":
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    else:
        model = UNet(in_channels=config["in_channels"], out_channels=config["out_channels"])
    
    # Construct loss function
    if config["loss_function"] == "monai_dice":
        if config["out_channels"] == 1:
            loss_function = monai.losses.DiceLoss(sigmoid=True)
        else:
            loss_function = monai.losses.DiceLoss(to_onehot_y=False, softmax=True)
    elif config["loss_function"] == "monai_DiceCELoss":
        if config["out_channels"] == 1:
            loss_function = monai.losses.DiceCELoss(sigmoid=True)
        else:
            loss_function = monai.losses.DiceCELoss(to_onehot_y=False, softmax=True, lambda_dice=(1.0 - config["lambda_ce"]), lambda_ce=config["lambda_ce"])
    elif config["loss_function"] == "CrossEntropyLoss":
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        loss_function = torch.nn.BCEWithLogitsLoss()

    model = model.to(device=device)

    # from torchinfo import summary
    # summary(model, input_size=(1, config["in_channels"], 128, 128))

    optimizer = Adam(model.parameters(), config["learning_rate"], weight_decay=config["weight_decay"])

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
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_frequency, gamma=learning_rate_decay_factor)

    # Metrics
    include_background = True if config["out_channels"] == 1 else False
    dice_metric = DiceMetric(include_background=include_background, reduction="mean")
    iou_metric = MeanIoU(include_background=include_background, reduction="mean")
    confusion_matrix_metric = ConfusionMatrixMetric(
        include_background=include_background, 
        metric_name=["accuracy", "precision", "sensitivity", "specificity", "f1_score"],
        reduction="mean"
    )

    if config["out_channels"] == 1:
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    else:
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=config["out_channels"])])

    # Train model
    for epoch in range(config["num_epochs"]):
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch in tqdm(train_dataloader):
            step += 1
            inputs = batch["image"].to(device=device)
            labels = batch["label"].to(device=device)
            if config["out_channels"] > 1:
                labels = monai.networks.one_hot(labels, num_classes=config["out_channels"])
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
                if config["out_channels"] > 1:
                    val_labels = monai.networks.one_hot(val_labels, num_classes=config["out_channels"])
                val_outputs = model(val_inputs)
                
                loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()
                
                # Compute metrics for current iteration
                if config["loss_function"] == "CrossEntropyLoss":
                    val_outputs = torch.softmax(val_outputs, dim=1)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = decollate_batch(val_labels)
                
                dice_metric(y_pred=val_outputs, y=val_labels)
                iou_metric(y_pred=val_outputs, y=val_labels)
                confusion_matrix_metric(y_pred=val_outputs, y=val_labels)

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
            outputs = model(inputs.to(device=device))

        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for i in range(3):
            axes[i, 0].imshow(inputs[i, 0, :, :], cmap="gray")
            axes[i, 1].imshow(labels[i].squeeze(), cmap="gray")
            if config["out_channels"] == 1:
                im = axes[i, 2].imshow(torch.sigmoid(outputs[i]).squeeze().detach().cpu(), cmap="gray")
            else:
                im = axes[i, 2].imshow(torch.softmax(outputs[i], dim=0).detach().cpu()[1, :, :], cmap="gray")  # Show only first segmentation class
            
            # Create an additional axis for the colorbar
            cax = fig.add_axes([axes[i, 2].get_position().x1 + 0.01,
                                axes[i, 2].get_position().y0,
                                0.02,
                                axes[i, 2].get_position().height])
            fig.colorbar(im, cax=cax)
        
        # Log current learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            logging.info(f"Current learning rate: {current_lr}")

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
            "lr": current_lr,
            "examples": wandb.Image(fig)})

        plt.close(fig)
        
        # Save model checkpoint (if not the last epoch)
        if (args.save_ckpt_freq > 0 
            and (epoch + 1) % args.save_ckpt_freq == 0
            and (epoch + 1) < config["num_epochs"]):
            ckpt_model_path = os.path.join(ckpt_dir, f"model_{epoch:03d}.pt")
            torch.save(model.state_dict(), ckpt_model_path)
            logging.info(f"Saved model checkpoint to {ckpt_model_path}")

    # Save the final model also under the name "model.pt" so that we can easily find it later.
    # This is useful if we want to use the model for inference without having to specify the model filename.
    model_path = os.path.join(run_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved model to {model_path}.")

    # Save model as TorchScript
    if args.save_torchscript:
        ts_model_path = os.path.join(run_dir, "model_traced.pt")
        model = model.to("cpu")
        model.eval()
        example_input = torch.rand(1, config["in_channels"], config["image_size"], config["image_size"])
        traced_script_module = torch.jit.trace(model, example_input)
        d = {"shape": example_input.shape}
        extra_files = {"config.json": json.dumps(d)}
        traced_script_module.save(ts_model_path, _extra_files=extra_files)
        logging.info(f"Saved traced model to {ts_model_path}.")

    run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
