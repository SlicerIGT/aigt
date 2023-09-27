"""
Train a u-net model on the ultrasound dataset.
For experiment tracking:
    - Save a copy of the configuration file and the trained model in the output folder
    - Log training metrics to a file or console
    - Log training metrics to Weights & Biases
"""

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to avoid error when running on server without GUI

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
from time import perf_counter
from datetime import datetime
from torch.optim import SGD

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

from UltrasoundDataset import UltrasoundDataset, ZScoreNormalized


# Default nnUNet hyperparameters
INITIAL_LR = 1e-2
WEIGHT_DECAY = 3e-5
NUM_BATCHES = 250
NUM_VAL_BATCHES = 50
NUM_EPOCHS = 100


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-folder", type=str)
    parser.add_argument("--val-data-folder", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--nnunet-plan-json", type=str, required=True)
    parser.add_argument("--nnunet-configuration", type=str, default="2d")
    parser.add_argument("--num-sample-images", type=int, default=3)
    parser.add_argument("--num-fps-test-images", type=int, default=100)
    parser.add_argument("--save-torchscript", action="store_true")
    parser.add_argument("--save-ckpt-freq", type=int, default=0)
    parser.add_argument("--wandb-entity-name", type=str)
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
    # Load nnUNet plan file
    with open(args.nnunet_plan_json, "r") as f:
        plan = json.load(f)
    
    # Create config yaml similar to other train scripts
    config = {
        "model_name": f"nnunet_{args.nnunet_configuration}",
        "loss_function": "DiceCE",
        "lambda_ce": 0.5,
        "class_weights": [1.0, 1.0],
        "image_size": plan["configurations"][args.nnunet_configuration]["patch_size"][0],
        "in_channels": 1,
        "out_channels": 2,
        "num_epochs": NUM_EPOCHS,
        "batch_size": plan["configurations"][args.nnunet_configuration]["batch_size"],
        "learning_rate": INITIAL_LR,
        "weight_decay": WEIGHT_DECAY,
        "shuffle": True,
        "seed": 42
    }
    
    # Initialize Weights & Biases
    wandb.login()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.wandb_exp_name is not None:
        experiment_name = f"{args.wandb_exp_name}_{timestamp}"
    else:
        experiment_name = f"nnunet_{args.nnunet_configuration}_{timestamp}"
    run = wandb.init(
        # Set the project where this run will be logged
        project=args.wandb_project_name,
        entity=args.wandb_entity_name,
        name=experiment_name
        )
    run.define_metric("dice", summary="max")
    run.define_metric("sensitivity", summary="max")

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
    # TODO: actually read the normalization scheme from plan
    transform = Compose([
        transforms.Transposed(keys=["image", "label"], indices=[2, 0, 1]),
        transforms.ToTensord(keys=["image", "label"]),
        transforms.EnsureTyped(keys=["image", "label"], dtype="float32"),
        transforms.Resized(
            keys=["image", "label"], 
            spatial_size=(plan["configurations"][args.nnunet_configuration]["patch_size"])
        ),
        ZScoreNormalized(keys=["image"])
    ])
    train_dataset = UltrasoundDataset(args.train_data_folder, transform=transform)
    val_dataset = UltrasoundDataset(args.val_data_folder, transform=transform)

    # Create dataloaders using UltrasoundDataset
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=plan["configurations"][args.nnunet_configuration]["batch_size"], 
        shuffle=config["shuffle"], 
        num_workers=2,
        generator=g,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=plan["configurations"][args.nnunet_configuration]["batch_size"], 
        shuffle=False, 
        num_workers=0,
        generator=g
    )

    # Construct model from nnUNet plan
    model = monai.networks.nets.DynUNet(
        spatial_dims=2,
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        kernel_size=(
            plan["configurations"][args.nnunet_configuration]["conv_kernel_sizes"]
        ),
        strides=(
            plan["configurations"][args.nnunet_configuration]["pool_op_kernel_sizes"]
        ),
        upsample_kernel_size=(
            plan["configurations"][args.nnunet_configuration]["pool_op_kernel_sizes"][1:]
        ),
        deep_supervision=True
    )
    model = model.to(device=device)
    
    # Construct loss function
    use_sigmoid = True if config["out_channels"] == 1 else False
    use_softmax = True if config["out_channels"] > 1 else False
    loss_function = monai.losses.DiceCELoss(
        include_background=False,
        sigmoid=use_sigmoid,
        softmax=use_softmax,
        batch=plan["configurations"][args.nnunet_configuration]["batch_dice"]
    )

    optimizer = SGD(
        model.parameters(), 
        INITIAL_LR, 
        weight_decay=WEIGHT_DECAY, 
        momentum=0.99,
        nesterov=True
    )
    # Set up learning rate decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / NUM_EPOCHS) ** 0.9)

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
    epochs = config["num_epochs"]
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        train_iter = iter(train_dataloader)
        for batch_idx in tqdm(range(NUM_BATCHES)):
            try:
                batch = next(train_iter)
            except StopIteration:  # reset dataloder
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            step += 1
            inputs = batch["image"].to(device=device)
            labels = batch["label"].to(device=device)
            if config["out_channels"] > 1:
                labels = monai.networks.one_hot(labels, num_classes=config["out_channels"])
            optimizer.zero_grad()
            outputs = model(inputs)
            if len(outputs.size()) - len(labels.size()) == 1:
                # deep supervision mode, need to unbind feature maps first.
                outputs = torch.unbind(outputs, dim=1)[0]  # first feature map is prediction
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
            val_iter = iter(val_dataloader)
            for val_batch_idx in tqdm(range(NUM_VAL_BATCHES)):
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataloader)
                    val_batch = next(val_iter)
                val_step += 1
                val_inputs = val_batch["image"].to(device=device)
                val_labels = val_batch["label"].to(device=device)
                if config["out_channels"] > 1:
                    val_labels = monai.networks.one_hot(val_labels, num_classes=config["out_channels"])
                val_outputs = model(val_inputs)
                loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()
                
                # Post processing
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

            logging.info(
                f"Validation results:\n"
                f"\tLoss: {val_loss}\n"
                f"\tDice: {dice}\n"
                f"\tIoU: {iou}\n"
                f"\tAccuracy: {(acc := cm[0].item())}\n"
                f"\tPrecision: {(pre := cm[1].item())}\n"
                f"\tSensitivity: {(sen := cm[2].item())}\n"
                f"\tSpecificity: {(spe := cm[3].item())}\n"
                f"\tF1 score: {(f1 := cm[4].item())}"
            )
        
        # Log a random sample of test images along with their ground truth and predictions
        random.seed(config["seed"])
        sample = random.sample(range(len(val_dataset)), args.num_sample_images)

        inputs = torch.stack([val_dataset[i]["image"] for i in sample])
        labels = torch.stack([val_dataset[i]["label"] for i in sample])
        with torch.no_grad():
            outputs = model(inputs.to(device=device))
        if isinstance(outputs, list):
            outputs = outputs[0]
        if isinstance(labels, list):
            labels = labels[0]

        fig, axes = plt.subplots(args.num_sample_images, 3, figsize=(9, 3 * args.num_sample_images))
        for i in range(args.num_sample_images):
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
        try:
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
        except FileNotFoundError:
            logging.error("Failed to log examples to Weights & Biases. Temporary image file not found.")
        finally:
            plt.close(fig)
        
        # Save model checkpoint (if not the last epoch)
        if (args.save_ckpt_freq > 0 
            and (epoch + 1) % args.save_ckpt_freq == 0
            and (epoch + 1) < config["num_epochs"]):
            ckpt_model_path = os.path.join(ckpt_dir, f"model_{epoch:03d}.pt")
            torch.save(model.state_dict(), ckpt_model_path)
            logging.info(f"Saved model checkpoint to {ckpt_model_path}.")

    # Save the final model also under the name "model.pt" so that we can easily find it later.
    # This is useful if we want to use the model for inference without having to specify the model filename.
    model_path = os.path.join(run_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved model to {model_path}.")

    # Save model as TorchScript
    if args.save_torchscript:
        ts_model_path = os.path.join(run_dir, "model_traced.pt")
        model = model.to("cpu")
        example_input = torch.rand(1, config["in_channels"], config["image_size"], config["image_size"])
        traced_script_module = torch.jit.trace(model, example_input)
        d = {"shape": example_input.shape}
        extra_files = {"config.json": json.dumps(d)}
        traced_script_module.save(ts_model_path, _extra_files=extra_files)
        logging.info(f"Saved traced model to {ts_model_path}.")

    # Test inference time (load images before loop to exclude from time measurement)
    logging.info("Measuring inference time...")
    num_test_images = args.num_fps_test_images
    inputs = torch.stack([val_dataset[i]["image"] for i in range(num_test_images)])
    model.to(device)
    model.eval()
    with torch.no_grad():
        start = perf_counter()
        for i in range(num_test_images):
            model(inputs[i, :, :, :].unsqueeze(0).to(device=device))
        end = perf_counter()
    avg_inf_time = (end - start) / num_test_images
    avg_inf_fps = 1 / avg_inf_time
    logging.info(f"Average inference time per image: {avg_inf_time:.4f}s ({avg_inf_fps:.2f} FPS)")
    run.log({
        "avg_inference_time": avg_inf_time,
        "avg_inference_fps": avg_inf_fps
    })

    run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
