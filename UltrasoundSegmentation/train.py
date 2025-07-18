"""
Train a u-net model on the ultrasound dataset.
For experiment tracking:
    - Save a copy of the configuration file and the trained model in the output folder
    - Log training metrics to a file or console
    - Log training metrics to Weights & Biases
"""

import importlib
import argparse
import logging
import monai
import random
import torch
import os
import sys
import json
import yaml
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
matplotlib.use("Agg")  # Use non-interactive backend to avoid error when running on server without GUI

from tqdm import tqdm
from time import perf_counter
from datetime import datetime
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

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
    HausdorffDistanceMetric, 
    ConfusionMatrixMetric
)

import datasets
from tracking_module import TrackingModule
from lr_scheduler import PolyLRScheduler, LinearWarmupWrapper


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--config-file", type=str, default="configs/train_config.yaml")
    parser.add_argument("--num-sample-images", type=int, default=3)
    parser.add_argument("--num-fps-test-images", type=int, default=100)
    parser.add_argument("--save-torchscript", action="store_true")
    parser.add_argument("--save-ckpt-freq", type=int, default=0)
    parser.add_argument("--wandb-entity-name", type=str)
    parser.add_argument("--wandb-project-name", type=str, default="aigt_ultrasound_segmentation")
    parser.add_argument("--wandb-exp-name", type=str)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--save-log", action="store_true")
    parser.add_argument("--resume-ckpt", type=str)

    try:
        parser, extra_args = parser.parse_known_args()
        deprecated = ["--train-data-folder", "--val-data-folder"]
        if any(arg in deprecated for arg in extra_args):
            print("Error: --train-data-folder and --val-data-folder are deprecated. " 
                  "Please update your config file to use 'train_folder' and 'val_folder' instead.")
            return None
        return parser
    except SystemExit:
        return None


def main(args):
    # Load config file
    # If config file is not given, use default config
    # If path is not specified, look for config file in the configs folder
    if args.config_file is None:
        args.config_file = os.path.join(os.path.dirname(__file__), "configs/train_config.yaml")
    else:
        if not os.path.isabs(args.config_file):
            args.config_file = os.path.join(os.path.dirname(__file__), "configs", args.config_file)
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize Weights & Biases
    wandb.login()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.wandb_exp_name is not None:
        experiment_name = f"{args.wandb_exp_name}_{timestamp}"
    else:
        experiment_name = f"{config['network']}_{config['loss_function']}_{timestamp}"
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

    # Create datasets
    dataset_cfg = config["dataset"]
    dataset_params = dataset_cfg.get("params", {})
    train_dataset = getattr(datasets, dataset_cfg["name"])(
        dataset_cfg["train_folder"],
        transform=train_transform,
        **(dataset_params if dataset_params else {})
    )
    val_dataset = getattr(datasets, dataset_cfg["name"])(
        dataset_cfg["val_folder"],
        transform=val_transform,
        **(dataset_params if dataset_params else {})
    )

    # Create dataloaders using UltrasoundDataset
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=config["shuffle"], 
        num_workers=4,
        generator=g
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=0,
        generator=g
    )

    # Construct loss function
    use_sigmoid = True if config["out_channels"] == 1 else False
    use_softmax = True if config["out_channels"] > 1 else False
    ce_weight = torch.tensor(config["class_weights"], device=device) \
                    if config["out_channels"] > 1 else None
    if config["loss_function"].lower() == "dicefocal":
        loss_function = monai.losses.DiceFocalLoss(
            sigmoid=use_sigmoid,
            softmax=use_softmax,
            lambda_dice=(1.0 - config["lambda_ce"]),
            lambda_focal=config["lambda_ce"]
        )
    elif config["loss_function"].lower() == "tversky":
        loss_function = monai.losses.TverskyLoss(
            sigmoid=use_sigmoid,
            softmax=use_softmax,
            alpha=0.3,
            beta=0.7  # best values from original paper
        )
    else:  # default to dice + cross entropy
        loss_function = monai.losses.DiceCELoss(
            sigmoid=use_sigmoid,
            softmax=use_softmax,
            lambda_dice=(1.0 - config["lambda_ce"]), 
            lambda_ce=config["lambda_ce"],
            weight=ce_weight
        )

    # Construct model
    use_tracking_layer = True
    dropout_rate = config["dropout_rate"] if "dropout_rate" in config else 0.0
    if config["network"].lower() == "attentionunet":
        model = monai.networks.nets.AttentionUnet(
            spatial_dims=2,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=dropout_rate
        )
    elif config["network"].lower() == "effnetunet":
        model = monai.networks.nets.FlexibleUNet(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            backbone="efficientnet-b0",
            pretrained=True
        )
    elif config["network"].lower() == "unetplusplus":
        model = monai.networks.nets.BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"]
        )
    elif config["network"].lower() == "unetr":
        model = monai.networks.nets.UNETR(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            img_size=config["image_size"],
            dropout_rate=dropout_rate,
            spatial_dims=2
        )
    elif config["network"].lower() == "swinunetr":
        model = monai.networks.nets.SwinUNETR(
            img_size=config["image_size"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            spatial_dims=2
        )
    elif config["network"].lower() == "segresnet":
        model = monai.networks.nets.SegResNet(
            spatial_dims=2,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"]
        )
    elif config["network"].lower() == "custom":
        try:
            model_cfg = config["model"]
            model_path = model_cfg["model_path"]
        except KeyError:
            logging.error("Custom model not found in config file.")
            return

        # load custom module
        module_name = os.path.basename(model_path).split(".")[0]
        loader = importlib.machinery.SourceFileLoader(module_name, model_path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)

        # instantiate model
        model_params = model_cfg.get("params", {})
        model = getattr(module, model_cfg["name"])(
            **(model_params if model_params else {})
        )
        use_tracking_layer = model_cfg["use_tracking_layer"]
    else:  # default to unet
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=config["num_res_units"] if "num_res_units" in config else 2,
            dropout=dropout_rate
        )

    # tracking layer is used by default unless otherwise specified in custom model
    if use_tracking_layer:
        # check if local (sliding window) or global (single frame)
        if train_dataset.__class__.__name__ == "LocalTrackedUSDataset":
            window_size = train_dataset.window_size
        else:
            window_size = 1
        tracking_layer = TrackingModule(config["image_size"], window_size)
        model = nn.Sequential(tracking_layer, model)
    
    model = model.to(device=device)
    # optimizer = Adam(model.parameters(), config["learning_rate"], weight_decay=config["weight_decay"])
    optimizer = AdamW(model.parameters(), config["learning_rate"], weight_decay=config["weight_decay"])

    # resume training
    if args.resume_ckpt:
        try:
            state = torch.load(args.resume_ckpt)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            best_val_loss = state["val_loss"]
            logging.info(f"Loaded model from {args.resume_ckpt}.")
        except Exception as e:
            logging.error(f"Failed to load model from {args.resume_ckpt}: {e}.")
    else:
        start_epoch = 0
        best_val_loss = np.inf

    # Set up learning rate scheduler
    try:
        learning_rate_decay_frequency = int(config["learning_rate_decay_frequency"])
    except Exception:
        learning_rate_decay_frequency = 100
    try:
        learning_rate_decay_factor = float(config["learning_rate_decay_factor"])
    except Exception:
        learning_rate_decay_factor = 1.0 # No decay
    # logging.info(f"Learning rate decay frequency: {learning_rate_decay_frequency}")
    # logging.info(f"Learning rate decay factor: {learning_rate_decay_factor}")
    # scheduler = StepLR(optimizer, step_size=learning_rate_decay_frequency, gamma=learning_rate_decay_factor)
    
    # next two schedulers use minibatch as step, not epoch
    # need to move scheduler.step() to inner loop
    start_step = start_epoch * len(train_dataloader)
    max_steps = len(train_dataloader) * config["num_epochs"]
    # scheduler = PolyLRScheduler(optimizer, config["learning_rate"], max_steps, last_step=start_step - 1)

    # cosine annealing with warmup
    warmup_steps = config["warmup_steps"] if "warmup_steps" in config else 250
    last_cosine_step = start_step - warmup_steps if start_step > warmup_steps else 0
    lr_scheduler = CosineAnnealingLR(
        optimizer, 
        max_steps - warmup_steps, 
        last_epoch=last_cosine_step - 1
    )
    scheduler = LinearWarmupWrapper(
        optimizer, 
        lr_scheduler, 
        config["learning_rate"], 
        warmup_steps=warmup_steps, 
        last_step=start_step - 1
    )

    # Metrics
    include_background = True if config["out_channels"] == 1 else False
    dice_metric = DiceMetric(include_background=include_background, reduction="mean")
    iou_metric = MeanIoU(include_background=include_background, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=include_background, percentile=95.0, reduction="mean")
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
    for epoch in range(start_epoch, epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch in tqdm(train_dataloader):
            step += 1
            inputs = batch["image"].to(device=device)
            labels = batch["label"].to(device=device)
            tfms = batch["transform"].to(device=device)
            if config["out_channels"] > 1:
                labels = monai.networks.one_hot(labels, num_classes=config["out_channels"])
            if use_tracking_layer:
                outputs = model((inputs, tfms))
            else:
                outputs = model(inputs, tfms)  # use as separate input
            if isinstance(outputs, list):  # for unet++ output
                outputs = outputs[0]
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= step
        logging.info(f"Training loss: {epoch_loss}")

        # Validation step
        model.eval()
        val_loss = 0
        val_step = 0
        with torch.no_grad():
            for val_batch in tqdm(val_dataloader):
                val_step += 1
                val_inputs = val_batch["image"].to(device=device)
                val_labels = val_batch["label"].to(device=device)
                val_tfms = val_batch["transform"].to(device=device)
                if config["out_channels"] > 1:
                    val_labels = monai.networks.one_hot(val_labels, num_classes=config["out_channels"])
                if use_tracking_layer:
                    val_outputs = model((val_inputs, val_tfms))
                else:
                    val_outputs = model(val_inputs, val_tfms)  # use as separate input
                if isinstance(val_outputs, list):
                    val_outputs = val_outputs[0]
                loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()
                
                # Post processing
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = decollate_batch(val_labels)
                
                dice_metric(y_pred=val_outputs, y=val_labels)
                iou_metric(y_pred=val_outputs, y=val_labels)
                hd95_metric(y_pred=val_outputs, y=val_labels)
                confusion_matrix_metric(y_pred=val_outputs, y=val_labels)

            val_loss /= val_step
            dice = dice_metric.aggregate().item()
            iou = iou_metric.aggregate().item()
            hd95 = hd95_metric.aggregate().item()
            cm = confusion_matrix_metric.aggregate()

            # reset status for next validation round
            dice_metric.reset()
            iou_metric.reset()
            hd95_metric.reset()
            confusion_matrix_metric.reset()

            logging.info(
                f"Validation results:\n"
                f"\tLoss: {val_loss}\n"
                f"\tDice: {dice}\n"
                f"\tIoU: {iou}\n"
                f"\t95% HD: {hd95}\n"
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
        tfms = torch.stack([torch.from_numpy(val_dataset[i]["transform"]) for i in sample])
        with torch.no_grad():
            if use_tracking_layer:
                outputs = model((inputs.to(device=device), tfms.to(device=device)))
            else:
                outputs = model(inputs.to(device=device), tfms.to(device=device))
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
                im = axes[i, 2].imshow(torch.softmax(outputs[i], dim=0).detach().cpu()[1, :, :], cmap="gray", vmin=0, vmax=1)  # Show only first segmentation class.
            
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
                "95hd": hd95,
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

        # Keep latest model as just model.pt, which can be used for inference or to resume training
        model_path = os.path.join(run_dir, "model.pt")
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss
        }
        torch.save(state, model_path)

        # Save latest model as TorchScript
        if args.save_torchscript:
            model.eval()  # disable dropout and batchnorm
            ts_model_path = os.path.join(run_dir, "model_traced.pt")
            model = model.to("cpu")
            example_input = (
                torch.rand(1, config["in_channels"], config["image_size"], config["image_size"]),
                torch.rand(1, config["in_channels"], 4, 4)
            )
            d = {"shape": example_input[0].shape, "use_tracking_layer": True}
            traced_script_module = torch.jit.trace(model, (example_input, ))
            extra_files = {"config.json": json.dumps(d)}
            traced_script_module.save(ts_model_path, _extra_files=extra_files)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(run_dir, "model_best.pt")
            torch.save(state, best_model_path)

            if args.save_torchscript:
                best_ts_model_path = os.path.join(run_dir, "model_traced_best.pt")
                traced_script_module.save(best_ts_model_path, _extra_files=extra_files)
        
        model = model.to(device=device)  # return model to original device

    # Test inference time (load images before loop to exclude from time measurement)
    logging.info("Measuring inference time...")
    num_test_images = args.num_fps_test_images
    inputs = torch.stack([val_dataset[i]["image"] for i in range(num_test_images)])
    tfms = torch.stack([torch.from_numpy(val_dataset[i]["transform"]) for i in range(num_test_images)])
    model.to(device)
    model.eval()
    with torch.no_grad():
        start = perf_counter()
        for i in range(num_test_images):
            if use_tracking_layer:
                model((
                    inputs[i, :, :, :].unsqueeze(0).to(device=device),
                    tfms[i, :, :, :].unsqueeze(0).to(device=device)
                ))
            else:
                model(
                    inputs[i, :, :, :].unsqueeze(0).to(device=device),
                    tfms[i, :, :, :].unsqueeze(0).to(device=device)
                )
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
    if args is not None:
        main(args)
