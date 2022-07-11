import os
import sys
import json
import tqdm
import argparse
import datetime
import logging
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from ruamel.yaml import YAML
from girder_apikey_read import girder_apikey_read

import utils
import ultrasound_batch_generator as generator
from losses import BCELoss, WeightedCategoricalCrossEntropy, DiceLoss, BCEDiceLoss
from Models.unet import UNet

# Set random seed for reproducibility
rng = np.random.default_rng(2022)
tf.random.set_seed(2022)
os.environ["TF_DETERMINISTIC_OPS"] = "1"  # GPU seed


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_folder",
        type=str,
        required=True,
        help="path to folder for all project files"
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="path to .yaml config file containing training settings"
    )
    parser.add_argument(
        "--girder_csv",
        type=str,
        required=True,
        help="CSV file containing Girder IDs and subject IDs for all files"
    )
    parser.add_argument(
        "--girder_url",
        type=str,
        default="https://pocus.cs.queensu.ca/api/v1",
        help="full path to the REST API of a Girder instance"
    )
    return parser.parse_args()


def parse_config(config_filepath):
    try:
        with open(config_filepath, "r") as f:
            data = YAML().load(f)
            return data
    except Exception as e:
        print(f"error parsing .yaml file: {e}")


def get_schedule_intervals(n_images, batch_size, epochs, interval=10, decay_rate=0.5):
    steps_per_epoch = n_images / batch_size
    num_intervals = epochs // interval  # Decay lr/weight every interval epochs
    boundaries = []
    values = []
    for i in range(num_intervals):
        if i < num_intervals - 1:  # len(boundaries) is one less than len(values)
            boundaries.append(np.floor(steps_per_epoch * interval * (i + 1)))
        values.append(1.0 * (decay_rate ** i))
    return boundaries, values


def get_default_class_weights(arr):
    size = arr.size
    num_non_zeros = np.count_nonzero(arr)
    num_zeros = size - num_non_zeros
    return [num_zeros / size, num_non_zeros / size]


def main(FLAGS):
    # Load training settings
    config = parse_config(FLAGS.config_yaml)

    # Create standard folders for save
    data_arrays_fullpath, notebooks_save_fullpath, results_save_fullpath, \
        models_save_fullpath, logs_save_fullpath, val_data_fullpath = \
            utils.create_standard_project_folders(FLAGS.save_folder)

    # Setup logging
    save_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_log_filename = config["model_name"] + "_log_" + save_timestamp + ".log"
    output_log_fullpath = os.path.join(logs_save_fullpath, output_log_filename)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(output_log_fullpath)
    c_format = logging.Formatter("%(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.info(f"Writing run output to {output_log_fullpath}.")

    # Prevent OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.exception(e)

    # Fetch Girder data
    ultrasound_arrays_by_patients, segmentation_arrays_by_patients = \
        utils.load_girder_data(FLAGS.girder_csv, data_arrays_fullpath, FLAGS.girder_url, girder_apikey_read)

    n_patients = len(ultrasound_arrays_by_patients)
    n_images = 0
    n_segmentations = 0
    for i in range(n_patients):
        n_patient_images = ultrasound_arrays_by_patients[i].shape[0]
        n_patient_segmentations = segmentation_arrays_by_patients[i].shape[0]
        n_images += n_patient_images
        n_segmentations += n_patient_segmentations
        logger.info(f"Patient {i} has {n_patient_images} ultrasounds and {n_patient_segmentations} segmentations")

    # Cross validation scheduling
    """
    Provide NxM numpy array to schedule cross validation
    N rounds of validation will be performed, leaving out M patients in each for validation data
    All values should be valid patient IDs, or negative. Negative values are ignored.

    Example 1: a leave-one-out cross validation with 3 patients would look like this:
    validation_schedule_patient = np.array([[0],[1],[2]])

    Example 2: a leave-two-out cross validation on 10 patients would look like this:
    validation_schedule_patient = np.array([[0,1],[2,3],[4,5],[6,7],[8,9]])

    Example 3: leave-one-out cross validation with 3 patients, then training on all available data (no validation):
    validation_schedule_patient = np.array([[0],[1],[2],[-1]])
    """
    validation_schedule_patient = np.array(config["train"]["validation_schedule"])
    if np.max(np.max(validation_schedule_patient)) > (n_patients - 1):
        raise Exception("Patient ID cannot be greater than {}".format(n_patients - 1))

    num_validation_rounds = len(validation_schedule_patient)
    logger.info("Planning {} rounds of validation".format(num_validation_rounds))
    for i in range(num_validation_rounds):
        logger.info("Validation on patients {} in round {}".format(validation_schedule_patient[i], i))

    # Initialize data transformations
    transforms = []
    if config["train"]["preprocess"]:
        for transform in config["train"]["preprocess"]:
            try:
                tfm_class = getattr(generator, transform["name"])(*[], **transform["args"])
            except KeyError:
                tfm_class = getattr(generator, transform["name"])()
            transforms.append(tfm_class)

    # Print all training settings
    logger.info("\nTimestamp for saved files:      {}".format(save_timestamp))
    logger.info("Saving models in:                 {}".format(models_save_fullpath))
    logger.info("Model name:                       {}".format(config["model_name"]))
    logger.info("Saving validation predictions in: {}".format(val_data_fullpath))

    logger.info("\nModel settings:")
    logger.info("Number of patients:                     {}".format(n_patients))
    logger.info("Total number of training images:        {}".format(n_images))
    logger.info("Total number of training segmentations: {}".format(n_images))
    logger.info("Image size:                             {}".format(config["image_size"]))
    logger.info("Epochs:                                 {}".format(config["train"]["epochs"]))
    logger.info("Batch size:                             {}".format(config["train"]["batch_size"]))
    logger.info("Patience:                               {}".format(config["train"]["patience"]))
    logger.info("Optimizer:                              {}".format(config["train"]["optimizer"]["name"]))
    logger.info("Learning rate:                          {}".format(config["train"]["optimizer"]["lr"]))
    logger.info("Loss function:                          {}".format(config["train"]["loss"]["name"]))

    logger.info("\nData augmentation transformations:")
    for config_transform in config["train"]["preprocess"]:
        try:
            logger.info(f"\t{config_transform['name']} - arguments: {dict(config_transform['args'])}")
        except KeyError:
            logger.info(f"\t{config_transform['name']}")

    # Cross validation
    cross_val_time_start = datetime.datetime.now()
    for val_round_index in range(num_validation_rounds):
        # Prepare data arrays
        train_ultrasound_data = np.zeros(
            [0,
             ultrasound_arrays_by_patients[0].shape[1],
             ultrasound_arrays_by_patients[0].shape[2],
             ultrasound_arrays_by_patients[0].shape[3]]
        )
        train_segmentation_data = np.zeros(
            [0,
             segmentation_arrays_by_patients[0].shape[1],
             segmentation_arrays_by_patients[0].shape[2],
             segmentation_arrays_by_patients[0].shape[3]]
        )
        val_ultrasound_data = np.zeros(
            [0,
             ultrasound_arrays_by_patients[0].shape[1],
             ultrasound_arrays_by_patients[0].shape[2],
             ultrasound_arrays_by_patients[0].shape[3]]
        )
        val_segmentation_data = np.zeros(
            [0,
             segmentation_arrays_by_patients[0].shape[1],
             segmentation_arrays_by_patients[0].shape[2],
             segmentation_arrays_by_patients[0].shape[3]]
        )
        for patient_index in range(n_patients):
            if patient_index not in validation_schedule_patient[val_round_index]:
                train_ultrasound_data = np.concatenate(
                    (train_ultrasound_data, ultrasound_arrays_by_patients[patient_index])
                )
                train_segmentation_data = np.concatenate(
                    (train_segmentation_data, segmentation_arrays_by_patients[patient_index])
                )
            else:
                val_ultrasound_data = np.concatenate(
                    (val_ultrasound_data, ultrasound_arrays_by_patients[patient_index])
                )
                val_segmentation_data = np.concatenate(
                    (val_segmentation_data, segmentation_arrays_by_patients[patient_index])
                )
        n_train = train_ultrasound_data.shape[0]
        n_val = val_ultrasound_data.shape[0]
        logger.info(f"\nLeave-{validation_schedule_patient.shape[1]}-out round #{val_round_index}")
        logger.info(f"\tTraining on {n_train} images, validating on {n_val} images...")

        # Initialize dataloader and validation metric
        training_generator = generator.UltrasoundSegmentationBatchGenerator(
            train_ultrasound_data,
            train_segmentation_data,
            config["train"]["batch_size"],
            (config["image_size"][0], config["image_size"][1]),
            transforms=transforms,
            rng=rng
        )
        metric = tf.keras.metrics.BinaryAccuracy()

        # Set decay schedule
        optimizer_step = tf.Variable(0, trainable=False)
        boundaries, values = get_schedule_intervals(n_train, config["train"]["batch_size"], config["train"]["epochs"])
        schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        initial_lr = config["train"]["optimizer"]["lr"]
        lr_schedule = lambda: initial_lr * schedule(optimizer_step)

        # Initialize optimizer
        optimizer_name = config["train"]["optimizer"]["name"]
        if optimizer_name.lower() == "adam":
            wd_schedule = lambda: 1e-4 * schedule(optimizer_step)
            optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule)
        elif optimizer_name.lower() == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
        else:
            raise ValueError(f"unsupported optimizer: {optimizer_name}")

        # Initialize loss function
        loss_fn_name = config["train"]["loss"]["name"]
        if loss_fn_name.lower() == "bce":
            loss_fn = BCELoss()
        elif loss_fn_name.lower() == "wbce":
            try:
                loss_fn = WeightedCategoricalCrossEntropy(config["train"]["loss"]["class_weights"])
            except KeyError:
                class_weights = get_default_class_weights(train_ultrasound_data)
                loss_fn = WeightedCategoricalCrossEntropy(class_weights)
                logger.info(f"No class weights provided, using class weights {class_weights}")
        elif loss_fn_name.lower() == "dice":
            loss_fn = DiceLoss()
        elif loss_fn_name.lower() == "bce_dice":
            try:
                loss_fn = BCEDiceLoss(config["train"]["loss"]["class_weights"])
            except KeyError:
                class_weights = get_default_class_weights(train_ultrasound_data)
                loss_fn = BCEDiceLoss(class_weights)
                logger.info(f"No class weights provided, using class weights {class_weights}")
        else:
            raise ValueError(f"unsupported loss function: {loss_fn_name}")

        # Initialize model
        model_name = config["model_name"]
        if model_name.lower() == "unet":
            model = UNet()
        else:
            raise ValueError(f"unsupported model: {model_name}")

        # Train step using eager execution
        @tf.function
        def train_step(img, label):
            with tf.GradientTape() as tape:
                out = model(img, training=True)
                loss = loss_fn(label, out)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return out, loss

        # TensorFlow profiler to profile model training performance
        tb_log_folder = os.path.join(logs_save_fullpath, model_name + "_logs-" + str(val_round_index) + "_" + save_timestamp)
        logger.info(f"Writing training performance data to {tb_log_folder}.")
        tf.profiler.experimental.start(tb_log_folder)

        # Dictionary to track loss and accuracy over training
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        # Variables for early stopping
        best_epoch_val_loss = np.inf
        last_improvement = 0

        # Training loop
        training_time_start = datetime.datetime.now()
        epoch_size = len(training_generator)
        for epoch in tqdm.tqdm(range(config["train"]["epochs"])):
            logger.info(f"Starting epoch {epoch}, lr = {initial_lr * schedule(optimizer_step)}")
            epoch_loss = 0
            epoch_accuracy = 0
            for batch_index in range(epoch_size):
                if epoch == 3:
                    with tf.profiler.experimental.Trace('train', step_num=optimizer_step, _r=1):
                        img, label = training_generator[batch_index]
                        out, loss = train_step(img, label)
                else:
                    img, label = training_generator[batch_index]
                    out, loss = train_step(img, label)

                # Calculate average training loss and accuracy
                epoch_loss += loss / epoch_size
                metric.update_state(label, out)
                epoch_accuracy += metric.result().numpy() / epoch_size

                # Increment step for learning rate and weight decay
                optimizer_step.assign_add(1)

            # Calculate validation loss and accuracy for the epoch
            epoch_val_loss = 0
            epoch_val_accuracy = 0
            with tf.device("cpu:0"):  # Prevent OOM
                n_val_batches = n_val // config["train"]["batch_size"]
                for val_batch_index in range(n_val_batches):
                    start_idx = val_batch_index * config["train"]["batch_size"]
                    end_idx = (val_batch_index + 1) * config["train"]["batch_size"]
                    val_ultrasound_batch = val_ultrasound_data[start_idx:end_idx]
                    val_segmentation_batch = tf.cast(val_segmentation_data[start_idx:end_idx], dtype=tf.float32)
                    val_predictions = model(val_ultrasound_batch)
                    epoch_val_loss += loss_fn(val_segmentation_batch, val_predictions) / n_val_batches
                    metric.update_state(val_segmentation_batch, val_predictions)
                    epoch_val_accuracy += metric.result().numpy() / n_val_batches

            # Log epoch history
            history["loss"].append(epoch_loss.numpy().tolist())  # Convert EagerTensor to float
            history["accuracy"].append(epoch_accuracy)
            history["val_loss"].append(epoch_val_loss.numpy().tolist())
            history["val_accuracy"].append(epoch_val_accuracy)

            # Print epoch results
            logger.info(f"Epoch {epoch + 1}/{config['train']['epochs']}"
                        f" - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}"
                        f" - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_accuracy:.4f}")

            # Early stopping if no improvement in validation loss
            if epoch_val_loss < best_epoch_val_loss:
                best_epoch_val_loss = epoch_loss
                last_improvement = 0

                # Save best performing model
                model_filename = config["model_name"] + "_model-" + str(val_round_index) + "_" + save_timestamp + ".tf"
                model_fullname = os.path.join(models_save_fullpath, model_filename)
                model.save(model_fullname)
                logger.info(f"Model checkpoint saved to {model_fullname}.")

                # Save training history
                history_filename = config["model_name"] + "_history-" + str(val_round_index) + "_" + save_timestamp + ".json"
                history_fullname = os.path.join(logs_save_fullpath, history_filename)
                with open(history_fullname, "w") as f:
                    json.dump(history, f)
                    logger.info(f"Training history saved to {history_fullname}.")
            else:
                last_improvement += 1
            if last_improvement > config["train"]["patience"]:
                logger.info(f"Validation loss has not decreased for {config['train']['patience']} epochs. "
                            f"Stopping training at {epoch = }.")
                break

            training_generator.on_epoch_end()

        training_time_stop = datetime.datetime.now()
        logger.info(f"\tValidation round #{val_round_index} training time: {training_time_stop - training_time_start}")
        tf.profiler.experimental.stop()
        tf.summary.flush()

    cross_val_time_stop = datetime.datetime.now()
    logger.info(f"Total training time: {cross_val_time_stop - cross_val_time_start}")


if __name__ == '__main__':
    FLAGS = get_parser()
    main(FLAGS)
