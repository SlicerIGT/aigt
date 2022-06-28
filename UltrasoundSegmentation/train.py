import os
import json
import tqdm
import argparse
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from ruamel.yaml import YAML

import utils
import ultrasound_batch_generator as generator
from losses import WeightedCategoricalCrossEntropy, DiceLoss, BCEDiceLoss
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
        "--girder_url",
        type=str,
        default="https://pocus.cs.queensu.ca/api/v1",
        help="full path to the REST API of a Girder instance"
    )
    parser.add_argument(
        "--girder_key",
        type=str,
        help="API key for private Girder collections"
    )
    parser.add_argument(
        "--girder_csv",
        type=str,
        required=True,
        help="CSV file containing Girder IDs and subject IDs for all files"
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
    # Prevent OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    # Load training settings
    config = parse_config(FLAGS.config_yaml)

    # Create standard folders for save
    data_arrays_fullpath, notebooks_save_fullpath, results_save_fullpath, \
        models_save_fullpath, logs_save_fullpath, val_data_fullpath =\
            utils.create_standard_project_folders(FLAGS.save_folder)

    # Fetch Girder data
    ultrasound_arrays_by_patients, segmentation_arrays_by_patients = \
        utils.load_girder_data(FLAGS.girder_csv, data_arrays_fullpath, FLAGS.girder_url, FLAGS.girder_key)

    n_patients = len(ultrasound_arrays_by_patients)
    n_images = 0
    n_segmentations = 0
    for i in range(n_patients):
        n_patient_images = ultrasound_arrays_by_patients[i].shape[0]
        n_patient_segmentations = segmentation_arrays_by_patients[i].shape[0]
        n_images += n_patient_images
        n_segmentations += n_patient_segmentations
        print(f"Patient {i} has {n_patient_images} ultrasounds and {n_patient_segmentations} segmentations")

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
    print("Planning {} rounds of validation".format(num_validation_rounds))
    for i in range(num_validation_rounds):
        print("Validation on patients {} in round {}".format(validation_schedule_patient[i], i))

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
    save_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("\nTimestamp for saved files:        {}".format(save_timestamp))
    print("Saving models in:                 {}".format(models_save_fullpath))
    print("Model name:                       {}".format(config["model_name"]))
    print("Saving validation predictions in: {}".format(val_data_fullpath))

    print("\nModel settings:")
    print("Number of patients:                     {}".format(n_patients))
    print("Total number of training images:        {}".format(n_images))
    print("Total number of training segmentations: {}".format(n_images))
    print("Image size:                             {}".format(config["image_size"]))
    print("Epochs:                                 {}".format(config["train"]["epochs"]))
    print("Batch size:                             {}".format(config["train"]["batch_size"]))
    print("Patience:                               {}".format(config["train"]["patience"]))
    print("Optimizer:                              {}".format(config["train"]["optimizer"]["name"]))
    print("Learning rate:                          {}".format(config["train"]["optimizer"]["lr"]))
    print("Loss function:                          {}".format(config["train"]["loss"]["name"]))

    print("\nData augmentation transformations:")
    for config_transform in config["train"]["preprocess"]:
        try:
            print(f"\t{config_transform['name']} - arguments: {dict(config_transform['args'])}")
        except KeyError:
            print(f"\t{config_transform['name']}")

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
        print(f"\nLeave-{validation_schedule_patient.shape[1]}-out round #{val_round_index}")
        print(f"\tTraining on {n_train} images, validating on {n_val} images...")

        # Initialize model
        training_generator = generator.UltrasoundSegmentationBatchGenerator(
            train_ultrasound_data,
            train_segmentation_data,
            config["train"]["batch_size"],
            (config["image_size"][0], config["image_size"][1]),
            transforms=transforms,
            rng=rng
        )

        # Set decay schedule
        optimizer_step = tf.Variable(0, trainable=False)
        boundaries, values = get_schedule_intervals(n_train, config["train"]["batch_size"], config["train"]["epochs"])
        schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        lr_schedule = lambda: config["train"]["optimizer"]["lr"] * schedule(optimizer_step)

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
        if loss_fn_name.lower() == "wcce":
            try:
                loss_fn = WeightedCategoricalCrossEntropy(config["train"]["loss"]["class_weights"])
            except KeyError:
                class_weights = get_default_class_weights(train_ultrasound_data)
                loss_fn = WeightedCategoricalCrossEntropy(class_weights)
                print(f"No class weights provided, using class weights {class_weights}")
        elif loss_fn_name.lower() == "dice":
            loss_fn = DiceLoss()
        elif loss_fn_name.lower() == "bce_dice":
            try:
                loss_fn = BCEDiceLoss(config["train"]["loss"]["class_weights"])
            except KeyError:
                class_weights = get_default_class_weights(train_ultrasound_data)
                loss_fn = BCEDiceLoss(class_weights)
                print(f"No class weights provided, using class weights {class_weights}")
        else:
            raise ValueError(f"unsupported loss function: {loss_fn_name}")

        # Initialize metric and model
        metric = tf.keras.metrics.BinaryAccuracy()
        model = UNet()

        # Train step using eager execution
        @tf.function
        def train_step(img, label):
            with tf.GradientTape() as tape:
                out = model(img, training=True)
                loss = loss_fn(label, out)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return out, loss

        # Dictionary to track loss and accuracy over training
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        # Variables for early stopping
        best_epoch_val_loss = np.inf
        last_improvement = 0

        # Training loop
        training_time_start = datetime.datetime.now()
        epoch_size = len(training_generator)
        for epoch in tqdm.tqdm(range(config["train"]["epochs"])):
            print(f"Starting epoch {epoch}, lr = {config['train']['optimizer']['lr'] * schedule(optimizer_step)}")
            epoch_loss = 0
            epoch_accuracy = 0
            for batch_index in range(epoch_size):
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
                for val_batch_index in range(epoch_size):
                    start_idx = val_batch_index * config["train"]["batch_size"]
                    end_idx = (val_batch_index + 1) * config["train"]["batch_size"]
                    val_ultrasound_batch = val_ultrasound_data[start_idx:end_idx]
                    val_segmentation_batch = val_segmentation_data[start_idx:end_idx]
                    val_segmentation_onehot = tf.keras.utils.to_categorical(val_segmentation_batch, 2)
                    val_predictions = model(val_ultrasound_batch)
                    epoch_val_loss += loss_fn(val_segmentation_onehot, val_predictions) / epoch_size
                    metric.update_state(val_segmentation_onehot, val_predictions)
                    epoch_val_accuracy += metric.result().numpy() / epoch_size

            # Log epoch history
            history["loss"].append(epoch_loss.numpy().tolist())  # Convert EagerTensor to float
            history["accuracy"].append(epoch_accuracy)
            history["val_loss"].append(epoch_val_loss.numpy().tolist())
            history["val_accuracy"].append(epoch_val_accuracy)

            # Print epoch results
            print(f"Epoch {epoch + 1}/{config['train']['epochs']}"
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
                print(f"Model checkpoint saved to {model_fullname}.")

                # Save training history
                history_filename = config["model_name"] + "_history-" + str(val_round_index) + "_" + save_timestamp + ".json"
                history_fullname = os.path.join(logs_save_fullpath, history_filename)
                with open(history_fullname, "w") as f:
                    json.dump(history, f)
                    print(f"Training history saved to {history_fullname}.")
            else:
                last_improvement += 1
            if last_improvement > config["train"]["patience"]:
                print(f"Validation loss has not decreased for {config['train']['patience']} epochs. "
                      f"Stopping training at {epoch = }.")
                break

            training_generator.on_epoch_end()

        training_time_stop = datetime.datetime.now()
        print(f"\tValidation round #{val_round_index} training time: {training_time_stop - training_time_start}")

    cross_val_time_stop = datetime.datetime.now()
    print(f"Total training time: {cross_val_time_stop - cross_val_time_start}")


if __name__ == '__main__':
    FLAGS = get_parser()
    main(FLAGS)
