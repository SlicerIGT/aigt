import os
import sys
import datetime
from random import sample
from pathlib import Path
import girder_client
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

from ultrasound_batch_generator import train_preprocess, train_preprocess_with_transforms, train_preprocess_with_maps, train_preprocess_with_transforms_with_maps, generate_weight_maps, generate_transform
import evaluation_metrics

from models import (
    new_unet,
    weighted_categorical_crossentropy,
    weighted_categorical_crossentropy_with_maps,
)
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

def train(
    batch_size=128, 
    num_epochs=500, 
    sagittal_only=False, 
    num_frames=1, 
    with_maps=False, 
    learning_rate=0.002,
    lr_decay=False,
    dropout=0.0,
    use_attention=True,
    num_layers=5,
    filters=16,
    use_batch_norm=True,
    use_transforms=False,
    load_from_save=False,
):

    run_str = ''
    run_str += 'NewUNet_'
    run_str += 'WMaps_' if with_maps else ''
    run_str += 'SagittalOnly_' if sagittal_only else 'AllImages_'
    run_str += 'SingleFrame' if num_frames == 1 else str(num_frames) + 'Frames_'
    run_str += 'Dropout' + str(dropout) + '_' if dropout else ''
    run_str += 'WAttention_' if use_attention else ''
    run_str += 'WBatchNorm_' if use_batch_norm else ''
    run_str += str(filters) + '-InitialFilters_' 
    run_str += str(num_layers) + '-Layers'     
    run_str += '_WTransforms' if use_transforms else ''

    local_data_folder = "/home/zbaum/Baum/aigt/UltrasoundSegmentation/Spine"

    data_arrays_folder = "DataArrays"
    results_save_folder = "SavedResults"
    models_save_folder = "SavedModels"
    data_arrays_fullpath = os.path.join(local_data_folder, data_arrays_folder)
    results_save_fullpath = os.path.join(local_data_folder, results_save_folder)
    models_save_fullpath = os.path.join(local_data_folder, models_save_folder)
    if not os.path.exists(data_arrays_fullpath):
        os.makedirs(data_arrays_fullpath)
        print("Created folder: {}".format(data_arrays_fullpath))

    if not os.path.exists(results_save_fullpath):
        os.makedirs(results_save_fullpath)
        print("Created folder: {}".format(results_save_fullpath))

    if not os.path.exists(models_save_fullpath):
        os.makedirs(models_save_fullpath)
        print("Created folder: {}".format(models_save_fullpath))

    # Learning parameters

    ultrasound_size = 128
    num_classes = 2
    min_learning_rate = 0.00001
    class_weights = np.array([0.1, 0.9])
    learning_rate_decay = (learning_rate - min_learning_rate) / num_epochs if lr_decay else 0

    if with_maps:
        loss_func = weighted_categorical_crossentropy_with_maps(class_weights)
        preprocess_func = train_preprocess_with_maps
        if use_transforms:
            preprocess_func = train_preprocess_with_transforms_with_maps
    else:
        loss_func = weighted_categorical_crossentropy(class_weights)
        preprocess_func = train_preprocess
        if use_transforms:
            preprocess_func = train_preprocess_with_transforms

    # Evaluation parameters

    acceptable_margin_mm = 1.0
    mm_per_pixel = 1.0

    roc_thresholds = [
        0.9,
        0.8,
        0.7,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.08,
        0.06,
        0.04,
        0.02,
        0.01,
        0.008,
        0.006,
        0.004,
        0.002,
        0.001,
    ]

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
    #validation_schedule_patient = np.array([[-1]])
    validation_schedule_patient = np.array([[0]])

    # Define what data to download

    girder_url = "https://pocus.cs.queensu.ca/api/v1"
    girder_key = "nwv5qqqrDYn9DVakp1XnYDqjrNsowxaXisawPNRR"

    if not use_transforms:
        verdure_axial_csv = "VerdureAxial-Transforms.csv"
        verdure_sagittal_csv = "VerdureSagittal-Transforms.csv"

        verdure_axial_ultrasound_arrays, verdure_axial_segmentation_arrays = utils.load_girder_data(
            verdure_axial_csv, data_arrays_fullpath, girder_url, girder_key=girder_key
        )

        verdure_sagittal_ultrasound_arrays, verdure_sagittal_segmentation_arrays = utils.load_girder_data(
            verdure_sagittal_csv, data_arrays_fullpath, girder_url, girder_key=girder_key
        )


        if not sagittal_only:
            ultrasound_arrays_by_patients = (
                verdure_axial_ultrasound_arrays
                + verdure_sagittal_ultrasound_arrays
            )

            segmentation_arrays_by_patients = (
                verdure_axial_segmentation_arrays
                + verdure_sagittal_segmentation_arrays
            )

        else:
            ultrasound_arrays_by_patients = (
                verdure_sagittal_ultrasound_arrays
            )

            segmentation_arrays_by_patients = (
                verdure_sagittal_segmentation_arrays
            )

    else:
        verdure_axial_csv = "VerdureAxial-Transforms.csv"
        verdure_sagittal_csv = "VerdureSagittal-Transforms.csv"

        verdure_axial_ultrasound_arrays, verdure_axial_transform_arrays, verdure_axial_segmentation_arrays = utils.load_girder_data(
            verdure_axial_csv, data_arrays_fullpath, girder_url, girder_key=girder_key, get_transforms=True
        )

        verdure_sagittal_ultrasound_arrays, verdure_sagittal_transform_arrays, verdure_sagittal_segmentation_arrays = utils.load_girder_data(
            verdure_sagittal_csv, data_arrays_fullpath, girder_url, girder_key=girder_key, get_transforms=True
        )

        if not sagittal_only:
            ultrasound_arrays_by_patients = (
                verdure_axial_ultrasound_arrays + verdure_sagittal_ultrasound_arrays
            )

            transform_arrays_by_patients = (
                verdure_axial_transform_arrays + verdure_sagittal_transform_arrays
            )

            segmentation_arrays_by_patients = (
                verdure_axial_segmentation_arrays + verdure_sagittal_segmentation_arrays
            )

        else:
            ultrasound_arrays_by_patients = (
                verdure_sagittal_ultrasound_arrays
            )

            transform_arrays_by_patients = (
                verdure_sagittal_transform_arrays
            )

            segmentation_arrays_by_patients = (
                verdure_sagittal_segmentation_arrays
            )

        # Normalize the transform matrices between scale of the patient
        for i in range(len(transform_arrays_by_patients)):

            # find mean center per scan
            mean_val = np.mean(transform_arrays_by_patients[i], axis=0)
            mean_subtract = np.zeros(mean_val.shape)
            mean_subtract[0:3, 3] = mean_val[0:3, 3]

            # subtract mean
            for j in range(len(transform_arrays_by_patients[i])):
                transform_arrays_by_patients[i][j] -= mean_subtract

            # find max/min values
            min_z = (np.min(transform_arrays_by_patients[i], axis=0))[2, 3]
            max_z = (np.max(transform_arrays_by_patients[i], axis=0))[2, 3]
            diff = max_z - min_z

            # normalize
            for j in range(len(transform_arrays_by_patients[i])):
                transform_arrays_by_patients[i][j][0, 3] = (2 * (transform_arrays_by_patients[i][j][0, 3] - min_z) / diff) - 1
                transform_arrays_by_patients[i][j][1, 3] = (2 * (transform_arrays_by_patients[i][j][1, 3] - min_z) / diff) - 1
                transform_arrays_by_patients[i][j][2, 3] = (2 * (transform_arrays_by_patients[i][j][2, 3] - min_z) / diff) - 1

    if num_frames > 1:
        multiframe_ultrasound_arrays_by_patients = []

        for patient_series in ultrasound_arrays_by_patients:
            multiframe_patient_series = np.zeros(
                (
                    patient_series.shape[0],
                    patient_series.shape[1],
                    patient_series.shape[2],
                    patient_series.shape[3] * num_frames,
                )
            )

            for frame_idx in range(len(patient_series)):
                multiframe_frame = np.zeros(
                    (
                        patient_series[frame_idx].shape[0],
                        patient_series[frame_idx].shape[1],
                        patient_series[frame_idx].shape[2] * num_frames,
                    )
                )

                for frame in range(num_frames):
                    if frame_idx - frame >= 0:
                        multiframe_frame[:, :, frame] = np.squeeze(patient_series[frame_idx - frame])
                    else:
                        multiframe_frame[:, :, frame] = np.zeros(
                            (
                                patient_series[frame_idx].shape[0],
                                patient_series[frame_idx].shape[1],
                            )
                        )

                multiframe_patient_series[frame_idx] = multiframe_frame
            multiframe_ultrasound_arrays_by_patients.append(multiframe_patient_series)

        ultrasound_arrays_by_patients = multiframe_ultrasound_arrays_by_patients

    n_patients = len(ultrasound_arrays_by_patients)

    for i in range(n_patients):
        if use_transforms:
            print(
                "Patient {} has {} ultrasounds, {} segmentations and {} transforms".format(
                    i,
                    ultrasound_arrays_by_patients[i].shape[0],
                    segmentation_arrays_by_patients[i].shape[0],
                    transform_arrays_by_patients[i].shape[0],
                )
            )
        else:
            print(
                "Patient {} has {} ultrasounds and {} segmentations".format(
                    i,
                    ultrasound_arrays_by_patients[i].shape[0],
                    segmentation_arrays_by_patients[i].shape[0],
                )
            )
    
    # Prepare validation rounds

    if np.max(validation_schedule_patient) > (n_patients - 1):
        raise Exception("Patient ID cannot be greater than {}".format(n_patients - 1))

    num_validation_rounds = len(validation_schedule_patient)
    print("Planning {} rounds of validation".format(num_validation_rounds))
    for i in range(num_validation_rounds):
        print(
            "Validation on patients {} in round {}".format(
                validation_schedule_patient[i], i
            )
        )

    # Print training parameters, to archive them together with the notebook output.

    time_sequence_start = datetime.datetime.now()

    print("Name for saved files: {}".format(run_str))
    print("\nTraining parameters")
    print("Max epochs:          {}".format(num_epochs))
    print("LR:                  {}".format(learning_rate))
    print("LR decay:            {}".format(learning_rate_decay))
    print("Batch size:          {}".format(batch_size))
    print("Frames used:         {}".format(num_frames))
    print("Dropout:             {}".format(dropout))
    print("Attention:           {}".format(use_attention))
    print("BatchNorm:           {}".format(use_batch_norm))
    print("Filters (Start):     {}".format(filters))
    print("Layers:              {}".format(num_layers))
    print("Transforms:          {}".format(use_transforms))
    print("\nSaving models in:  {}".format(models_save_fullpath))

    # ROC data will be saved in these containers

    val_best_metrics = dict()
    val_fuzzy_metrics = dict()
    val_aurocs = np.zeros(num_validation_rounds)
    val_best_thresholds = np.zeros(num_validation_rounds)

    # Perform validation rounds

    for val_round_index in range(num_validation_rounds):

        # Prepare data arrays

        train_ultrasound_data = np.zeros(
            [
                0,
                ultrasound_arrays_by_patients[0].shape[1],
                ultrasound_arrays_by_patients[0].shape[2],
                ultrasound_arrays_by_patients[0].shape[3],
            ]
        )

        if use_transforms:
            train_transform_data = np.zeros(
                [
                    0,
                    transform_arrays_by_patients[0].shape[1],
                    transform_arrays_by_patients[0].shape[2],
                ]
            )

        train_segmentation_data = np.zeros(
            [
                0,
                segmentation_arrays_by_patients[0].shape[1],
                segmentation_arrays_by_patients[0].shape[2],
                segmentation_arrays_by_patients[0].shape[3],
            ]
        )

        val_ultrasound_data = np.zeros(
            [
                0,
                ultrasound_arrays_by_patients[0].shape[1],
                ultrasound_arrays_by_patients[0].shape[2],
                ultrasound_arrays_by_patients[0].shape[3],
            ]
        )

        if use_transforms:
            val_transform_data = np.zeros(
                [
                    0,
                    transform_arrays_by_patients[0].shape[1],
                    transform_arrays_by_patients[0].shape[2],
                ]
            )

        val_segmentation_data = np.zeros(
            [
                0,
                segmentation_arrays_by_patients[0].shape[1],
                segmentation_arrays_by_patients[0].shape[2],
                segmentation_arrays_by_patients[0].shape[3],
            ]
        )

        for patient_index in range(n_patients):
            if patient_index not in validation_schedule_patient[val_round_index]:
                train_ultrasound_data = np.concatenate(
                    (train_ultrasound_data, ultrasound_arrays_by_patients[patient_index])
                )
                if use_transforms:
                    train_transform_data = np.concatenate(
                        (train_transform_data, transform_arrays_by_patients[patient_index])
                    )
                train_segmentation_data = np.concatenate(
                    (train_segmentation_data, segmentation_arrays_by_patients[patient_index])
                )
            else:
                val_ultrasound_data = np.concatenate(
                    (val_ultrasound_data, ultrasound_arrays_by_patients[patient_index])
                )
                if use_transforms:
                    val_transform_data = np.concatenate(
                        (val_transform_data, transform_arrays_by_patients[patient_index])
                    )
                val_segmentation_data = np.concatenate(
                    (val_segmentation_data, segmentation_arrays_by_patients[patient_index])
                )

        n_train = train_ultrasound_data.shape[0]
        n_val = val_ultrasound_data.shape[0]

        print("\n*** Leave-one-out round # {}".format(val_round_index))
        print(
            "    Training on {} images, validating on {} images...".format(n_train, n_val)
        )

        train_segmentation_data_onehot = tf.keras.utils.to_categorical(
            train_segmentation_data, num_classes
        )
        val_segmentation_data_onehot = tf.keras.utils.to_categorical(
            val_segmentation_data, num_classes
        )

        # Create and train model
        model = new_unet(
            ultrasound_size,
            num_classes=num_classes,
            num_channels=num_frames,
            use_batch_norm=use_batch_norm,
            upsample_mode="deconv",  # 'deconv' or 'simple'
            dropout=dropout,
            dropout_type="spatial",
            use_attention=use_attention,
            filters=filters,
            num_layers=num_layers,
            use_transforms=use_transforms,
            output_activation="softmax",
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                lr=learning_rate, decay=learning_rate_decay
            ),
            loss=loss_func,
            metrics=["accuracy", evaluation_metrics.dice_coef],
        )

        d = (train_ultrasound_data, train_segmentation_data_onehot)
        if use_transforms:
            d = (train_ultrasound_data, train_transform_data, train_segmentation_data_onehot)
        dataset = tf.data.Dataset.from_tensor_slices(d)

        dataset = dataset.map(preprocess_func, num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
        dataset = dataset.prefetch(1)

        vd = (val_ultrasound_data, val_segmentation_data_onehot)
        if use_transforms:
            vd = (val_ultrasound_data, val_transform_data, val_segmentation_data_onehot)
        val_dataset = tf.data.Dataset.from_tensor_slices(vd)
        
        if with_maps:
            val_dataset = val_dataset.map(generate_weight_maps, num_parallel_calls=4)
        if use_transforms:
            val_dataset = val_dataset.map(generate_transform, num_parallel_calls=4)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(1)
        
        '''
        fig = plt.figure(figsize=(30, 20 * 5))
        i = 0
        for im, lb in dataset:

            a1 = fig.add_subplot(25, 6, i * 6 + 1)
            img1 = a1.imshow(np.squeeze(np.flipud(im[i, :, :, 1])), cmap='gray')

            a2 = fig.add_subplot(25, 6, i * 6 + 2)
            img2 = a2.imshow(np.squeeze(np.flipud(im[i, :, :, 0])), cmap='gray')

            a3 = fig.add_subplot(25, 6, i * 6 + 3)
            img3 = a3.imshow(np.squeeze(np.flipud(lb[i, :, :, 0])))
            c = fig.colorbar(img3, fraction=0.046, pad=0.04)

            a4 = fig.add_subplot(25, 6, i * 6 + 4)
            img4 = a4.imshow(np.squeeze(np.flipud(lb[i, :, :, 1])))
            c = fig.colorbar(img4, fraction=0.046, pad=0.04)

            a5 = fig.add_subplot(25, 6, i * 6 + 5)
            img5 = a5.imshow(np.squeeze(np.flipud(lb[i, :, :, -2])))
            c = fig.colorbar(img5, fraction=0.046, pad=0.04)

            a6 = fig.add_subplot(25, 6, i * 6 + 6)
            img6 = a6.imshow(np.squeeze(np.flipud(lb[i, :, :, -1])))
            c = fig.colorbar(img6, fraction=0.046, pad=0.04)

            i += 1
            if i >= 20:
                break

        plt.savefig("us_seg.png")
        '''

        model_file_name = run_str + "_model-" + str(val_round_index) + ".h5"
        model_best_file_name = run_str + "_model-" + str(val_round_index) + "_best.h5"
        model_fullname = os.path.join(models_save_fullpath, model_file_name)
        model_best_fullname = os.path.join(models_save_fullpath, model_best_file_name)

        if not load_from_save:
            training_time_start = datetime.datetime.now()

            if n_val > 0:
                training_log = model.fit(
                    dataset,
                    validation_data=val_dataset,
                    epochs=num_epochs,
                    verbose=2,
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(model_best_fullname, save_best_only=True)],
                    #callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100),
                    #           tf.keras.callbacks.ModelCheckpoint(model_best_fullname, save_best_only=True)],
                )

            else:
                training_log = model.fit(
                    dataset,
                    epochs=num_epochs,
                    verbose=2,
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(model_best_fullname, save_best_only=True)],
                )

            training_time_stop = datetime.datetime.now()

            # Print training log

            print("  Training time: {}".format(training_time_stop - training_time_start))

            # Plot training loss and metrics

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            axes[0].set_yscale('log')
            axes[0].plot(training_log.history["loss"], "bo--")
            if n_val > 0:
                axes[0].plot(training_log.history["val_loss"], "ro-")
            axes[0].set(xlabel="Epochs (n)", ylabel="Loss")
            if n_val > 0:
                axes[0].legend(["Training loss", "Validation loss"])

            axes[1].set_yscale('log')
            axes[1].plot(training_log.history["accuracy"], "bo--")
            if n_val > 0:
                axes[1].plot(training_log.history["val_accuracy"], "ro-")
            axes[1].set(xlabel="Epochs (n)", ylabel="Accuracy")
            if n_val > 0:
                axes[1].legend(["Training accuracy", "Validation accuracy"])

            fig.tight_layout()
            plt.savefig(run_str + "_val-round_" + str(val_round_index) + ".png")

        # Archive trained model with unique filename based on params
        if load_from_save:
            model.load_weights(model_fullname)
        else:
            model.save(model_fullname)

        # Predict on validation data
        if n_val > 0:
            if use_transforms:
                y_pred_val = model.predict([val_ultrasound_data, val_transform_data])
            else:
                y_pred_val = model.predict(val_ultrasound_data)

            # Validation results
            vali_metrics_dicts, vali_best_threshold_index, vali_area = evaluation_metrics.compute_roc(
                roc_thresholds, y_pred_val, val_segmentation_data, acceptable_margin_mm, mm_per_pixel)

            val_fuzzy_metrics[val_round_index] = evaluation_metrics.compute_evaluation_metrics(
                y_pred_val, val_segmentation_data, acceptable_margin_mm, mm_per_pixel)

            val_best_metrics[val_round_index]    = vali_metrics_dicts[vali_best_threshold_index]
            val_aurocs[val_round_index]          = vali_area
            val_best_thresholds[val_round_index] = roc_thresholds[vali_best_threshold_index]

        # Display sample results
        num_vali = val_ultrasound_data.shape[0]
        num_show = 20
        if num_vali < num_show:
            num_show = 0
        num_col = 4

        indices = [i for i in range(num_vali)]
        sample_indices = sample(indices, num_show)

        threshold = 0.5

        # Uncomment for comparing the same images
        sample_indices = list(range(0, 200, 10))

        fig = plt.figure(figsize=(18, num_show * 5))
        for i in range(num_show):
            a0 = fig.add_subplot(num_show, num_col, i * num_col + 1)
            img0 = a0.imshow(
                np.flipud(
                    val_ultrasound_data[sample_indices[i], :, :, 0].astype(np.float32)
                )
            )
            a0.set_title("Ultrasound #{}".format(sample_indices[i]))
            a1 = fig.add_subplot(num_show, num_col, i * num_col + 2)
            img1 = a1.imshow(
                np.flipud(val_segmentation_data[sample_indices[i], :, :, 0]),
                vmin=0.0,
                vmax=1.0,
            )
            a1.set_title("Segmentation #{}".format(sample_indices[i]))
            c = fig.colorbar(img1, fraction=0.046, pad=0.04)
            a2 = fig.add_subplot(num_show, num_col, i * num_col + 3)
            img2 = a2.imshow(
                np.flipud(y_pred_val[sample_indices[i], :, :, 1]), vmin=0.0, vmax=1.0
            )
            a2.set_title("Prediction #{}".format(sample_indices[i]))
            c = fig.colorbar(img2, fraction=0.046, pad=0.04)
            a3 = fig.add_subplot(num_show, num_col, i * num_col + 4)
            img3 = a3.imshow(
                (np.flipud(y_pred_val[sample_indices[i], :, :, 1]) > threshold),
                vmin=0.0,
                vmax=1.0,
            )
            c = fig.colorbar(img3, fraction=0.046, pad=0.04)
            a3.set_title("Thresholded #{}".format(sample_indices[i]))
        plt.savefig(
            run_str + "-samples-" + str(val_round_index) + ".png"
        )

        # Printing total time of this validation round
        if not load_from_save:
            print(
                "\nTotal round time:  {}".format(datetime.datetime.now() - training_time_start)
            )
            print("")

    time_sequence_stop = datetime.datetime.now()

    print("\nTotal training time:   {}".format(time_sequence_stop - time_sequence_start))

    # Arrange results in tables
    metric_labels = [
        "AUROC",
        "best thresh",
        "best TP",
        "best FP",
        "best recall",
        "best precis",
        "fuzzy recall",
        "fuzzy precis",
        "fuzzy Fscore",
    ]

    results_labels = []

    for label in metric_labels:
        results_labels.append("Vali " + label)

    results_df = pd.DataFrame(columns=results_labels)

    for i in range(num_validation_rounds):
        if i in val_best_metrics.keys():
            results_df.loc[i] = [
                val_aurocs[i],
                val_best_thresholds[i],
                val_best_metrics[i][evaluation_metrics.TRUE_POSITIVE_RATE],
                val_best_metrics[i][evaluation_metrics.FALSE_POSITIVE_RATE],
                val_best_metrics[i][evaluation_metrics.RECALL],
                val_best_metrics[i][evaluation_metrics.PRECISION],
                val_fuzzy_metrics[i][evaluation_metrics.RECALL],
                val_fuzzy_metrics[i][evaluation_metrics.PRECISION],
                val_fuzzy_metrics[i][evaluation_metrics.FSCORE],
            ]

    # Save results table
    csv_filename = run_str + ".csv"
    csv_fullname = os.path.join(results_save_fullpath, csv_filename)
    results_df.to_csv(csv_fullname)

    print("Results saved to: {}".format(csv_fullname))


#train(batch_size=128, num_epochs=500, sagittal_only=False, num_frames=3, num_layers=5, dropout=0.5, use_transforms=True)
#train(batch_size=128, num_epochs=500, sagittal_only=False, num_frames=3, num_layers=4, dropout=0.5, use_transforms=True)
#train(batch_size=128, num_epochs=500, sagittal_only=False, num_frames=3, num_layers=3, dropout=0.5, use_transforms=True)

train(batch_size=128, num_epochs=500, sagittal_only=False, num_frames=3, num_layers=5, dropout=0.5)
train(batch_size=128, num_epochs=500, sagittal_only=False, num_frames=3, num_layers=4, dropout=0.5)
train(batch_size=128, num_epochs=500, sagittal_only=False, num_frames=3, num_layers=3, dropout=0.5)
