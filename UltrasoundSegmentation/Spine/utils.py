import girder_client
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def create_standard_project_folders(local_data_folder):
    # These subfolders will be created/populated in the data folder

    data_arrays_folder    = "DataArrays"
    notebooks_save_folder = "SavedNotebooks"
    results_save_folder   = "SavedResults"
    models_save_folder    = "SavedModels"
    val_data_folder       = "PredictionsValidation"

    data_arrays_fullpath = os.path.join(local_data_folder, data_arrays_folder)
    notebooks_save_fullpath = os.path.join(local_data_folder, notebooks_save_folder)
    results_save_fullpath = os.path.join(local_data_folder, results_save_folder)
    models_save_fullpath = os.path.join(local_data_folder, models_save_folder)
    val_data_fullpath = os.path.join(local_data_folder, val_data_folder)

    if not os.path.exists(data_arrays_fullpath):
        os.makedirs(data_arrays_fullpath)
        print("Created folder: {}".format(data_arrays_fullpath))

    if not os.path.exists(notebooks_save_fullpath):
        os.makedirs(notebooks_save_fullpath)
        print("Created folder: {}".format(notebooks_save_fullpath))

    if not os.path.exists(results_save_fullpath):
        os.makedirs(results_save_fullpath)
        print("Created folder: {}".format(results_save_fullpath))

    if not os.path.exists(models_save_fullpath):
        os.makedirs(models_save_fullpath)
        print("Created folder: {}".format(models_save_fullpath))

    if not os.path.exists(val_data_fullpath):
        os.makedirs(val_data_fullpath)
        print("Created folder: {}".format(val_data_fullpath))
    
    return data_arrays_fullpath, notebooks_save_fullpath, results_save_fullpath, models_save_fullpath, val_data_fullpath


def load_girder_data(csv_fullname, data_arrays_fullpath, girder_url, girder_key=None, overwrite_existing_files=False):
    """
    Download numpy array files from a Girder server to a local folder. Then load them from the local folder to the
    memory as numpy arrays and return them. Optionally, files can be overwritten.
    :param csv_fullname: CSV file containing Girder IDs and subject IDs (e.g. patient) for all files.
    :param data_arrays_fullpath: Local folder to be used. Must have write access.
    :param girder_url: Internet address of the Girder server API
    :param girder_key: (optional) API key for private Girder collections.
    :param overwrite_existing_files: Set True to force overwrite of existing files (default False).
    :return: Ultrasound and matching segmentation arrays, one for each subject (e.g. patient)
    """
    csv_df = pd.read_csv(csv_fullname, sep=",")
    n_arrays = csv_df.shape[0]
    groupby_subjects = csv_df.groupby('subject_id')
    n_subjects = len(groupby_subjects.groups.keys())

    gclient = girder_client.GirderClient(apiUrl=girder_url)
    if girder_key is not None:
        gclient.authenticate(apiKey=girder_key)

    # Download

    for i in range(n_arrays):
        ultrasound_fullname = os.path.join(data_arrays_fullpath, csv_df.iloc[i]['ultrasound_filename'])
        if not os.path.exists(ultrasound_fullname) or overwrite_existing_files:
            print("Downloading {}...".format(ultrasound_fullname))
            gclient.downloadFile(csv_df.iloc[i]['ultrasound_id'], ultrasound_fullname)

        segmentation_fullname = os.path.join(data_arrays_fullpath, csv_df.iloc[i]['segmentation_filename'])
        if not os.path.exists(segmentation_fullname) or overwrite_existing_files:
            print("Downloading {}...".format(segmentation_fullname))
            gclient.downloadFile(csv_df.iloc[i]['segmentation_id'], segmentation_fullname)

    # Load arrays from local files

    ultrasound_arrays = []
    segmentation_arrays = []

    for i in range(n_arrays):
        ultrasound_fullname = os.path.join(data_arrays_fullpath, csv_df.iloc[i]['ultrasound_filename'])
        segmentation_fullname = os.path.join(data_arrays_fullpath, csv_df.iloc[i]['segmentation_filename'])

        ultrasound_data = np.load(ultrasound_fullname)
        segmentation_data = np.load(segmentation_fullname)

        ultrasound_arrays.append(ultrasound_data)
        segmentation_arrays.append(segmentation_data)

    # Concatenate arrays by subjects (e.g. patients)

    ultrasound_arrays_by_subjects = []
    segmentation_arrays_by_subjects = []

    subject_ids = groupby_subjects.groups.keys()

    ultrasound_pixel_type = ultrasound_arrays[0].dtype
    segmentation_pixel_type = segmentation_arrays[0].dtype

    for subject_id in subject_ids:
        subject_ultrasound_array = np.zeros([0, ultrasound_arrays[0].shape[1], ultrasound_arrays[0].shape[2], 1],
                                            dtype=ultrasound_pixel_type)

        subject_segmentation_array = np.zeros([0, segmentation_arrays[0].shape[1], segmentation_arrays[0].shape[2], 1],
                                              dtype=segmentation_pixel_type)

        for i in range(len(groupby_subjects.groups[subject_id])):
            array_index = groupby_subjects.groups[subject_id][i]
            subject_ultrasound_array = np.concatenate([subject_ultrasound_array, ultrasound_arrays[array_index]])
            subject_segmentation_array = np.concatenate([subject_segmentation_array, segmentation_arrays[array_index]])

        ultrasound_arrays_by_subjects.append(subject_ultrasound_array)
        segmentation_arrays_by_subjects.append(subject_segmentation_array)

    return ultrasound_arrays_by_subjects, segmentation_arrays_by_subjects

class PlotLosses(tf.keras.callbacks.Callback):
    def __init__(self, fname_tag):
        self.fname_tag = fname_tag

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        f, ax1 = plt.subplots(1, 1, sharex=True)
                
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="v loss")
        ax1.set_yscale('log')
        ax1.legend()        
        plt.show()
        plt.savefig(self.fname_tag + '-live_metrics.png', dpi=1000)
        plt.close()