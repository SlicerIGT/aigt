import os


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