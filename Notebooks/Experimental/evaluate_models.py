import argparse
import datetime
import os

import numpy as np

start_timestamp = datetime.datetime.now()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable TensorFlow warnings for cleaner logging

from keras import backend as K
from keras.models import load_model

from local_vars import root_folder


# Parse command line arguments

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models-folder", type=str, required=True)
ap.add_argument("-d", "--data-folder", type=str, required=True)
ap.add_argument("-o", "--output-folder", type=str, required=True)
ap.add_argument("-u", "--ultrasound-filename", type=str, required=False, default="ultrasound-test.npy")
ap.add_argument("-s", "--segmentation-filename", type=str, required=False, default="segmentation-test.npy")
args = vars(ap.parse_args())


# Listing models

models_folder_fullpath = os.path.join(root_folder, args["models_folder"])

models_file_list = [f for f in os.listdir(models_folder_fullpath) if f.endswith('.h5')]
num_models = len(models_file_list)

print("Found {} models in folder {}".format(num_models, models_folder_fullpath))


# Reading data files

data_fullpath = os.path.join(root_folder, args["data_folder"])

test_ultrasound_fullname = os.path.join(data_fullpath, args["ultrasound_filename"])
test_segmentation_fullname = os.path.join(data_fullpath, args["segmentation_filename"])

print("Reading test ultarsound from:   {}".format(test_ultrasound_fullname))
print("Reading test segmentation from: {}".format(test_segmentation_fullname))

test_ultarsound_data = np.load(test_ultrasound_fullname)
test_segmentation_data = np.load(test_segmentation_fullname)

num_ultrasound = test_ultarsound_data.shape[0]
num_segmentation = test_segmentation_data.shape[0]

print("Found {} test ultarsound images and {} segmentations".format(num_ultrasound, num_segmentation))

# Check if loaded ultrasound matches segmentation

if num_ultrasound != num_segmentation:
    raise Exception("Number of images should be equal!")


# Output folder

output_fullpath = os.path.join(root_folder, args["output_folder"])
if not os.path.exists(output_fullpath):
    os.makedirs(output_fullpath)
    print("Creating folder: {}".format(output_fullpath))

print("Will save predictions in folder: {}".format(output_fullpath))


# Iterate through models

for model_filename in models_file_list:
    model_fullname = os.path.join(models_folder_fullpath, model_filename)
    basename, extension = os.path.splitext(model_filename)
    print("Loading model:  {}".format(model_fullname))
    model = load_model(model_fullname)
    prediction = model.predict(test_ultarsound_data)
    prediction_filename = "prediction_" + basename
    prediction_fullname = os.path.join(output_fullpath, prediction_filename)
    np.save(prediction_fullname, prediction)
    print("Execution time: {}".format(datetime.datetime.now()-start_timestamp))
    K.clear_session()
