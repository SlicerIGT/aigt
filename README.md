# AIGT
This repository contains scripts for deep learning for guided medical interventions. For some projects, the complete workflow is implemented, from formatting and annotations to deployment of models in real time. Most projects use ultrasound imaging.
The source code was originally written for TensorFlow 1.13 and Keras 2.2, but all active projects will gradually migrate to TensorFlow 2.0 (tf.keras). If no TensorFlow version is specified in a comments at the top of a file, then it is still using the older software version.

# Getting started
## Install and set up Anaconda environment
- Install [Anaconda (with Python 3.7)](https://www.anaconda.com/distribution/)
- Clone this repository on your computer. Your local clone may be in a path like `c:\dev\aigt`
- Start the *Anaconda Prompt* application and navigate to the environment setup folder `cd c:\dev\aigt\SetupAnaconda`
- Run the *setup_env.bat* file to create environment in a folder, e.g. `setup_env.bat c:\dev\dlenv`
This will install TensorFlow 2.0, as we are using tf.keras by default. The previous environment setup script is still available as `setup_env_tf1.bat`
## Additional local files you may need, but they are not in the code repository
Please do not commit/push your local_vars.py, as everybody sets it up with their own local paths!
- **local_vars.py** - Some notebooks require a file in the Notebooks folder of your local repository clone, named **local_vars.py**. This file should define the *root_folder* variable. The file may just contain this single line of code: `root_folder = r"c:\Data"`. 
- **girder_apikey_read.py** - Some notebooks require a file named **girder_apikey_read.py** with a single line of code that specifies your personal API key to the private data collection. If you work with non-public data stored on a Girder server, ask your supervisor for a Girder account and how to generate API keys for yourself.
## To run Slicer notebooks
- Install Slicer 4.11 or newer version (later than 2019-09-16 is recommended, for full functionality)
- Install the *SlicerJupyter* extension for Slicer, and follow the extension user guide to add Slicer as a kernel in Jupyter Notebook (use the *Copy command to clipboard* button and paste it in the active Anaconda environment).
- Install additional packages in the Slicer python environment, to be able to run all Slicer notebooks. Use the Python console of Slicer to run this command (you may change the tensorflow version to 2.0 and skip keras , if you are already using tf.keras):
```
pip_install("tensorflow==1.14.0 keras==2.2.4 scikit-learn opencv-contrib-python")
```
- You may also install `tensorflow-gpu`, if your computer supports that. In that case copy CUDA files (dll files on Windows) to the Slicer bin folder. (Feel free to add instructions for Mac and Linux.)
- If you are using the latest version of Slicer, but running an older notebook, then you may need to specify the TensorFlow version to be installed.
- To run notebooks, start the Anaconda command prompt, navigate to the Notebooks folder of your clone of this repository, and type the `jupyter notebook` command.

# Acquire tracked ultrasound
- Use the Sequences extension in 3D Slicer to record tracked ultrasound sequences.
- Record **Image_Image** (the ultrasound in the Image coordinate system) and **ImageToReference** transform sequences. Note that Slicer cannot record transformed images, so recording *Image_Reference* is not an option.
- If you work on segmentation, you can use the *SingleSliceSegmentation* Slicer module in this repository to speed up manual segmentation.
- Create annotations by placing fiducials or creating segmentations with the *SingleSliceSegmentation* module.
- For segmentations, you may use the *SingleSliceSegmentation* module to export images and segmentations in png file format.
- Use scripts in the Notebooks/Slicer folder to export images and annotations.

# Process exported data
- It is recommended that you save separate sequences for validation and testing
- Use Notebooks/FoldersToSavedArrays to save data sequences as single files for faster data loading during training.
