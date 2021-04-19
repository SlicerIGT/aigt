# AIGT
This repository contains scripts for deep learning for guided medical interventions. For some projects, the complete workflow is implemented, from formatting and annotations to deployment of models in real time. Most projects use ultrasound imaging.

# Getting started
## Install and set up Anaconda environment
- Install [Anaconda (with Python 3.7)](https://www.anaconda.com/distribution/)
- Clone this repository on your computer. Your local clone may be in a path like `c:\dev\aigt`
- Start the *Anaconda Prompt* application and navigate to the environment setup folder `cd c:\dev\aigt\SetupAnaconda`
- Run the *setup_env.bat* file to create environment in a folder, e.g. `setup_env.bat c:\dev\dlenv`
This will install TensorFlow 2.0 and other packages that are used by projects. The previous environment setup script (for TensorFlow v1 is still available as `setup_env_tf1.bat`
## Additional local files you may need, but they are not in the code repository
Please do not commit/push these local files, as everybody sets them up with values that only apply to their environment.
- **local_vars.py** - Some notebooks require a file in the Notebooks folder of your local repository clone, named **local_vars.py**. This file should define the *root_folder* variable. The file may just contain this single line of code: `root_folder = r"c:\Data"`. 
- **girder_apikey_read.py** - Some notebooks require a file named **girder_apikey_read.py** with a single line of code that specifies your personal API key to the private data collection. If you work with non-public data stored on a Girder server, ask your supervisor for a Girder account and how to generate API keys for yourself.
## To run Slicer notebooks
- Install Slicer 4.11 or newer version (later than 2019-09-16 is recommended, for full functionality)
- Install the *SlicerJupyter* extension for Slicer, and follow the extension user guide to add Slicer as a kernel in Jupyter Notebook (use the *Copy command to clipboard* button and paste it in the active Anaconda environment).
- If you have a GPU and would like Slicer's TensorFlow to use it, then install CUDA 10.1 and cuDNN 7.6.5. GPUs can make training of models much faster, but may not significantly speed up trained models for prediction compared to CPUs.
- Some users have reported that they needed to install Visual Studio (2015 or later) to be able to use TensorFlow.
- Install additional packages in the Slicer python environment, to be able to run all Slicer notebooks. Use the Python console of Slicer to run this command:
```
pip_install("tensorflow opencv-contrib-python girder_client pandas nbformat nbconvert")
```
- To run notebooks, start the Anaconda command prompt, navigate to the aigt folder, and type the `jupyter notebook` command.

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

# Acknowledgements

![Canarie Logo](Logos/CanarieLogo.png) This work was supported through CANARIE’s Research Software Program through RS3-036. Principal investigator: Gabor Fichtinger.
