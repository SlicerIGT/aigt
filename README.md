# AIGT
This repository contains scripts for deep learning in guided medical interventions. The complete workflow of data processing is implemented, from formatting and annotations to deployment of models in real time.

# Getting started
## Install and set up Anaconda environment
- Install Anaconda (with Python 3.7)
- Run the *setup_env.bat* file (in SetupAnaconda folder) to create environment in a folder.
- Clone this repository on your computer.
- Some notebooks will require that you createa a new file in the Notebooks folder of your local clone, named **local_vars.py**, and define the **root_folder** variable in that file. The file can just contain this single line of code: `root_folder = r"c:\Data"`. Please do not commit/push your local_vars.py, as everybody sets it up with their own local paths!
## To run Slicer notebooks
- Install Slicer 4.11 or newer version (later than 2019-09-16 is recommended, for full functionality)
- Install the Jupyter extension for Slicer, and follow the extension user guide to add Slicer as a kernel in Jupyter Notebook.
- Install additional packages in the Slicer python environment, to be able to run all Slicer notebooks. Use the Python console of Slicer to run this command (you may add additional python libraries that you use in your code):
```
pip_install("tensorflow keras scikit-learn opencv-contrib-python")
```
- You may also install `tensorflow-gpu`, if you computer supports that, but then copy CUDA files (dll files on Windows) to the Slicer bin folder. (Feel free to add instructions for Mac and Linux.)
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
