# UsAnnotationExport
Notebooks for 3D Slicer  to export ultrasound annotations for machine learning.
Notebooks also contain scripts for processing exported data, and some example deep learning methods.

# Getting started
## Install Anaconda
- Install Anaconda (with Python 3.7)
## Set up Anaconda environment with GPU version of TensorFlow
- Run the *setup_env.bat* file (in SetupAnaconda folder) to create environment in a folder.
- Clone this repository on your computer.
- Some notebooks will require that you createa a new file in the Notebooks folder of your local clone, named local_vars.py, and define the root_folder variable in that file, e.g. root_folder = r"c:\Data". Please do not commit/push your local_vars.py!
## To run Slicer notebooks
- Install Slicer 4.10 or newer version.
- Install the Jupyter extension for Slicer, and follow the extension user guide to add the Python environment of Slicer as a kernel in Jupyter Notebook.
- To run notebooks, start the Anaconda command prompt, navigate to the Notebooks folder of your clone of this repository, and type the "jupyter notebook" command.

# Acquire tracked ultrasound
- Use the Sequences extension in 3D Slicer to record tracked ultrasound sequences.
- Record **Image_Image** (the ultrasound in the Image coordinate system) and **ImageToReference** transform sequences. Note that Slicer cannot record transformed images, so recording *Image_Reference* is not an option.
- Create annotations by placing fiducials or creating segmentations.
- Save the Slicer scene.
- Use Notebooks/Slicer/AverageIntensities to determine intensity threshold and image region for settinput up filter in later notebooks to skip images with no skin contact.
- Use scripts in the Notebooks/Slicer folder to export images and annotations. These scripts will automatically open a Slicer managed by Jupyter. Load the saved Slicer scenes in these managed Slicer instances and run the notebooks to export data.

# Process exported data
- (optional) Use Notebooks/SplitAnnotatedData to separate part of the data into testing, validation, and training folders.
- Use Notebooks/FoldersToSavedArrays to save data sequences as single files for faster data loading during training.
