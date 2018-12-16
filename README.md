# UsAnnotationExport
Notebooks for 3D Slicer  to export ultrasound annotations for machine learning.
Notebooks also contain scripts for processing exported data, and some example deep learning methods.

# Getting started
## Install Windows Anaconda environment with GPU version of tensorflow
- Install CUDA 9.0
- Install CuDNN for CUDA 9.0
- Install Anaconda
- Run the setup_env.bat file (in SetupAnaconda folder) to create project folder with dedicated Anaconda environment.
- Clone this repository on your computer.
- Some notebooks will require that you createa a new file in the Notebooks folder, named local_vars.py, and define the root_folder variable in that file, e.g. root_folder = r"c:\Data"
- Install Slicer 4.10 or newer version.
- To install the Slicer extension, drag and drop UsAnnotationExport.py in Slicer / Edit / Application settings / Modules / Additional module paths. And restart Slicer.
- To run notebooks, start the Anaconda command prompt, navigate to the Notebooks folder of the clone of this repository, and type the "jupyter notebook" command.

# Acquire tracked ultrasound
- Use the Sequences extension in 3D Slicer to record tracked ultrasound sequences.
- Record Image_Image (the ultrasound in the Image coordinate system) and ImageToReference transform sequences.
- Create annotations by placing fiducials or creating a segmentation.
- Save the Slicer scene.
- Use Notebooks/Slicer/AverageIntensities to determine intensity threshold to filter out images with no skin contact.
- Use scripts in the Notebooks/Slicer folder to export images and annotations. These scripts will automatically open a Slicer managed by Jupyter. Load the Slicer scenes in these managed Slicer instances and run the notebooks to export data.

# Process exported data
- Use Notebooks/SplitAnnotatedData to separate part of the data into testing, validation, and training folders.
- Use Notebooks/FoldersToSavedArrays to save data sequences as single files for faster data loading during training.
