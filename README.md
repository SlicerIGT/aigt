# UsAnnotationExport
3D Slicer module to export ultrasound annotations for machine learning.
Notebooks folder contains scripts to further process exported data, and some example machine learning methods.

# Getting started
## Windows
- Install CUDA 9.0
- Install CuDNN for CUDA 9.0
- Install Anaconda
- Run the setup_env.bat file (in SetupAnaconda folder) to create project folder with dedicated Anaconda environment.
- Clone this repository on your computer.
- Some notebooks will require that you createa a new file in the Notebooks folder, named local_vars.py, and define the root_folder variable in that file, e.g. root_folder = r"c:\Data"
- Install Slicer 4.10 or newer version.
- To install the Slicer extension, drag and drop UsAnnotationExport.py in Slicer / Edit / Application settings / Modules / Additional module paths. And restart Slicer.
- To run notebooks, start the Anaconda command prompt, navigate to the Notebooks folder of the clone of this repository, and type the "jupyter notebook" command.
