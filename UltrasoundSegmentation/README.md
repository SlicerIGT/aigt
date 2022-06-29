# Ultrasound Segmentation
This work contains scripts for training deep learning models for ultrasound 
image segmentation. Specifically, we focus on generating accurate and clinically 
effective segmentations of the spine and breast tumors. Models generated 
here can be used in various applications within SlicerIGT, including the Slicer 
extensions [LumpNav](https://github.com/SlicerIGT/LumpNav) and 
[SegmentationUNet](https://github.com/SlicerIGT/aigt/tree/master/SlicerExtension/LiveUltrasoundAi/SegmentationUNet), 
or in any other related projects.

## Getting started
### Install and setup Anaconda environment
- Follow the instructions [here](https://github.com/SlicerIGT/aigt/blob/a1d3d835e511ee32729dfbd3ea0e5cd9158c9458/README.md?plain=1#L6) 
  for installing Anaconda and cloning this repository
- Start an _Anaconda Prompt_ and navigate to the `aigt/UltrasoundSegmentation` 
  directory on your local repository
- Run the following command to create a new Anaconda environment with the 
  required dependencies (the new environment will be in the `envs` directory in your
  local conda directory):

`conda env create -f environment.yml`

- Activate the new environment using:

`conda activate AIGT-UltrasoundSegmentation`

### Basic usage
Ensure you have the `girder_apikey_read.py` file containing the API key to 
the private Girder collection in the `aigt/UltrasoundSegmentation` directory
(details described [here](https://github.com/SlicerIGT/aigt/blob/a1d3d835e511ee32729dfbd3ea0e5cd9158c9458/README.md?plain=1#L14)).

Models are trained using the [`train.py`](https://github.com/SlicerIGT/aigt/blob/master/UltrasoundSegmentation/train.py) 
file which is run from the command line using the syntax below (using example 
filepaths). All three arguments in the command are required to run the script.

`python train.py --save_folder c:\Data\BreastUltrasound --config_yaml 
config.yaml --girder_csv BreastGirder.csv`

**_--save_folder:_** path to the directory in which to save all project files 
(including downloaded data, saved models, results, and logs)

**_--config_yaml:_** path to a `.yaml` file detailing the training settings (an example 
can be found [here](https://github.com/SlicerIGT/aigt/blob/master/UltrasoundSegmentation/config.yaml))

**_--girder_csv:_** path to a `.csv` file containing the Girder IDs and subject 
IDs for all image files, example table below:

| subject_id | ultrasound_id | ultrasound_filename | segmentation_id | segmentation_filename |
|------------|---------------|---------------------|-----------------|-----------------------|
| 1          | a1b2c3        | ultrasound1.npy     | i9h8g7          | segmentation1.npy     |
| 2          | d4e5f6        | ultrasound2.npy     | f6e5d4          | segmentation2.npy     |
| 3          | g7h8i9        | ultrasound3.npy     | c3b2a1          | segmentation3.npy     |

A brief description of all command-line arguments can be shown using:

`python train.py --help`
