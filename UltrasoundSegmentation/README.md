# Ultrasound Segmentation
This folder includes scripts to prepare data exported from 3D Slicer and utilize that data to train an AI model that predicts ultrasound segmentations.

## Installing required packages
The required dependencies can be found in the [environment.yml](environment.yml) file. The following section describes the installation process using Anaconda.

* Install Anaconda if it is not already installed on your computer from the [Anaconda Download Page](https://www.anaconda.com/download)
* Clone this repository to your computer
* Open Anaconda Prompt
* Navigate to the "Ultrasound Segmentation" folder of your local clone. The Anaconda Prompt uses the same syntax as the Window Command Prompt.
* Type in the terminal the following code: 
```
conda env create -n environmentName -f environment.yml
```
  * You can change `environmentName` to any name you would like the environment to have
  * Anaconda will create the environment and start loading the dependencies from environment.yml. This may take a while.
* There are two methods to running the scripts and are outlined under the description of each script on this readme:
  * From Anaconda Prompt using terminal commands
  * From Microsoft Visual Studio Code (VS Code) using debug configurations
* To run scripts from Anaconda prompt, the environment must be activated every time Anaconda Prompt is booted. You can see which environment is active from the text in parentheses before the current location. "base" will always be the default upon start up.
  * To activate your environment, enter the following code (replacing name as necessary): 

```
conda activate environmentName
```

  * You will see that the text in the parentheses changes to your environment name
  * To deactivate your environment, enter the following into terminal: 

```
conda deactivate
```

## Exporting data

In 3D Slicer, segmented data can be exported in the Single Slice Segmentation module under Ultrasound, which can be loaded from the aigt repository. 

* Click the checkbox on "Export as .npy" 
* Click "Export segmentation sequence" to begin the exporting process. 3D Slicer will export the segmentation according to the settings of the "Input/output selections" configuration on Single Slice Segmentation.
* File name prefix can be anything, but it's best to keep track each data so name it something relevant to the segmentation like "001-right-AK" which specifies that the exported data is from patient 001, it is the right kidney segmentation, and it was completed by Andrew Kim.
* The output folder can be changed, and will contain the files that are exported. E.g. "C:/Data/KidneySegmentationOutput"


## prepare_data.py

This script takes the exported data from 3D Slicer, filters the data to what is needed to train the model, then resizes the image to what is specified in prepare_data_config.yaml. The prepare_data_config.yaml file contains settings that can be altered to fit the needs of the study.

To run this script, you can use the command line or configure a JSON file if you are on Visual Studio Code. Both methods are below.

For the command line method, open command line and navigate to the working directory. 
* To view the arguments that are required, type `python fileName -h` and replace "fileName" with the name of the file you would like to run (prepare_data.py).

* `--input_dir` should be the folder with the segmentations that were exported from 3D Slicer 
* `"--output_dir"` should be the folder for the output of this script
* `--config_file` should be `prepare_data_config.yaml`
* `--log_level` should be `INFO`
* `"--log_file"` should be the name of a file where the log will end up, and should be a .log file.

Example command line code to run prepare_data.py looks like this:
```
python prepare_data.py --input_dir d:/SegmentationOutput --output_dir d:/PatientArrays --config_file prepare_data_config.yaml --log_file "PrepareData.log"
```

In Visual Studio Code, a launch.json configuration can be used to define the debug configuration, rather than running the scripts on command line. To open the JSON configuration, navigate to the "Debug and Run" menu option on the left taskbar of VSCode. Click the gear cog/the settings symbol in the top next to "RUN AND DEBUG". This will open a launch.json file. prepare_data.py can be configured with the following structure:

```
{
    "name": "prepare_data",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "justMyCode": "true",
    "args": ["--input_dir", "D:/SegmentationOutput",
             "--output_dir", "D:/PatientArrays",
             "--config_file", "prepare_data_config.yaml",
             "--log_level", "INFO",
             "--log_file", "PrepareData.log",]
}
```

Copy and paste the above dictionary into the configurations list.

`"name"` is the name of the launch configuration and can be changed to any name as long as it is a string. They can be altered to match the location of your directories and desired file names. `"type"`, `"request"`, `"program"`, `"console"`, and `"justmycode"` should be untouched. `"args"` should contain a list of input arguments like shown in the code block above.

This will add a configuration option so that you can select it as a debug configuration by clicking the down arrow in the top of the "Run and Debug" menu next to the gear cog/settings

With the output of this script, you can separate your data into different folders. Create a folder for a training set, validation set, and testing set. Pick a few participants' data from the output folder and move them to the testing set folder. Pick another few and move them to the validation set folder. The remaining data in the folder can be used to train the model.

## Extract scanlines (optional)

If the ultrasound images are recorded usinga  curvilinear transducer, training and inference may be more efficient if the scan lines are extracted from the image and arranged in a rectangular array as one column for each scan line. Since the output image size can be controlled by the scan line configuration (see ```--scanconvert-config``` commandline argument), this step can also be used to resize input images.

Example launch.json configuration for running extract_scanlines.py

```
        {
            "name": "Extract scanlines: training 128",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": ["--input-dir", "g:/Spine/TrainingData_0_512",
                     "--output-dir", "g:/Spine/TrainingData_sc_0_128",
                     "--scanconvert-config", "scanconvert_config.yaml",
                     "--log-file", "extract_scanlines.log"]
        },
```

Example content for ```scanconvert_config.yaml``` producing 128x128 linear scan images from 512x512 curved scan input images:

```
num_lines: !!int 128
num_samples_along_lines: !!int 128
curvilinear_image_size: !!int 512
center_coordinate_pixel: [0, 256]
radius_start_pixels: 100
radius_end_pixels: 420
angle_min_degrees: -36
angle_max_degrees: 36
```

## convert_to_slice.py

The 3D patient arrays need to be converted into individual 2D slices for training. This can be done using the [convert_to_slice.py](convert_to_slice.py) script:

```
python convert_to_slice.py --data-folder d:/PatientArrays --output-dir d:/data/train
```

* `--data-folder`: path to directory containing the 3D patient .npy files
* `--output-dir`: path to directory to save 2D slices (as .npy files)
* `--use-file-prefix`: include this argument to use the 3D patient file prefixes as folder names

The output files will be saved in the following folder structure:

```
output-dir   
└───images
│   └───0000 (default, can be replaced using --use-file-prefix)
│   │   └───0000_ultrasound.npy
│   │   └───0001_ultrasound.npy
│   │   └───...
│   
└───labels
│   └───0000
│   │   └───0000_segmentation.npy
│   │   └───0001_segmentation.npy
│   │   └───...
|
└───transforms
│   └───0000 (folder will be empty if no transforms exist)
│   │   └───0000_transform.npy
│   │   └───0001_transform.npy
│   │   └───...
```

## train.py

This script takes prepared data as input and trains an AI model for ultrasound segmentation prediction. The results of scores like f1 score, accuracy, train loss, validation loss, etc. are plotted on a graph on **_Weights and Biases_**. The first time the script is ran, it will ask you for your _Weights and Biases_ API key. Create an account at _Weights and Biases_ and your API key can be found in the "User Settings" tab. The model can be customized, as well as the graphs to display on _Weights and Biases_.

Training hyperparameters can be modified on train_config.yaml.

Similar to running the prepare_data.py script, train.py can be run from command line or by configuring a JSON file.

* `--train-data-folder` should be the path of the folder with the training set (which should be a subset of the output of prepare_data.py)
* `--val-data-folder` should be the path of the folder with validation set 
* `--output-dir` is the name of the directory in which to save the run
* `--config-file` is the yaml file detailing the training settings. See [train_config.yaml](train_config.yaml) for an example. Also see [Supported networks](#supported-networks) to see the available networks.
* `--save-torchscript` saves the model as a torchscript
* `--save-ckpt-freq` is the integer value for how often (number of epochs) the model saves and is 0 by default
* `--wandb-entity-name` should be set to your username if you are working on a solo project or the username of the owner of a collaborative team on wandb
* `--wandb-project-name` should be the name of the project in _Weights and Biases_ and has a default name
* `--wandb-exp-name` is the experiment name in _Weights and Biases_ and is by default the model name and timestamp of the experiment
* `--log-level` is by default `"INFO"`
* `--save-log` saves the log to a file named train.log.
* `--nnunet-dataset-name` is the name of the dataset used by nnUNet (see [below](#using-the-nnunet))
* `--verify-nnunet-dataset` checks dataset integrity if included

The output of this script is a PyTorch model that is trained from your training data. For using the trained model for 3D volume reconstruction with the [Slicer module](../SlicerExtension/LiveUltrasoundAi/TorchSequenceSegmentation/), it is necessary to include the `--save-torchscript` flag.

Example command line code to run train.py looks like this:
```
python train.py --train-data-folder d:/data/train --val-data-folder d:/data/val --output-dir d:/runs --save-torchscript --save-log
```

To configure the JSON file for train.py, open the launch.json again. Copy and paste the following block of code after a comma after the previous configuration dictionary:

```
{
    "name": "train",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "justMyCode": "true",
    "args": ["--train-data-folder", "D:/data/train",
             "--val-data-folder", "D:/data/val",
             "--output-dir", "D:/runs",
             "--save-torchscript", 
             "--save-log"]
}
```

Similar to the configuration for prepare_data.py, `"name"` can be changed, but `"type"`, `"request"`, `"program"`, `"console"`, and `"justmycode"` should be untouched. Argument parameters are outlined above, but should be in string format and separated by commas in the configuration.

## Testing trained models

- You may use `test_models.py` to run metric calculations on a list of trained models on a specified test dataset. Look at the comment in `test_models.py` for how to specify input for that script.
- `test_models.py` uses `test.py`, which may also be used directly if you only have one trained model to test.
- Test result metrics will be exported in CSV files as they are specified in by the `--models_csv` argument.

Example entry for VSCode's `launch.json`:
```
{
    "name": "test_models: Study",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "justMyCode": true,
    "args": ["--models_csv", "f:/SpineUs/SegStudy/Round2/Round2ModelsTest.csv",
              "--test_data_path", "f:/SpineUs/SegStudy/TestingData_sc_0_128_Slices",
              "--num_sample_images", "10"
            ]
}
```

## Supported networks

The network architectures that are currently supported (and their required `names` in the config file) are listed below:

- UNet (`unet`)
- UNet with EfficientNetB4 as backbone (`effnetunet`)
- Attention UNet (`attentionunet`)
- UNet++ (`unetplusplus`)
- UNETR (`unetr`)
- SegResNet (`segresnet`)
- nnUNet (`nnunet`)

Using any of the networks other than the nnUNet will use the hyperparameters described in the config file. Otherwise, due to the nature of the nnUNet, many of the hyperparameters will be automatically set and the config file will be ignored. However, using the nnUNet requires additional packages and flags to be set. This will be described in the following section.

### Using the nnUNet

The implementation of the nnUNet is found on [GitHub](https://github.com/MIC-DKFZ/nnUNet/tree/master). Most of the instructions in this guide can be found in much more detail on the [official repo](https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation). This section will explain what is necessary to use the nnUNet from the aigt repo.

### Installation

The source code must first be downloaded by cloning the official repository. From the UltrasoundSegmentation folder of the aigt repo, run the following commands:

```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

In addition, you will need to create 3 folders: `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`. These three folders can be located anywhere you wish, but their locations must be saved as separate environment variables. More detailed instructions can be found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).

### Dataset conversion

The nnUNet requires the data to be in a format that is notably different from the format used for training all other networks in this repository. Specifically, it requires all training and validation *images* to be in the *same folder*. Likewise, all corresponding training and validation *segmentations* are to be the same separate folder. These two folders need to be located in one `Dataset` folder that is located in the `nnUNet_raw` folder. The folder structure of `nnUNet_raw` should look something like this (`nnUNet_preprocessed` and `nnUNet_results` are kept empty):

```
nnUNet_raw  
└───Dataset001_Breast
│   └───imagesTr
│   │   └───LN003_0000_0000.nii.gz
│   │   └───LN003_0001_0000.nii.gz
│   │   └───LN003_0002_0000.nii.gz
│   │   └───...
│   └───labelsTr
│   │   └───LN003_0000.nii.gz
│   │   └───LN003_0001.nii.gz
│   │   └───LN003_0002.nii.gz
│   │   └───...
│   └───dataset.json
```

All training data from one dataset are located under the `Dataset` folder, which must be named in the following convention: `Dataset{ID}_{NAME}`, where `ID` is a unique 3-digit integer and `NAME` is a string. This is the string that should be provided to the `--nnunet_dataset-name` argument in `train.py`. **Without this flag, the nnUNet will not run.** `ID` can be whatever (3-digit) integer you like, as long as there is no other dataset in the `nnUNet_raw` folder that has that ID.

The images themselves follow this naming convention: `{CASE_IDENTIFIER}_{CHANNEL}.{FILE_ENDING}`. In the above example, `LN003_0000` is the case identifier, `0000` is the 4-digit channel identifier, and `nii.gz` is the file ending. The segmentations follow a similar convention but without the channel indicator. This will be further explained below.

Luckily, assuming your data has already been converted to slices (hopefully using the `--use-file-prefix` flag), `train.py` will convert the data into the nnUNet format for you. However, you must first set the values of 2 dictionaries in the `.yaml` config file. An example is shown in [train_config.yaml](train_config.yaml):

```
# For nnUNet only:
channel_names:
  0: "Ultrasound"

labels:
  background: 0
  tumor: 1
```

The first dictionary, `channel_names`, corresponds to the 4-digit channel identifier of the nifti files in `imagesTr`. The nnUNet requires each channel/modality of an image to be in a separate file. This is mostly relevant in MRI images where you may have T1 or T2-weighted scans, which is why there is only one channel in the example above. For most cases in this repo, this dictionary can be left as-is.

The second dictionary, `labels`, matches a string label to each integer class in the segmentation images. For multi-class segmentation, you will need to add more key/value pairs to this dictionary.

More information about the data format can be found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

### Running `train.py`

If you have completed the above instructions, you are now ready to run `train.py`. First, make sure you change the `model_name` parameter in the config file to "nnunet". Then, run `train.py` as you would with other models following [the previous instructions](#trainpy), but make sure you include the `--nnunet-dataset-name` flag with your desired name. You may also choose to set the `--verify-nnunet-dataset` flag to check that everything is formatted correctly.

The first time you run `train.py` using the nnUNet, it will convert the slice dataset to the nnUNet format as described above. It will also find the optimal preprocessing steps, model architecture, and training settings for your dataset, referred to as the plan. Any subsequent runs using the same dataset will skip the conversion and the finding of the plan and use the existing one. These plans can be found in the `nnUNet_preprocessed` folder.

One note on the data splits: by default, the nnUNet runs a 5-fold cross validation on all the data in the `imagesTr` folder. However, since we have now mixed all of our patient data together into one folder, randomly splitting the data this way will cause biases in our model. To mitigate this, a custom split will be generated automatically using the same training/validation split that was defined for training other networks in this repo (i.e. you don't need to do anything, this is just an FYI). The split can be found in a `splits_final.json` file in the `nnUNet_preprocessed` folder once a plan has been generated.
