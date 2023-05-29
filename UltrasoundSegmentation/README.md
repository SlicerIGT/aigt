# Ultrasound Segmentation
This folder includes scripts to prepare data exported from 3D Slicer and utilize that data to train an AI model that predicts ultrasound segmentations.

## Exporting Data
---
In 3D Slicer, segmented data can be exported in the Single Slice Segmentation module under Ultrasound, which can be loaded from the aigt repository. 

* Click the checkbox on "Export as .npy" 
* Click "Export segmentation sequence" to begin the exporting process. 3D Slicer will export the segmentation according to the settings of the "Input/output selections" configuration on Single Slice Segmentation.
* File name prefix can be anything, but it's best to keep track each data so name it something relevant to the segmentation like "001-right-AK" which specifies that the exported data is from patient 001, it is the right kidney segmentation, and it was completed by Andrew Kim.
* The output folder can be changed, and will contain the files that are exported. E.g. "C:/Data/KidneySegmentationOutput"


## prepare_data.py
---
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
python prepare_data.py --input_dir "C:/Users/Andrew/Desktop/Andrew USRA 2023/KidneySegmentationOutput" --output_dir "C:/Users/Andrew/Desktop/Andrew USRA 2023/KidneyTrainingData_128" --config_file "prepare_data_config.yaml" --log_file "PrepareData.log"
```

In Visual Studio Code, a launch.json configuration can be used to define the debug configuration, rather than running the scripts on command line. To open the JSON configuration, navigate to the "Debug and Run" menu option on the left taskbar of VSCode. Click the gear cog/the settings symbol in the top next to "RUN AND DEBUG". This will open a launch.json file. prepare_data.py can be configured with the following structure:

```
{
    "name": "Kidney: prepare_data_128",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "justMyCode": "true",
    "args": ["--input_dir", "C:/Users/Bob/Desktop/KidneySegmentationOutput",
            "--output_dir", "C:/Users/Bob/Desktop/TrainingData",
            "--config_file", "prepare_data_config.yaml",
            "--log_level", "INFO",
            "--log_file", "PrepareData.log",]
        }
```

Copy and paste the above dictionary into the configurations list.

`"name"` is the name of the launch configuration and can be changed to any name as long as it is a string. They can be altered to match the location of your directories and desired file names. `"type"`, `"request"`, `"program"`, `"console"`, and `"justmycode"` should be untouched. `"args"` should contain a list of input arguments like shown in the code block above.

This will add a configuration option so that you can select it as a debug configuration by clicking the down arrow in the top of the "Run and Debug" menu next to the gear cog/settings

With the output of this script, you can separate your data into different folders. Create a folder for a training set, validation set, and testing set. Pick a few participants' data from the output folder and move them to the testing set folder. Pick another few and move them to the validation set folder. The remaining data in the folder can be used to train the model.

## Train.py
---
This script takes prepared data as input and trains an AI model for ultrasound segmentation prediction. The results of scores like f1 score, accuracy, train loss, validation loss, etc. are plotted on a graph on **_Weights and Biases_**. The first time the script is ran, it will ask you for your _Weights and Biases_ API key. Create an account at _Weights and Biases_ and your API key can be found in the "User Settings" tab. The model can be customized, as well as the graphs to display on _Weights and Biases_.

Training hyperparameters can be modified on train_config.yaml.

Similar to running the prepare_data.py script, train.py can be run from command line or by configuring a JSON file.

The output for this script is a PyTorch model that is trained from your training data.

* `--train-data-folder` should be the path of the folder with the training set (which should be a subset of the output of prepare_data.py)
* `--val-data-folder` should be the path of the folder with validation set 
* `--config-file` is by default `"train_config.yaml"`
* `--save-torchscript` saves the model as a torchscript
* `--save-ckpt-freq` is the integer value for how often (number of epochs) the model saves and is 0 by default
* `--wandb-project-name` should be the name of the project in _Weights and Biases_ and has a default name
* `--wandb-exp-name` is the experiment name in _Weights and Biases_ and is by default the timestamp of the experiment
* `--log-level` is by default `"INFO"`
* `--save-log` saves the log to a file named train.log.

Example command line code to run train.py looks like this:
```
python train.py --train-data-folder "C:/Users/Andrew/Desktop/Andrew USRA 2023/KidneyTrainingData_128" --val-data-folder "C:/Users/Andrew/Desktop/Andrew USRA 2023/KidneyValidation_128" --output-dir "C:/Users/Andrew/Desktop/Andrew USRA 2023/KidneyRuns_128" --config file --wandb-project-name "KidneySegmentation_128" --save-log
```

To configure the JSON file for train.py, open the launch.json again. Copy and paste the following block of code after a comma after the previous configuration dictionary:

```
{
            "name": "Kidney: train_128",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": "true",
            "args": ["--train-data-folder", "C:/KidneyTrainingData_128",
                     "--val-data-folder", "C:/KidneyValidation_128",
                     "--output-dir", "C:/KidneyRuns_128",
                     "--config-file", "train_config.yaml",
                     "--save-log",
                     "--wandb-project-name", "KidneySegmentation_128",
                     "--save-torchscript"]
        }
```

Similar to the configuration for prepare_data.py, `"name"` can be changed, but `"type"`, `"request"`, `"program"`, `"console"`, and `"justmycode"` should be untouched. Argument parameters are outlined above, but should be in string format and separated by commas in the configuration.