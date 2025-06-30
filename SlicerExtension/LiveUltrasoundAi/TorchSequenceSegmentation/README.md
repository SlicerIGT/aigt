# Torch Sequence Segmentation

Slicer module for generating AI predictions and 3D reconstructions from recorded ultrasound sequences.

## Usage

1. Load a Slicer scene with the recorded ultrasound sequence browser(s). For 3D reconstruction, the sequence browser should include ImageToReference transforms.
2. Select a folder containing trained PyTorch model(s) in TorchScript format.
3. Select a trained model:
  - _Select individual model_: If selected, use the Model dropdown menu to select the TorchScript file to use.
  - _Use all models in folder_: If selected, all valid models will be used for prediction/reconstruction.
4. Select the sequence browser to use as input to the model(s).
  - Check the _Use all sequence browsers_ box to generate predictions/reconstructions for all ultrasound sequences in the scene.
  - Note: users have reported crashes using this option; run one at a time if you want to be safe
5. Select the input volume and volume reconstruction node, if not already correctly selected.
6. _Optional_: Select a tracking integration method.
7. _Optional_: If the model expects a linear image but the input image is curvilinear, specify the scan conversion config file in the _Advanced_ menu.
8. Under _Controls_, toggle/untoggle the actions you want to run. If none selected, will only generate 2D segmentations.
9. Click _Start_ to run the selections.
