import logging
import os
import numpy as np
import time
import vtk, qt, ctk, slicer

# Import yaml. If import fails, install yaml package

try:
    import yaml
except:
    slicer.util.pip_install('pyyaml')
    import yaml

# Import scipy. If import fails, install scipy package

try:
    from scipy.ndimage import map_coordinates
    from scipy.interpolate import griddata
    from scipy.spatial import Delaunay
except:
    slicer.util.pip_install('scipy')
    from scipy.ndimage import map_coordinates
    from scipy.interpolate import griddata
    from scipy.spatial import Delaunay

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

try:
    import torch
except:
    logging.error('TorchLiveUs module requires PyTorch to be installed')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


try:
    import cv2
except:
    logging.error('TorchLiveUs module requires OpenCV to be installed')


IMAGE_SIZE = 128
IMAGE_CHANNELS = 1

#
# TorchLiveUs
#

class TorchLiveUs(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Torch Live Ultrasound"
        self.parent.categories = ["Ultrasound"]
        self.parent.dependencies = []
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]

        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#TorchLiveUs">module documentation</a>.
"""

        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # TorchLiveUs1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='TorchLiveUs',
        sampleName='TorchLiveUs1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'TorchLiveUs1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='TorchLiveUs1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='TorchLiveUs1'
    )

    # TorchLiveUs2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='TorchLiveUs',
        sampleName='TorchLiveUs2',
        thumbnailFileName=os.path.join(iconsPath, 'TorchLiveUs2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='TorchLiveUs2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='TorchLiveUs2'
    )


#
# TorchLiveUsWidget
#

class TorchLiveUsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/TorchLiveUs.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = TorchLiveUsLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in either the MRML scene
        # (in the selected parameter node), or in the application settings (independent of the scene).

        self.ui.modelPathLineEdit.connect("currentPathChanged(QString)", self.updateSettingsFromGUI)
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.verticalFlipCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.scanConversionPathLineEdit.connect("currentPathChanged(QString)", self.updateSettingsFromGUI)

        # Buttons

        self.ui.applyButton.connect('toggled(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)

        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

        lastModelPath = self.logic.getLastModelPath()
        if lastModelPath is not None:
            self.ui.modelPathLineEdit.currentPath = lastModelPath
            self.logic.loadModel(lastModelPath)

        lastScanConversionPath = slicer.util.settingsValue(self.logic.LAST_SCANCONVERSION_PATH_SETTING, None)
        if lastScanConversionPath is not None:
            self.ui.scanConversionPathLineEdit.currentPath = lastScanConversionPath

        self.updateSettingsFromGUI()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.INPUT_IMAGE))
        self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.OUTPUT_IMAGE))
        self.ui.outputTransformSelector.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.OUTPUT_TRANSFORM))

        self.ui.applyButton.checked = (self._parameterNode.GetParameter(self.logic.PREDICTION_ACTIVE).lower() == "true")

        self.ui.verticalFlipCheckBox.checked = (self._parameterNode.GetParameter(self.logic.VERTICAL_FLIP).lower() == "true")

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID(self.logic.INPUT_IMAGE, self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID(self.logic.OUTPUT_IMAGE, self.ui.outputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID(self.logic.OUTPUT_TRANSFORM, self.ui.outputTransformSelector.currentNodeID)
        self._parameterNode.SetParameter(self.logic.VERTICAL_FLIP, "true" if self.ui.verticalFlipCheckBox.checked else "false")

        self._parameterNode.EndModify(wasModified)

    def updateSettingsFromGUI(self, caller=None, event=None):

        settings = qt.QSettings()

        # Update model path and load model if changed

        modelPath = self.ui.modelPathLineEdit.currentPath
        if modelPath is None or modelPath == "":
            settings.setValue(self.logic.LAST_AI_MODEL_PATH_SETTING, "")
        else:
            if modelPath != slicer.util.settingsValue(self.logic.LAST_AI_MODEL_PATH_SETTING, None):
                settings.setValue(self.logic.LAST_AI_MODEL_PATH_SETTING, modelPath)
                self.logic.loadModel(modelPath)

        # Update scan conversion file path if changed

        scanConversionPath = self.ui.scanConversionPathLineEdit.currentPath
        if scanConversionPath is None or scanConversionPath == "":
            settings.setValue(self.logic.LAST_SCANCONVERSION_PATH_SETTING, "")
        else:
            if scanConversionPath != slicer.util.settingsValue(self.logic.LAST_SCANCONVERSION_PATH_SETTING, None):
                settings.setValue(self.logic.LAST_SCANCONVERSION_PATH_SETTING, scanConversionPath)
                self.logic.loadScanConversion(scanConversionPath)
                scanConversionDict = self.logic.scanConversionDict
                if scanConversionDict is not None:
                    self.ui.statusLabel.text = "Scan conversion loaded"
                    logging.info(f"Scan conversion loaded from {scanConversionPath}")
                    logging.info(f"Scan conversion: {scanConversionDict}")

    def onApplyButton(self, toggled):
        """
        Run processing when user clicks "Apply" button.
        """

        if self._parameterNode.GetNodeReference(self.logic.INPUT_IMAGE) is None:
            self.ui.statusLabel.text = "Input volume is required"
            self.ui.applyButton.checked = False
            return

        if self._parameterNode.GetNodeReference(self.logic.OUTPUT_IMAGE) is None:
            self.ui.statusLabel.text = "Output volume is required"
            self.ui.applyButton.checked = False
            return

        modelPath = slicer.util.settingsValue(self.logic.LAST_AI_MODEL_PATH_SETTING, None)
        if modelPath is None or modelPath == "":
            self.ui.statusLabel.text = "Model path is required"
            self.ui.applyButton.checked = False
            return

        try:
            if toggled:
                self.ui.inputSelector.enabled = False
                self.ui.outputSelector.enabled = False
                self.ui.modelPathLineEdit.enabled = False
                self.ui.scanConversionPathLineEdit.enabled = False
                self.ui.applyButton.text = "Stop processing"
                self.ui.statusLabel.text = "Running"
            else:
                self.ui.inputSelector.enabled = True
                self.ui.outputSelector.enabled = True
                self.ui.modelPathLineEdit.enabled = True
                self.ui.scanConversionPathLineEdit.enabled = True
                self.ui.applyButton.text = "Start processing"
                self.ui.statusLabel.text = "Stopped"

            self.ui.statusLabel.text = self.logic.togglePrediction(toggled)

        except Exception as e:
            slicer.util.errorDisplay("Failed to start segmentation: "+str(e))
            import traceback
            traceback.print_exc()



#
# TorchLiveUsLogic
#

class TorchLiveUsLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    # Parameter and reference names

    MODEL_PATH = "AiModelPath"
    INPUT_IMAGE = "InputImage"
    OUTPUT_IMAGE = "OutputImage"
    OUTPUT_TRANSFORM = "OutputTransform"
    PREDICTION_ACTIVE = "PredictionActive"
    VERTICAL_FLIP = "FlipVertical"

    # Settings

    LAST_AI_MODEL_PATH_SETTING = "TorchLiveUs/LastModelPath"
    LAST_SCANCONVERSION_PATH_SETTING = "TorchLiveUs/LastScanConvertPath"


    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        VTKObservationMixin.__init__(self)

        self.model = None
        self.scanConversionDict = None
        self.x_cart = None
        self.y_cart = None
        self.input_image_size = None
        self.grid_x, self.grid_y = None, None
        self.vertices, self.weights = None, None
        self.curvilinear_mask = None

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def getLastModelPath(self):
        return slicer.util.settingsValue(self.LAST_AI_MODEL_PATH_SETTING, None)

    def loadModel(self, modelPath):
        """
        Load PyTorch model from file.
        :param modelPath: path to the model file
        :return: None
        """
        if modelPath is None or modelPath == "":
            logging.warning("Model path is empty")
            self.model = None
        elif not os.path.isfile(modelPath):
            logging.error("Model file does not exist: "+modelPath)
            self.model = None
        else:
            self.model = torch.jit.load(modelPath).to(DEVICE)
            logging.info(f"Model loaded from {modelPath}")

    def loadScanConversion(self, scanConversionPath):
        if scanConversionPath is None or scanConversionPath == "":
            logging.warning("Scan conversion path is empty")
            self.scanConversionDict = None
            self.x_cart = None
            self.y_cart = None
        elif not os.path.isfile(scanConversionPath):
            logging.error("Scan conversion file does not exist: "+scanConversionPath)
            self.scanConversionDict = None
            self.x_cart = None
            self.y_cart = None
        else:
            with open(scanConversionPath, "r") as f:
                self.scanConversionDict = yaml.safe_load(f)

        if self.scanConversionDict is not None:
            initial_radius = np.deg2rad(self.scanConversionDict["angle_min_degrees"])
            final_radius = np.deg2rad(self.scanConversionDict["angle_max_degrees"])
            self.input_image_size = self.scanConversionDict["curvilinear_image_size"]
            radius_start_px = self.scanConversionDict["radius_start_pixels"]
            radius_end_px = self.scanConversionDict["radius_end_pixels"]
            num_samples_along_lines = self.scanConversionDict["num_samples_along_lines"]
            num_lines = self.scanConversionDict["num_lines"]
            center_coordinate_pixel = self.scanConversionDict["center_coordinate_pixel"]

            theta, r = np.meshgrid(np.linspace(initial_radius, final_radius, num_samples_along_lines),
                                   np.linspace(radius_start_px, radius_end_px, num_lines))

            # Precompute mapping parameters between scan converted and curvilinear images

            self.x_cart = r * np.cos(theta) + center_coordinate_pixel[0]
            self.y_cart = r * np.sin(theta) + center_coordinate_pixel[1]

            self.grid_x, self.grid_y = np.mgrid[0:self.input_image_size, 0:self.input_image_size]

            triangulation = Delaunay(np.vstack((self.x_cart.flatten(), self.y_cart.flatten())).T)

            simplices = triangulation.find_simplex(np.vstack((self.grid_x.flatten(), self.grid_y.flatten())).T)
            self.vertices = triangulation.simplices[simplices]

            X = triangulation.transform[simplices, :2]
            Y = np.vstack((self.grid_x.flatten(), self.grid_y.flatten())).T - triangulation.transform[simplices, 2]
            b = np.einsum('ijk,ik->ij', X, Y)
            self.weights = np.c_[b, 1 - b.sum(axis=1)]

            # Compute curvilinear mask, one pixel tighter to avoid artifacts

            angle1 = 90.0 + (self.scanConversionDict["angle_min_degrees"] + 1)
            angle2 = 90.0 + (self.scanConversionDict["angle_max_degrees"] - 1)

            self.curvilinear_mask = np.zeros((self.input_image_size, self.input_image_size), dtype=np.int8)
            self.curvilinear_mask = cv2.ellipse(self.curvilinear_mask,
                                                (center_coordinate_pixel[1], center_coordinate_pixel[0]),
                                                (radius_end_px - 1, radius_end_px - 1), 0.0, angle1, angle2, 1, -1)
            self.curvilinear_mask = cv2.circle(self.curvilinear_mask,
                                               (center_coordinate_pixel[1], center_coordinate_pixel[0]),
                                               radius_start_px + 1, 0, -1)

    def scanConvert(self, linear_array):
        z = linear_array.flatten()
        zi = np.einsum('ij,ij->i', np.take(z, self.vertices), self.weights)
        return zi.reshape(self.input_image_size, self.input_image_size)

    def togglePrediction(self, toggled):
        """
        Start or stop prediction.
        """
        if self.model is None:
            logging.error("Model is not loaded")
            return "Model is not loaded"

        parameterNode = self.getParameterNode()

        inputVolume = parameterNode.GetNodeReference(self.INPUT_IMAGE)
        if inputVolume is None:
            logging.error("Input volume is not selected")
            return "Input volume is not selected"

        if toggled:
            # Start observing input volume
            parameterNode.SetParameter(self.PREDICTION_ACTIVE, "true")
            inputVolume.AddObserver(slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent, self.onInputVolumeModified)
            self.onInputVolumeModified(inputVolume, None)
        else:
            # Stop observing input volume
            parameterNode.SetParameter(self.PREDICTION_ACTIVE, "false")
            inputVolume.RemoveObservers(slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent)


    def onInputVolumeModified(self, inputVolume, event):
        """
        Called when input volume is modified.
        """
        if self.model is None:
            logging.error("Model is not loaded")
            return

        parameterNode = self.getParameterNode()

        if parameterNode.GetParameter(self.PREDICTION_ACTIVE) != "true":
            logging.debug("Prediction is not active")
            return

        if inputVolume.GetImageData() is None:
            logging.debug("Input volume has no image data")
            return

        input_array = slicer.util.array(inputVolume.GetID())

        if parameterNode.GetParameter(self.VERTICAL_FLIP) == "true":
            input_array = np.flip(input_array, axis=0)

        # If scan conversion given, map image accordingly.
        # Otherwise, resize input using opencv with linear interpolation to match model input size

        if self.scanConversionDict is not None:
            resized_array = map_coordinates(input_array[0, :, :], [self.x_cart, self.y_cart], order=1, mode='constant', cval=0.0)
        else:
            resized_array = cv2.resize(input_array[0, :, :], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor and add batch dimension

        input_tensor = torch.from_numpy(resized_array).unsqueeze(0).unsqueeze(0).float()

        # Run inference

        with torch.inference_mode():
            output_logits = self.model(input_tensor.to(DEVICE))
        if isinstance(output_logits, list):
            output_logits = output_logits[0]

        output_tensor = torch.softmax(output_logits, dim=1)

        # If scan conversion given, map image accordingly. Otherwise, resize output to match input size

        if self.scanConversionDict is not None:
            # resized_output_array = griddata((self.x_cart.flatten(), self.y_cart.flatten()), output_array[1, :, :].flatten(),
            #                         (self.grid_x, self.grid_y), method="linear", fill_value=0)
            output_array = output_tensor.detach().cpu().numpy() * 255
            resized_output_array = self.scanConvert(output_array[0, 1, :, :])
            resized_output_array = resized_output_array * self.curvilinear_mask
        else:
            output_array = output_tensor.squeeze().detach().cpu().numpy() * 255
            resized_output_array = cv2.resize(output_array[1], (input_array.shape[2], input_array.shape[1]), interpolation=cv2.INTER_LINEAR)

        # Set output volume image data

        outputVolume = parameterNode.GetNodeReference(self.OUTPUT_IMAGE)
        slicer.util.updateVolumeFromArray(outputVolume, resized_output_array.astype(np.uint8)[np.newaxis, ...])



    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# TorchLiveUsTest
#

class TorchLiveUsTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_TorchLiveUs1()

    def test_TorchLiveUs1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('TorchLiveUs1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = TorchLiveUsLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
