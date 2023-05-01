import logging
import os
import numpy as np
import time
import vtk, qt, ctk, slicer

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

try:
    import torch
except:
    logging.error('TorchLiveUs module requires PyTorch to be installed')

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

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).

        self.ui.modelPathLineEdit.connect("currentPathChanged(QString)", self.updateParameterNodeFromGUI)
        lastModelPath = self.logic.getLastModelPath()
        if lastModelPath is not None:
            self.ui.modelPathLineEdit.setCurrentPath(lastModelPath)
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.verticalFlipCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)

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

        self.ui.modelPathLineEdit.setCurrentPath(self._parameterNode.GetParameter(self.logic.MODEL_PATH))
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

        # Update model path and load model if changed

        modelPath = self.ui.modelPathLineEdit.currentPath
        if modelPath is None or modelPath == "":
            self._parameterNode.SetParameter(self.logic.MODEL_PATH, "")
        else:
            if modelPath != self._parameterNode.GetParameter(self.logic.MODEL_PATH):
                self._parameterNode.SetParameter(self.logic.MODEL_PATH, modelPath)
                self.logic.loadModel(modelPath)

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self, toggled):
        """
        Run processing when user clicks "Apply" button.
        """

        if self._parameterNode.GetNodeReference(self.logic.INPUT_IMAGE) is None:
            self.ui.statusLabel.text = "Input volume is required"
            return

        if self._parameterNode.GetNodeReference(self.logic.OUTPUT_IMAGE) is None:
            self.ui.statusLabel.text = "Output volume is required"
            return

        if self._parameterNode.GetParameter(self.logic.MODEL_PATH) == "":
            self.ui.statusLabel.text = "Model path is required"
            return

        try:
            if toggled:
                self.ui.inputSelector.enabled = False
                self.ui.outputSelector.enabled = False
                self.ui.modelPathLineEdit.enabled = False
                self.ui.applyButton.text = "Stop processing"
                self.ui.statusLabel.text = "Running"
            else:
                self.ui.inputSelector.enabled = True
                self.ui.outputSelector.enabled = True
                self.ui.modelPathLineEdit.enabled = True
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


    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        VTKObservationMixin.__init__(self)

        self.model = None

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
            self.model = torch.jit.load(modelPath)
        settings = qt.QSettings()
        settings.setValue(self.LAST_AI_MODEL_PATH_SETTING, modelPath)

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

        # Resize input using opencv with linear interpolation to match model input size
        resized_array = cv2.resize(input_array[0, :, :], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor and add batch dimension
        input_tensor = torch.from_numpy(resized_array).unsqueeze(0).unsqueeze(0).float()

        # Run inference
        with torch.no_grad():
            output_logits = self.model(input_tensor)

        # Convert logits to probabilities
        output_tensor = torch.sigmoid(output_logits)

        # Convert output to numpy array
        output_array = output_tensor.squeeze().numpy() * 255

        # Resize output to match input size
        output_array = cv2.resize(output_array, (input_array.shape[2], input_array.shape[1]), interpolation=cv2.INTER_LINEAR)

        # Set output volume image data
        outputVolume = parameterNode.GetNodeReference(self.OUTPUT_IMAGE)
        slicer.util.updateVolumeFromArray(outputVolume, output_array.astype(np.uint8)[np.newaxis, ...])



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
