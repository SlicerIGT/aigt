import os
import unittest
import logging
import numpy as np
import scipy.ndimage
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from Processes import Process, ProcessesLogic
import pickle

import sys
if not hasattr(sys, 'argv'):
  sys.argv  = ['']
import tensorflow as tf

#
# SegmentationUNet
#

class SegmentationUNet(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "SegmentationUNet"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Ultrasound"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""  # TODO: update with short description of the module
    self.parent.helpText += self.getDefaultModuleDocumentationLink()  # TODO: verify that the default URL is correct or change it to the actual documentation
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""  # TODO: replace with organization, grant and thanks.

#
# SegmentationUNetWidget
#

class SegmentationUNetWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None

    self.inputModifiedObserverTag = None

    self.apply_logarithmic_transformation = True
    self.logarithmic_transformation_decimals = 4

    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer)
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/SegmentationUNet.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create a new parameterNode (it stores user's node and parameter values choices in the scene)
    self.logic = SegmentationUNetLogic()

    self.ui.parameterNodeSelector.addAttribute("vtkMRMLScriptedModuleNode", "ModuleName", self.moduleName)
    self.setParameterNode(self.logic.getParameterNode())

    # Connections
    self.ui.parameterNodeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.setParameterNode)

    self.ui.modelPathLineEdit.connect("currentPathChanged(QString)", self.onModelSelected)
    lastModelPath = self.logic.getLastModelPath()
    if lastModelPath is not None:
      self.ui.modelPathLineEdit.setCurrentPath(lastModelPath)
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputImageSelected)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputImageSelected)
    self.ui.processCheckBox.connect("toggled(bool)", self.onProcessToggled)

    self.ui.applyButton.connect('toggled(bool)', self.onApplyButton)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def enter(self):
    slicer.util.setApplicationLogoVisible(False)
    slicer.util.setDataProbeVisible(False)

  def exit(self):
    slicer.util.setDataProbeVisible(True)

  def onInputImageSelected(self, selectedNode):
    self.logic.setInputImage(selectedNode)

  def onOutputImageSelected(self, selectedNode):
    self.logic.setOutputImage(selectedNode)

  def onProcessToggled(self, checked):
    self._parameterNode.SetParameter(self.logic.USE_SEPARATE_PROCESS, "True" if checked else "False")

  def onModelSelected(self, modelFullname):
    self.logic.setModelPath(modelFullname)

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def setParameterNode(self, inputParameterNode):
    """
    Adds observers to the selected parameter node. Observation is needed because when the
    parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)
    wasBlocked = self.ui.parameterNodeSelector.blockSignals(True)
    self.ui.parameterNodeSelector.setCurrentNode(inputParameterNode)
    self.ui.parameterNodeSelector.blockSignals(wasBlocked)
    if inputParameterNode == self._parameterNode:
      return
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    if inputParameterNode is not None:
      self.addObserver(inputParameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    # Disable all sections if no parameter node is selected

    self.ui.basicCollapsibleButton.enabled = self._parameterNode is not None

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    self._updatingGUIFromParameterNode = True

    # Update each widget from parameter node

    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.INPUT_IMAGE))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.OUTPUT_IMAGE))
    self.ui.processCheckBox.checked = (self._parameterNode.GetParameter(self.logic.USE_SEPARATE_PROCESS).lower() == "true")

    # Update buttons states and tooltips

    if self._parameterNode.GetNodeReference(self.logic.INPUT_IMAGE) and self._parameterNode.GetNodeReference("OutputVolume"):
      self.ui.applyButton.toolTip = "Compute output volume"
    else:
      self.ui.applyButton.toolTip = "Select input and output volume nodes"

    self._updatingGUIFromParameterNode = False

  def onApplyButton(self, toggled):
    """
    Run processing when user clicks "Apply" button.
    """
    self.inputImageNode = self.ui.inputSelector.currentNode()
    if self.inputImageNode is None:
      self.ui.feedbackLabel.text = "Input image not selected!"
      logging.info("Apply button clicked without input selection")
      return
    else:
      logging.info("Input image: {}".format(self.inputImageNode.GetName()))

    self.outputImageNode = self.ui.outputSelector.currentNode()
    if self.outputImageNode is None:
      self.ui.feedbackLabel.text = "Output image not selected!"
      logging.info("Apply button clicked without output selection")
      return
    else:
      logging.info("Output image: {}".format(self.outputImageNode.GetName()))

    if self.logic.unet_model is None:
      self.ui.feedbackLabel.text = "UNet model not selected!"
      logging.info("Apply button clicked without UNet model selected")
      return
    else:
      logging.info("Using UNet")

    try:
      if toggled == True:
        logging.info("Staring live segmentation")
        useProcess = self.logic.getUseProcess()
        if useProcess:
          logging.info("Starting separate process for prediction")
          self.logic.setupProcess()
        self.ui.processCheckBox.enabled = False
        # self.inputModifiedObserverTag = self.inputImageNode.AddObserver(slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent,
        #                                                                 self.logic.onInputNodeModified)

      else:
        logging.info("Stopping live segmentation")
        if self.inputModifiedObserverTag is not None:
          self.inputImageNode.RemoveObserver(self.inputModifiedObserverTag)
          self.inputModifiedObserverTag = None
        self.logic.stopLiveSegmentation()
        self.ui.processCheckBox.enabled = True

    except Exception as e:
      slicer.util.errorDisplay("Failed to start live segmentation: "+str(e))
      import traceback
      traceback.print_exc()


class LivePredictionProcess(Process):

  def __init__(self, scriptPath, volume, model_path, active=True):
    Process.__init__(self, scriptPath)
    self.volume = volume  # Numpy array, to use as input for the model.
    self.model_path = model_path  # Path to the TF model you'd like to load, as TF Models are not picklable.
    self.active = bytes([1]) if active else bytes([0])  # Used to stop the process by enabling/disabling the script.
    self.name = f"LivePrediction-{os.path.basename(model_path)}"
    self.output = None

    logging.info("  Process script:   {}".format(scriptPath))
    logging.info("  AI model file:    {}".format(model_path))
    logging.info("  Input dimensions: {}".format(volume.shape))

  def setActive(self, active=True):
    self.active = bytes([1]) if active else bytes([0])

  def onStarted(self):
    logging.info("LivePredictionProcess.onStarted")
    input = self.prepareProcessInput()
    input_len = len(input).to_bytes(8, byteorder='big')
    self.write(self.active)  # Write if the predictions are still running
    self.write(input_len)  # Write the length of the input we have yet to recieve in the buffer
    self.write(input)  # Write the pickled inputs for the model

  def prepareProcessInput(self):
    logging.info("LivePredictionProcess.prepareProcessInput")
    input = {}
    input['model_path'] = self.model_path
    input['volume'] = self.volume
    return pickle.dumps(input)

  def useProcessOutput(self, processOutput):
    logging.info("LivePredictionProcess.useProcessOutput")
    try:
      output = pickle.loads(processOutput)
      logging.info("  output shape: {}".format(output['prediction'].shape))
      self.output = output
    except EOFError:
      logging.info("  EOF error")
      self.output = None



#
# SegmentationUNetLogic
#

class SegmentationUNetLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  INPUT_IMAGE = "InputImage"
  OUTPUT_IMAGE = "OutputImage"
  AI_MODEL_FULLPATH = "AiModelFullpath"
  LAST_AI_MODEL_PATH_SETTING = "SegmentationUNet/LastAiModelPath"
  USE_SEPARATE_PROCESS = "UseSeparateProcess"

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    VTKObservationMixin.__init__(self)

    self.slicer_to_model_scaling_x = 1.0
    self.slicer_to_model_scaling_y = 1.0
    self.model_to_slicer_scaling_x = 1.0
    self.model_to_slicer_scaling_y = 1.0

    self.unet_model = None

    self.livePredictionProcess = None

  def setDefaultParameters(self, parameterNode):
    if not parameterNode.GetParameter("Threshold"):
      parameterNode.SetParameter("Threshold", "50.0")
    if not parameterNode.GetParameter("Invert"):
      parameterNode.SetParameter("Invert", "false")
    if not parameterNode.GetParameter(self.USE_SEPARATE_PROCESS):
      parameterNode.SetParameter(self.USE_SEPARATE_PROCESS, "True")

  def setInputImage(self, inputImageNode):
    """
    Sets input image node
    :param inputImageNode: vtkMRMLScalarVolumeNode
    :return: None
    """
    parameterNode = self.getParameterNode()
    if inputImageNode is None:
      parameterNode.SetNodeReferenceID(self.INPUT_IMAGE, None)
      return

    parameterNode.SetNodeReferenceID(self.INPUT_IMAGE, inputImageNode.GetID())
    input_array = slicer.util.array(inputImageNode.GetID())
    self.slicer_to_model_scaling_x = self.unet_model.layers[0].input_shape[0][1] / input_array.shape[1]
    self.slicer_to_model_scaling_y = self.unet_model.layers[0].input_shape[0][2] / input_array.shape[2]
    self.model_to_slicer_scaling_x = input_array.shape[1] / self.unet_model.layers[0].input_shape[0][1]
    self.model_to_slicer_scaling_y = input_array.shape[2] / self.unet_model.layers[0].input_shape[0][2]

  def setOutputImage(self, outputImageNode):
    """
    Sets output image node
    :param outputImageNode: vtkMRMLScalarVolumeNode
    :return: None
    """
    paramterNode = self.getParameterNode()
    if outputImageNode is None:
      paramterNode.SetNodeReferenceID(self.OUTPUT_IMAGE, None)
      return

    paramterNode.SetNodeReferenceID(self.OUTPUT_IMAGE, outputImageNode.GetID())

  def setModelPath(self, modelFullpath):
    """
    Sets the AI model file full path
    :param modelFullpath: str
    :return: None
    """
    parameterNode = self.getParameterNode()

    if modelFullpath == "" or modelFullpath is None:
      parameterNode.SetParameter(self.AI_MODEL_FULLPATH, "")
      return

    parameterNode.SetParameter(self.AI_MODEL_FULLPATH, modelFullpath)

    try:
      self.unet_model = tf.keras.models.load_model(modelFullpath, compile=False)
      self.unet_model.call = tf.function(self.unet_model.call)
      logging.info("Model loaded from file: {}".format(modelFullpath))
      settings = qt.QSettings()
      settings.setValue(self.LAST_AI_MODEL_PATH_SETTING, modelFullpath)
    except Exception as e:
      logging.error("Could not load model from file: {}".format(modelFullpath))
      logging.error("Exception: {}".format(str(e)))


  def setupProcess(self):
    parameterNode = self.getParameterNode()
    scriptFolder = slicer.modules.segmentationunet.path.replace("SegmentationUNet.py", "")
    scriptPath = os.path.join(scriptFolder, "Resources", "ProcessScripts")

    inputImageNode = parameterNode.GetNodeReference(self.INPUT_IMAGE)
    if inputImageNode is None:
      logging.error("Cannot start segmentation process with no input image specified")
      return
    imageArray = np.squeeze(slicer.util.array(inputImageNode.GetID()))
    modelPath = parameterNode.GetParameter(self.AI_MODEL_FULLPATH)

    self.stopLiveSegmentation()
    self.livePredictionProcess = LivePredictionProcess(scriptPath, imageArray, modelPath)
    self.livePredictionProcess.connect('readyReadStandardOutput()', self.imageToPredictReady)

    def onLivePredictProcessCompleted():
      logging.info('Live Prediction: Process Finished')
      self.livePredictionProcess.disconnect('readyReadStandardOutput()', self.imageToPredictReady)

    logic = ProcessesLogic(completedCallback=lambda: onLivePredictProcessCompleted())
    logic.addProcess(self.livePredictionProcess)
    logic.run()
    logging.info('Live Prediction: Process Started')

  def stopLiveSegmentation(self):
    """
    Stops real-time segmentation, if it is active.
    :return: None
    """
    if self.livePredictionProcess is not None:
      self.livePredictionProcess.setActive(False)
      self.livePredictionProcess.close()
      self.livePredictionProcess = None

  def imageToPredictReady(self):
    """
    Callback to receive prediction and update volume in Slicer.
    :return: None
    """
    logging.info("imageToPredictReady()")
    parameterNode = self.getParameterNode()
    stdout = self.livePredictionProcess.readAllStandardOutput().data()
    self.livePredictionProcess.useProcessOutput(stdout)
    output_array = self.livePredictionProcess.output['prediction']
    predictionVolumeNode = parameterNode.GetNodeReference(self.OUTPUT_IMAGE)
    slicer.util.updateVolumeFromArray(predictionVolumeNode, output_array.astype(np.uint8)[np.newaxis, ...])

  def getUseProcess(self):
    parameterNode = self.getParameterNode()
    useProcessStr = parameterNode.GetParameter(self.USE_SEPARATE_PROCESS)
    if useProcessStr is None or useProcessStr.lower() != "true":
      return False
    else:
      return True

  def getLastModelPath(self):
    return slicer.util.settingsValue(self.LAST_AI_MODEL_PATH_SETTING, None)

  def onInputNodeModified(self, caller, event):
    """
    Callback function for input image modified event.
    :returns: None
    """
    useProcesses = self.getUseProcess()
    if useProcesses:
      self.updatePreditionOnProcess()
    else:
      self.updatePrecitionOnMain()

  def updatePreditionOnProcess(self):
    parameterNode = self.getParameterNode()
    inputImageNode = parameterNode.GetNodeReference(self.INPUT_IMAGE)
    input_array = slicer.util.array(inputImageNode.GetID())
    self.livePredictionProcess.volume = input_array
    self.livePredictionProcess.onStarted()

  def updatePrecitionOnMain(self):
    parameterNode = self.getParameterNode()
    inputImageNode = parameterNode.GetNodeReference(self.INPUT_IMAGE)
    input_array = slicer.util.array(inputImageNode.GetID())

    resized_input_array = scipy.ndimage.zoom(input_array[0, :, :],
                                             (self.slicer_to_model_scaling_x, self.slicer_to_model_scaling_y),
                                             prefilter=False, order=1)
    resized_input_array = np.flip(resized_input_array, axis=1)
    # resized_input_array = resized_input_array / resized_input_array.max()  # Scaling intensity to 0-1
    resized_input_array = resized_input_array / 256.0
    resized_input_array = np.expand_dims(resized_input_array, axis=0)
    resized_input_array = np.expand_dims(resized_input_array, axis=3)

    y = self.unet_model.predict(resized_input_array)
    y = y[:, :, :, 1]
    if self.apply_logarithmic_transformation:
      e = self.logarithmic_transformation_decimals
      y = np.log10(np.clip(y, 10 ** (-e), 1.0) * (10 ** e)) / e
    y = np.flip(y, axis=2)
    y = y * 255

    upscaled_output_array = scipy.ndimage.zoom(y[0, :, :],
                                               (self.model_to_slicer_scaling_x, self.model_to_slicer_scaling_y),
                                               prefilter=False, order=1)
    upscaled_output_array = np.clip(upscaled_output_array, 0, 255).astype(np.uint8)

    slicer.util.updateVolumeFromArray(self.outputImageNode, upscaled_output_array.astype(np.uint8)[np.newaxis, ...])


#
# SegmentationUNetTest
#

class SegmentationUNetTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_SegmentationUNet1()

  def test_SegmentationUNet1(self):
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
    inputVolume = SampleData.downloadFromURL(
      nodeNames='MRHead',
      fileNames='MR-Head.nrrd',
      uris='https://github.com/Slicer/SlicerTestingData/releases/download/MD5/39b01631b7b38232a220007230624c8e',
      checksums='MD5:39b01631b7b38232a220007230624c8e')[0]
    self.delayDisplay('Finished with download and loading')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 279)

    self.delayDisplay('Test passed')


