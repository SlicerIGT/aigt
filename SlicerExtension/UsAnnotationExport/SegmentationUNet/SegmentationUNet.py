import os
import unittest
import logging
import numpy as np
import scipy.ndimage
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

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
    self.unet_model = None
    self.inputModifiedObserver = None
    self.inputImageNode = None
    self.outputImageNode = None
    self.slicer_to_model_scaling = 1.0
    self.model_to_slicer_scaling = 1.0

    self.apply_logarithmic_transformation = True
    self.logarithmic_transformation_decimals = 4

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
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    self.ui.applyButton.connect('toggled(bool)', self.onApplyButton)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def onModelSelected(self, modelFullname):
    try:
      self.unet_model = tf.keras.models.load_model(modelFullname, compile=False)
      logging.info("Model loaded from file: {}".format(modelFullname))
    except Exception as e:
      logging.error("Could not load model from file: {}".format(modelFullname))
      logging.error("Exception: {}".format(str(e)))


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

    if self._parameterNode is None:
      return

    # Update each widget from parameter node

    wasBlocked = self.ui.inputSelector.blockSignals(True)
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    self.ui.inputSelector.blockSignals(wasBlocked)

    wasBlocked = self.ui.outputSelector.blockSignals(True)
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    self.ui.outputSelector.blockSignals(wasBlocked)

    # Update buttons states and tooltips
    if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
      self.ui.applyButton.toolTip = "Compute output volume"
    else:
      self.ui.applyButton.toolTip = "Select input and output volume nodes"

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes changes in the GUI.
    The changes are saved into the parameter node (so that they are preserved when the scene is saved and loaded).
    """

    if self._parameterNode is None:
      return

    self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)

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

    if self.unet_model is None:
      self.ui.feedbackLabel.text = "UNet model not selected!"
      logging.info("Apply button clicked without UNet model selected")
      return
    else:
      logging.info("Using UNet")

    try:
      input_array = slicer.util.array(self.inputImageNode.GetID())
      self.slicer_to_model_scaling = self.unet_model.layers[0].input_shape[0][1] / input_array.shape[1]
      self.model_to_slicer_scaling = input_array.shape[1] / self.unet_model.layers[0].input_shape[0][1]
      if toggled == True:
        logging.info("Staring live segmentation")
        self.inputModifiedObserver = self.inputImageNode.AddObserver(
          slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent, self.onInputNodeModified)
      else:
        logging.info("Stopping live segmentation")
        if self.inputModifiedObserver is not None:
          self.inputImageNode.RemoveObserver(self.inputModifiedObserver)
          self.inputModifiedObserver = None
    except Exception as e:
      slicer.util.errorDisplay("Failed to start live segmentation: "+str(e))
      import traceback
      traceback.print_exc()

  def onInputNodeModified(self, caller, event):
    input_array = slicer.util.array(self.inputImageNode.GetID())

    resized_input_array = scipy.ndimage.zoom(input_array[0, :, :],
                                             self.slicer_to_model_scaling, prefilter=False, order=1)
    resized_input_array = np.flip(resized_input_array, axis=0)
    resized_input_array = resized_input_array / resized_input_array.max()  # Scaling intensity to 0-1
    resized_input_array = np.expand_dims(resized_input_array, axis=0)
    resized_input_array = np.expand_dims(resized_input_array, axis=3)
    y = self.unet_model.predict(resized_input_array)
    if self.apply_logarithmic_transformation:
      e = self.logarithmic_transformation_decimals
      y = np.log10(np.clip(y, 10 ** (-e), 1.0) * (10 ** e)) / e
    y[0, :, :, :] = np.flip(y[0, :, :, :], axis=0)

    upscaled_output_array = scipy.ndimage.zoom(y[0, :, :, 1], self.model_to_slicer_scaling,
                                               prefilter=False, order=1)
    upscaled_output_array = upscaled_output_array * 255
    upscaled_output_array = np.clip(upscaled_output_array, 0, 255)

    slicer.util.updateVolumeFromArray(self.outputImageNode, upscaled_output_array.astype(np.uint8)[np.newaxis, ...])

#
# SegmentationUNetLogic
#

class SegmentationUNetLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setDefaultParameters(self, parameterNode):
    if not parameterNode.GetParameter("Threshold"):
      parameterNode.SetParameter("Threshold", "50.0")
    if not parameterNode.GetParameter("Invert"):
      parameterNode.SetParameter("Invert", "false")


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
