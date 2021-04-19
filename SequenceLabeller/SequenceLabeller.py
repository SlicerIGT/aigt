import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

#
# SequenceLabeller
#

class SequenceLabeller(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Sequence Labeller"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#SequenceLabeller">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

#
# SequenceLabellerWidget
#

class SequenceLabellerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
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
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/SequenceLabeller.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = SequenceLabellerLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.segInputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.segOutputSequenceBrowserSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.segSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.segSequenceBrowserSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    self.ui.markupSequenceBrowserSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.markupMarkupSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.markupVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.markupsOutputLabelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.markupOutputSequenceBrowserSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.segApply.connect('clicked(bool)', self.onSegApplyButton)
    self.ui.markupApply.connect('clicked(bool)', self.onMarkupApply)

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

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
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
    self.ui.segSequenceBrowserSelector.setCurrentNode(self._parameterNode.GetNodeReference("SegInputSequenceBrowser"))
    self.ui.segSegmentationSelector.setCurrentNode(self._parameterNode.GetNodeReference("SegInputSegmentation"))
    self.ui.segInputVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("SegInputVolume"))
    self.ui.segOutputSequenceBrowserSelector.setCurrentNode(self._parameterNode.GetNodeReference("SegOutputSequenceBrowser"))

    self.ui.markupSequenceBrowserSelector.setCurrentNode(self._parameterNode.GetNodeReference("MarkupInputSequenceBrowser"))
    self.ui.markupMarkupSelector.setCurrentNode(self._parameterNode.GetNodeReference("MarkupInputMarkup"))
    self.ui.markupVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("MarkupInputVolume"))
    self.ui.markupsOutputLabelSelector.setCurrentNode(self._parameterNode.GetNodeReference("MarkupOutputLabel"))
    self.ui.markupOutputSequenceBrowserSelector.setCurrentNode(self._parameterNode.GetNodeReference("MarkupOutputSequenceBrowser"))

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

    self._parameterNode.SetNodeReferenceID("SegInputSequenceBrowser", self.ui.segSequenceBrowserSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("SegInputSegmentation", self.ui.segSegmentationSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("SegInputVolume", self.ui.segInputVolumeSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("SegOutputSequenceBrowser", self.ui.segOutputSequenceBrowserSelector.currentNodeID)

    self._parameterNode.SetNodeReferenceID("MarkupInputSequenceBrowser", self.ui.markupSequenceBrowserSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("MarkupInputMarkup", self.ui.markupMarkupSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("MarkupInputVolume", self.ui.markupVolumeSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("MarkupOutputLabel", self.ui.markupsOutputLabelSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("MarkupOutputSequenceBrowser", self.ui.markupOutputSequenceBrowserSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)

  def onSegApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:
      slicer.app.pauseRender()
      sequenceBrowser = self._parameterNode.GetNodeReference("SegInputSequenceBrowser")
      inputSegmentation = self._parameterNode.GetNodeReference("SegInputSegmentation")
      inputVolume = self._parameterNode.GetNodeReference("SegInputVolume")
      outputSequenceBrowser = self._parameterNode.GetNodeReference("SegOutputSequenceBrowser")
      self.logic.applySegmentationLabel(sequenceBrowser, inputSegmentation, inputVolume, outputSequenceBrowser)

    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()
    finally:
      slicer.app.resumeRender()
      pass

  def onMarkupApply(self):
    try:
      slicer.app.pauseRender()
      markupSequenceBrowser = self._parameterNode.GetNodeReference("MarkupInputSequenceBrowser")
      markupMarkup = self._parameterNode.GetNodeReference("MarkupInputMarkup")
      markupVolume = self._parameterNode.GetNodeReference("MarkupInputVolume")
      markupsOutputLabel = self._parameterNode.GetNodeReference("MarkupOutputLabel")
      markupOutputSequenceBrowser = self._parameterNode.GetNodeReference("MarkupOutputSequenceBrowser")
      self.logic.applyMarkupsLabel(markupSequenceBrowser, markupMarkup, markupVolume, markupsOutputLabel, markupOutputSequenceBrowser)

    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()
    finally:
      slicer.app.resumeRender()
      pass

    

#
# SequenceLabellerLogic
#

class SequenceLabellerLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def generateMergedLabelmapInReferenceGeometry(self, segmentationNode, referenceVolumeNode):
    if segmentationNode is None:
        logging.error("Invalid segmentation node")
        return None
    if referenceVolumeNode is None:
        logging.error("Invalid reference volume node")
        return None
  
    # Get reference geometry in the segmentation node's coordinate system
    referenceGeometry_Reference = slicer.vtkOrientedImageData() # reference geometry in reference node coordinate system
    referenceGeometry_Segmentation = slicer.vtkOrientedImageData()
    mergedLabelmap_Reference = slicer.vtkOrientedImageData()
    referenceGeometryToSegmentationTransform = vtk.vtkGeneralTransform()
  
    # Set reference image geometry
    referenceGeometry_Reference.SetExtent(referenceVolumeNode.GetImageData().GetExtent())
    ijkToRasMatrix = vtk.vtkMatrix4x4()
    referenceVolumeNode.GetIJKToRASMatrix(ijkToRasMatrix)
    referenceGeometry_Reference.SetGeometryFromImageToWorldMatrix(ijkToRasMatrix)
  
    # Transform it to the segmentation node coordinate system
    referenceGeometry_Segmentation = slicer.vtkOrientedImageData()
    referenceGeometry_Segmentation.DeepCopy(referenceGeometry_Reference)
  
    # Get transform between reference volume and segmentation node
    if (referenceVolumeNode.GetParentTransformNode() != segmentationNode.GetParentTransformNode()):
      slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(referenceVolumeNode.GetParentTransformNode(),
        segmentationNode.GetParentTransformNode(), referenceGeometryToSegmentationTransform)
      slicer.vtkOrientedImageDataResample.TransformOrientedImage(referenceGeometry_Segmentation, referenceGeometryToSegmentationTransform, True)
  
    # Generate shared labelmap for the exported segments in segmentation coordinates
    sharedImage_Segmentation = slicer.vtkOrientedImageData()
    if (not segmentationNode.GenerateMergedLabelmapForAllSegments(sharedImage_Segmentation, 0, None)):
      logging.error("ExportSegmentsToLabelmapNode: Failed to generate shared labelmap")
      return None
  
    # Transform shared labelmap to reference geometry coordinate system
    segmentationToReferenceGeometryTransform = referenceGeometryToSegmentationTransform.GetInverse()
    segmentationToReferenceGeometryTransform.Update()
    slicer.vtkOrientedImageDataResample.ResampleOrientedImageToReferenceOrientedImage(sharedImage_Segmentation, referenceGeometry_Reference, mergedLabelmap_Reference,
      False, False, segmentationToReferenceGeometryTransform)
  
    return mergedLabelmap_Reference

  def applySegmentationLabel(self, sequenceBrowser_node, segmentation_node, image_node, output_sequenceBrowser_node):
    new_volume_sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", image_node.GetName() + "_ImageSequence")
    output_sequenceBrowser_node.AddSynchronizedSequenceNode(new_volume_sequence_node)

    labelmap_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", image_node.GetName() + "_LabelledSequence")
    labelmap_sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", image_node.GetName() + "_Label")
    output_sequenceBrowser_node.AddSynchronizedSequenceNode(labelmap_sequence_node)

    masterSequence = sequenceBrowser_node.GetMasterSequenceNode()
    for i in range(min(100, sequenceBrowser_node.GetNumberOfItems())):
      sequenceBrowser_node.SetSelectedItemNumber(i)

      slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentation_node, labelmap_volume_node)
      labelmapOriented_Reference = self.generateMergedLabelmapInReferenceGeometry(segmentation_node, image_node)
      slicer.modules.segmentations.logic().CreateLabelmapVolumeFromOrientedImageData(labelmapOriented_Reference, labelmap_volume_node)

      sequenceValue = masterSequence.GetNthIndexValue(i)
      new_volume_sequence_node.SetDataNodeAtValue(image_node, sequenceValue)
      labelmap_sequence_node.SetDataNodeAtValue(labelmap_volume_node, sequenceValue)


    output_sequenceBrowser_node.AddProxyNode(image_node, new_volume_sequence_node, False)
    output_sequenceBrowser_node.AddProxyNode(labelmap_volume_node, labelmap_sequence_node, False)

    currentTransformableNode = labelmap_volume_node
    currentTransformNode = image_node.GetParentTransformNode()
    while not currentTransformNode is None:
      currentTransformableNode.SetAndObserveTransformNodeID(image_node.GetParentTransformNode().GetID())
      sequenceNode = sequenceBrowser_node.GetSequenceNode(currentTransformNode)
      if sequenceNode:
        output_sequenceBrowser_node.AddSynchronizedSequenceNode(sequenceNode)
        output_sequenceBrowser_node.AddProxyNode(currentTransformNode, sequenceNode, False)
      currentTransformNode = currentTransformNode.GetParentTransformNode()

  def applyMarkupsLabel(self, input_sequence_browser_node, input_markup_node, input_volume_node, output_text_node, output_sequence_browser_node):

    text_sequence_node = output_sequence_browser_node.GetSequenceNode(output_text_node)
    if text_sequence_node is None:
      text_sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", input_volume_node.GetName() + "_LabelSequence")
      output_sequence_browser_node.AddSynchronizedSequenceNode(text_sequence_node)
      output_sequence_browser_node.AddProxyNode(output_text_node, text_sequence_node, False)
    text_sequence_node.RemoveAllDataNodes()

    image_sequence_node = output_sequence_browser_node.GetSequenceNode(input_volume_node)
    if image_sequence_node is None:
      image_sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", input_volume_node.GetName() + "_ImageSequence")
      output_sequence_browser_node.AddSynchronizedSequenceNode(image_sequence_node)
      output_sequence_browser_node.AddProxyNode(input_volume_node, image_sequence_node, False)
    image_sequence_node.RemoveAllDataNodes()

    masterSequence = input_sequence_browser_node.GetMasterSequenceNode()
    for i in range(min(100, input_sequence_browser_node.GetNumberOfItems())):
      input_sequence_browser_node.SetSelectedItemNumber(i)
      sequenceValue = masterSequence.GetNthIndexValue(i)

      extent = input_volume_node.GetImageData().GetExtent()

      markupToVolumeTransform = vtk.vtkGeneralTransform()
      slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(input_markup_node.GetParentTransformNode(), input_volume_node.GetParentTransformNode(), markupToVolumeTransform)

      rasToIJKMatrix = vtk.vtkMatrix4x4()
      input_volume_node.GetRASToIJKMatrix(rasToIJKMatrix)
      rasToIJKTransform = vtk.vtkTransform()
      rasToIJKTransform.SetMatrix(rasToIJKMatrix)

      markupToVolumeTransform.Concatenate(rasToIJKTransform)

      overlappingMarkupIndex = -1
      for pointIndex in range(input_markup_node.GetNumberOfControlPoints()):
        position_Markup = [0,0,0]
        input_markup_node.GetNthControlPointPosition(pointIndex, position_Markup)
        point_IJK = [0,0,0]
        markupToVolumeTransform.TransformPoint(position_Markup, point_IJK)

        threshold_mm = 10 # TODO: Make threshold markup parameter
        if (  point_IJK[0] > extent[0] - threshold_mm
          and point_IJK[0] < extent[1] + threshold_mm
          and point_IJK[1] > extent[2] - threshold_mm
          and point_IJK[1] < extent[3] + threshold_mm
          and point_IJK[2] > extent[4] - threshold_mm
          and point_IJK[2] < extent[5] + threshold_mm):
          overlappingMarkupIndex = pointIndex
          break

      labelText = "None"
      if overlappingMarkupIndex >= 0:
        labelText = input_markup_node.GetNthControlPointLabel(overlappingMarkupIndex)
        overlappingMarkupIndex = threshold_mm

      output_text_node.SetText(labelText)

      text_sequence_node.SetDataNodeAtValue(output_text_node, sequenceValue)  
      image_sequence_node.SetDataNodeAtValue(input_volume_node, sequenceValue)

    currentTransformNode = input_volume_node.GetParentTransformNode()
    while not currentTransformNode is None:
      sequenceNode = input_sequence_browser_node.GetSequenceNode(currentTransformNode)
      if sequenceNode:
        output_sequence_browser_node.AddSynchronizedSequenceNode(sequenceNode)
        output_sequence_browser_node.AddProxyNode(currentTransformNode, sequenceNode, False)
      currentTransformNode = currentTransformNode.GetParentTransformNode()

#
# SequenceLabellerTest
#

class SequenceLabellerTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_SequenceLabeller1()

  def test_SequenceLabeller1(self):
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
    self.delayDisplay('Test passed')
