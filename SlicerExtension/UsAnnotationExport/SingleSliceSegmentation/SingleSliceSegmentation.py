from __future__ import print_function
import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

import numpy as np


#
# SingleSliceSegmentation
#

class SingleSliceSegmentation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Single Slice Segmentation"
    self.parent.categories = ["Ultrasound"]
    self.parent.dependencies = []
    self.parent.contributors = ["Tamas Ungi (Queen's University)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# SingleSliceSegmentationWidget
#

class SingleSliceSegmentationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  ATTRIBUTE_PREFIX = 'SingleSliceSegmentation_'
  INPUT_BROWSER = ATTRIBUTE_PREFIX + 'InputBrowser'
  INPUT_LAST_INDEX = ATTRIBUTE_PREFIX + 'InputLastIndex'
  INPUT_IMAGE = ATTRIBUTE_PREFIX + 'InputImage'
  SEGMENTATION = ATTRIBUTE_PREFIX + 'Segmentation'
  OUTPUT_BROWSER = ATTRIBUTE_PREFIX + 'OutputBrowser'


  def __init__(self, parent):
    ScriptedLoadableModuleWidget.__init__(self, parent)

    self.logic = SingleSliceSegmentationLogic()

    # Members

    self.parameterSetNode = None
    self.editor = None
    self.ui = None


  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer)

    uiWidget = slicer.util.loadUI(self.resourcePath('UI/SingleSliceSegmentation.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set up widgets

    self.ui.inputSequenceBrowserSelector.setMRMLScene(slicer.mrmlScene)
    self.ui.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
    self.ui.inputSegmentationSelector.setMRMLScene(slicer.mrmlScene)
    self.ui.segmentationBrowserSelector.setMRMLScene(slicer.mrmlScene)

    # connections

    self.ui.inputSequenceBrowserSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputBrowserChanged)
    self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeChanged)
    self.ui.inputSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationChanged)
    self.ui.segmentationBrowserSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationBrowserChanged)

    self.ui.captureButton.connect('clicked(bool)', self.onCaptureButton)
    self.ui.captureButton.connect('clicked(bool)', self.onExportButton)

    import qSlicerSegmentationsEditorEffectsPythonQt
    # TODO: For some reason the instance() function cannot be called as a class function although it's static
    factory = qSlicerSegmentationsEditorEffectsPythonQt.qSlicerSegmentEditorEffectFactory()
    self.effectFactorySingleton = factory.instance()
    self.effectFactorySingleton.connect('effectRegistered(QString)', self.editorEffectRegistered)



  def cleanup(self):
    self.effectFactorySingleton.disconnect('effectRegistered(QString)', self.editorEffectRegistered)


  def onInputBrowserChanged(self, currentNode):
    browserNodes = slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')
    for browser in browserNodes:
      browser.SetAttribute(self.INPUT_BROWSER, "False")

    if currentNode is None:
      return

    currentNode.SetAttribute(self.INPUT_BROWSER, "True")
    logging.debug("onSequenceBrowserSelected: {}".format(currentNode.GetName()))


  def onInputVolumeChanged(self, currentNode):
    volumeNodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
    for volume in volumeNodes:
      volume.SetAttribute(self.INPUT_IMAGE, "False")

    if currentNode is None:
      return
    else:
      currentNode.SetAttribute(self.INPUT_IMAGE, "True")


  def onSegmentationChanged(self, currentNode):
    segmentationNodes = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
    for segmentation in segmentationNodes:
      segmentation.SetAttribute(self.SEGMENTATION, "False")

    if currentNode is None:
      return
    else:
      currentNode.SetAttribute(self.SEGMENTATION, "True")


  def onSegmentationBrowserChanged(self, currentNode):
    browserNodes = slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')
    for browser in browserNodes:
      browser.SetAttribute(self.OUTPUT_BROWSER, "False")

    if currentNode is None:
      return
    else:
      currentNode.SetAttribute(self.OUTPUT_BROWSER, "True")


  def onCaptureButton(self):
    logic = SingleSliceSegmentationLogic()
    enableScreenshotsFlag = self.ui.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.ui.imageThresholdSliderWidget.value
    logic.run(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)


  def onExportButton(self):
    pass


  # Segment Editor Functionalities


  def editorEffectRegistered(self):
    self.editor.updateEffectList()


  def selectParameterNode(self):
    # Select parameter set node if one is found in the scene, and create one otherwise
    segmentEditorSingletonTag = "SegmentEditor"
    segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
    if segmentEditorNode is None:
      segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
      segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
      segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
    if self.parameterSetNode == segmentEditorNode:
      # nothing changed
      return
    self.parameterSetNode = segmentEditorNode
    self.editor.setMRMLSegmentEditorNode(self.parameterSetNode)


  def getCompositeNode(self, layoutName):
    """ use the Red slice composite node to define the active volumes """
    count = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceCompositeNode')

    for n in range(count):
      compNode = slicer.mrmlScene.GetNthNodeByClass(n, 'vtkMRMLSliceCompositeNode')
      if layoutName and compNode.GetLayoutName() != layoutName:
        continue
      return compNode


  def getDefaultMasterVolumeNodeID(self):
    layoutManager = slicer.app.layoutManager()
    # Use first background volume node in any of the displayed layouts
    for layoutName in layoutManager.sliceViewNames():
      compositeNode = self.getCompositeNode(layoutName)
      if compositeNode.GetBackgroundVolumeID():
        return compositeNode.GetBackgroundVolumeID()
    # Use first background volume node in any of the displayed layouts
    for layoutName in layoutManager.sliceViewNames():
      compositeNode = self.getCompositeNode(layoutName)
      if compositeNode.GetForegroundVolumeID():
        return compositeNode.GetForegroundVolumeID()
    # Not found anything
    return None


  def enter(self):
    """Runs whenever the module is reopened"""
    if self.ui.editor.turnOffLightboxes():
      slicer.util.warningDisplay('Segment Editor is not compatible with slice viewers in light box mode.'
                                 'Views are being reset.', windowTitle='Segment Editor')

    # Allow switching between effects and selected segment using keyboard shortcuts
    self.ui.editor.installKeyboardShortcuts()

    # Set parameter set node if absent
    self.selectParameterNode()
    self.ui.editor.updateWidgetFromMRML()

    self.updateSelections()

    # If no segmentation node exists then create one so that the user does not have to create one manually
    if not self.ui.editor.segmentationNodeID():
      segmentationNode = slicer.mrmlScene.GetFirstNode(None, "vtkMRMLSegmentationNode")
      if not segmentationNode:
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
      self.ui.editor.setSegmentationNode(segmentationNode)
      if not self.ui.editor.masterVolumeNodeID():
        masterVolumeNodeID = self.getDefaultMasterVolumeNodeID()
        self.ui.editor.setMasterVolumeNodeID(masterVolumeNodeID)


  def selectParameterNode(self):
    # Select parameter set node if one is found in the scene, and create one otherwise
    segmentEditorSingletonTag = "SegmentEditor"
    segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
    if segmentEditorNode is None:
      segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
      segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
      segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
    if self.parameterSetNode == segmentEditorNode:
      # nothing changed
      return
    self.parameterSetNode = segmentEditorNode
    self.ui.editor.setMRMLSegmentEditorNode(self.parameterSetNode)


  def exit(self):
    self.ui.editor.setActiveEffect(None)
    self.ui.editor.uninstallKeyboardShortcuts()
    self.ui.editor.removeViewObservations()


  def onSceneStartClose(self, caller, event):
    self.parameterSetNode = None
    self.ui.editor.setSegmentationNode(None)
    self.ui.editor.removeViewObservations()


  def onSceneEndClose(self, caller, event):
    if self.parent.isEntered:
      self.selectParameterNode()
      self.ui.editor.updateWidgetFromMRML()


  def onSceneEndImport(self, caller, event):
    if self.parent.isEntered:
      self.selectParameterNode()
      self.ui.editor.updateWidgetFromMRML()

    self.updateSelections()


  def updateSelections(self):
    browserNodes = slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')
    self.ui.inputCollapsibleButton.collapsed = False
    for browser in browserNodes:
      if browser.GetAttribute(self.INPUT_BROWSER) == "True":
        self.ui.inputSequenceBrowserSelector.setCurrentNode(browser)
        slicer.modules.sequencebrowser.setToolBarActiveBrowserNode(browser)
        self.ui.inputCollapsibleButton.collapsed = True
        selectedItem = browser.GetAttribute(self.INPUT_LAST_INDEX)
        if selectedItem is not None:
          browser.SetSelectedItemNumber(int(selectedItem))
      if browser.GetAttribute(self.OUTPUT_BROWSER) == "True":
        self.ui.segmentationBrowserSelector.setCurrentNode(browser)

    volumeNodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
    for volume in volumeNodes:
      if volume.GetAttribute(self.INPUT_IMAGE) == "True":
        self.ui.inputVolumeSelector.setCurrentNode(volume)
        layoutManager = slicer.app.layoutManager()
        sliceLogic = layoutManager.sliceWidget('Red').sliceLogic()
        compositeNode = sliceLogic.GetSliceCompositeNode()
        compositeNode.SetBackgroundVolumeID(volume.GetID())

    segmentationNodes = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
    for segmentation in segmentationNodes:
      if segmentation.GetAttribute(self.SEGMENTATION) == "True":
        self.ui.inputSegmentationSelector.setCurrentNode(segmentation)
        self.logic.eraseCurrentSegmentation(segmentation)


#
# SingleSliceSegmentationLogic
#

class SingleSliceSegmentationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleLogic.__init__(self, parent)

    self.LabelmapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')

  def exportSlice(self,
                  selectedImage,
                  selectedSegmentation,
                  outputFolder,
                  filenamePrefix,
                  itemNumber):
    if not os.path.exists(outputFolder):
      logging.error("Export folder does not exist {}".format(outputFolder))
      return

    ic = vtk.vtkImageCast()
    ic.SetOutputScalarTypeToUnsignedChar()
    ic.Update()

    png_writer = vtk.vtkPNGWriter()

    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
      selectedSegmentation, self.LabelmapNode, selectedImage)
    segmentedImageData = self.LabelmapNode.GetImageData()
    ultrasoundData = selectedImage.GetImageData()

    seg_file_name = filenamePrefix + "_%04d_segmentation" % itemNumber + ".png"
    img_file_name = filenamePrefix + "_%04d_ultrasound" % itemNumber + ".png"
    seg_fullname = os.path.join(outputFolder, seg_file_name)
    img_fullname = os.path.join(outputFolder, img_file_name)

    ic.SetInputData(segmentedImageData)
    ic.Update()
    png_writer.SetInputData(ic.GetOutput())
    png_writer.SetFileName(seg_fullname)
    png_writer.Update()
    png_writer.Write()

    ic.SetInputData(ultrasoundData)
    ic.Update()
    png_writer.SetInputData(ic.GetOutput())
    png_writer.SetFileName(img_fullname)
    png_writer.Update()
    png_writer.Write()

    num_segments = selectedSegmentation.GetSegmentation().GetNumberOfSegments()

    # Assuming we are working with one (or the first) segment
    # Erases the current segmentation
    for i in range(num_segments):
      segmentId = selectedSegmentation.GetSegmentation().GetNthSegmentID(i)
      labelMapRep = selectedSegmentation.GetBinaryLabelmapRepresentation(segmentId)
      labelMapRep.Initialize()
      labelMapRep.Modified()
      selectedSegmentation.Modified()

  def exportNumpySlice(self,
                       selectedImage,
                       selectedSegmentation,
                       selectedSegmentationSequence,
                       outputFolder):

    if not os.path.exists(outputFolder):
      logging.error("Export folder does not exist {}".format(outputFolder))
      return

    seg_file_name = r"segmentation"
    img_file_name = r"ultrasound"
    seg_fullname = os.path.join(outputFolder, seg_file_name)
    img_fullname = os.path.join(outputFolder, img_file_name)

    num_items = selectedSegmentationSequence.GetNumberOfItems()
    n = num_items
    selectedSegmentationSequence.SelectFirstItem()

    for i in range(n):
      slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(selectedSegmentation,
                                                                               self.LabelmapNode, selectedImage)
      seg_numpy = slicer.util.arrayFromVolume(self.LabelmapNode)
      resize_seg_numpy = np.expand_dims(seg_numpy, axis=3)

      img_numpy = slicer.util.arrayFromVolume(selectedImage)
      resize_img_numpy = np.expand_dims(img_numpy, axis=3)

      if i == 0:
        seg_seq_numpy = resize_seg_numpy
        img_seq_numpy = resize_img_numpy
      else:
        seg_seq_numpy = np.concatenate((seg_seq_numpy, resize_seg_numpy))
        img_seq_numpy = np.concatenate((img_seq_numpy, resize_img_numpy))

      selectedSegmentationSequence.SelectNextItem()

    np.save(img_fullname, img_seq_numpy)
    np.save(seg_fullname, seg_seq_numpy)

  def exportPngSequence(self,
                        selectedImage,
                        selectedSegmentation,
                        selectedSegmentationSequence,
                        outputFolder,
                        baseName):
    if not os.path.exists(outputFolder):
      logging.error("Export folder does not exist {}".format(outputFolder))
      return

    imageCast = vtk.vtkImageCast()
    imageCast.SetOutputScalarTypeToUnsignedChar()
    imageCast.Update()

    pngWriter = vtk.vtkPNGWriter()

    num_items = selectedSegmentationSequence.GetNumberOfItems()
    selectedSegmentationSequence.SelectFirstItem()
    for i in range(num_items):
      slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(selectedSegmentation,
                                                                               self.LabelmapNode,
                                                                               selectedImage)
      segmentedImageData = self.LabelmapNode.GetImageData()
      ultrasoundData = selectedImage.GetImageData()

      segmentationFileName = baseName + "_%04d_segmentation" % i + ".png"
      ultrasoundFileName = baseName + "_%04d_ultrasound" % i + ".png"
      segmentationFullname = os.path.join(outputFolder, segmentationFileName)
      ultrasoundFullname = os.path.join(outputFolder, ultrasoundFileName)

      imageCast.SetInputData(segmentedImageData)
      imageCast.Update()
      pngWriter.SetInputData(imageCast.GetOutput())
      pngWriter.SetFileName(segmentationFullname)
      pngWriter.Update()
      pngWriter.Write()

      imageCast.SetInputData(ultrasoundData)
      imageCast.Update()
      pngWriter.SetInputData(imageCast.GetOutput())
      pngWriter.SetFileName(ultrasoundFullname)
      pngWriter.Update()
      pngWriter.Write()

      selectedSegmentationSequence.SelectNextItem()
      slicer.app.processEvents()

  def captureSlice(self, selectedSegmentationSequence, selectedSegmentation):
    selectedSegmentationSequence.SaveProxyNodesState()
    self.eraseCurrentSegmentation(selectedSegmentation)

  def eraseCurrentSegmentation(self, selectedSegmentation):
    num_segments = selectedSegmentation.GetSegmentation().GetNumberOfSegments()
    for i in range(num_segments):
      segmentId = selectedSegmentation.GetSegmentation().GetNthSegmentID(i)

      # editor = slicer.modules.singleslicesegmentation.widgetRepresentation().self().editor
      # editor.setActiveEffectByName("Logical operators")
      # effect = editor.activeEffect()
      # effect.setParameter("Operation", "CLEAR")
      # effect.self().onApply()

      # import vtkSegmentationCorePython as vtkSegmentationCore
      labelMapRep = selectedSegmentation.GetBinaryLabelmapRepresentation(segmentId)
      slicer.vtkOrientedImageDataResample.FillImage(labelMapRep, 0, labelMapRep.GetExtent())
      slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
        labelMapRep, selectedSegmentation, segmentId, slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE)

      # labelMapRep.Initialize()
      ## numpy_data = numpy_support.vtk_to_numpy(labelMapRep.GetPointData().GetScalars())
      ## numpy_data.fill(0)
      # labelMapRep.Modified()
      # selectedSegmentation.Modified()

  def hasImageData(self, volumeNode):
    """This is an example logic method that
returns true if the passed in volume
node has valid image data
"""
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
"""
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID() == outputVolumeNode.GetID():
      logging.debug(
        'isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True


class SingleSliceSegmentationTest(ScriptedLoadableModuleTest):
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
    self.test_SingleSliceSegmentation1()


  def test_SingleSliceSegmentation1(self):
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
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767',
      checksums='SHA256:12d17fba4f2e1f1a843f0757366f28c3f3e1a8bb38836f0de2a32bb1cd476560')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = SingleSliceSegmentationLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
