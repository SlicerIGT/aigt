from __future__ import print_function
import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

import numpy as np


DEFAULT_INPUT_IMAGE_NAME = "Image_Image"

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


    def setup(self):
      # Register subject hierarchy plugin
      import SubjectHierarchyPlugins
      scriptedPlugin = slicer.qSlicerSubjectHierarchyScriptedPlugin(None)
      scriptedPlugin.setPythonSource(SubjectHierarchyPlugins.SegmentEditorSubjectHierarchyPlugin.filePath)

#
# SingleSliceSegmentationWidget
#

class SingleSliceSegmentationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  ATTRIBUTE_PREFIX = 'SingleSliceSegmentation_'
  INPUT_BROWSER = ATTRIBUTE_PREFIX + 'InputBrowser'
  INPUT_SKIP_NUMBER = ATTRIBUTE_PREFIX + 'InputSkipNumber'
  DEFAULT_INPUT_SKIP_NUMBER = 4
  INPUT_LAST_INDEX = ATTRIBUTE_PREFIX + 'InputLastIndex'
  INPUT_IMAGE_ID = ATTRIBUTE_PREFIX + 'InputImageId'
  SEGMENTATION = ATTRIBUTE_PREFIX + 'Segmentation'
  OUTPUT_BROWSER = ATTRIBUTE_PREFIX + 'OutputBrowser'
  ORIGINAL_IMAGE_INDEX = ATTRIBUTE_PREFIX + 'OriginalImageIndex'


  def __init__(self, parent):
    ScriptedLoadableModuleWidget.__init__(self, parent)

    self.logic = SingleSliceSegmentationLogic()

    # Members

    self.parameterSetNode = None
    self.editor = None
    self.ui = None
    
    # Shortcuts

    self.shortcutS = qt.QShortcut(slicer.util.mainWindow())
    self.shortcutS.setKey(qt.QKeySequence('s'))
    self.shortcutD = qt.QShortcut(slicer.util.mainWindow())
    self.shortcutD.setKey(qt.QKeySequence('d'))
    self.shortcutC = qt.QShortcut(slicer.util.mainWindow())
    self.shortcutC.setKey(qt.QKeySequence('c'))




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
    self.ui.skipImagesSpinBox.connect("valueChanged(int)", self.onSkipImagesNumChanged)

    self.ui.captureButton.connect('clicked(bool)', self.onCaptureButton)
    self.ui.clearSegmentationButton.connect('clicked(bool)', self.onClearButton)
    self.ui.skipImageButton.connect('clicked(bool)', self.onSkipButton)
    self.ui.exportButton.connect('clicked(bool)', self.onExportButton)
    self.ui.layoutSelectButton.connect('clicked(bool)', self.onLayoutSelectButton)

    self.ui.editor.setMRMLScene(slicer.mrmlScene)
    
    import qSlicerSegmentationsEditorEffectsPythonQt
    # TODO: For some reason the instance() function cannot be called as a class function although it's static
    factory = qSlicerSegmentationsEditorEffectsPythonQt.qSlicerSegmentEditorEffectFactory()
    self.effectFactorySingleton = factory.instance()
    self.effectFactorySingleton.connect('effectRegistered(QString)', self.editorEffectRegistered)

    # Add custom layout to layout selection menu
    customLayout = """
    <layout type="horizontal" split="true">
      <item>
       <view class="vtkMRMLSliceNode" singletontag="Red">
        <property name="orientation" action="default">Axial</property>
        <property name="viewlabel" action="default">R</property>
        <property name="viewcolor" action="default">#F34A33</property>
       </view>
      </item>
      <item>
       <view class="vtkMRMLViewNode" singletontag="1">
         <property name="viewlabel" action="default">1</property>
       </view>
      </item>
    </layout>
    """

    customLayoutId = 501

    layoutManager = slicer.app.layoutManager()
    layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(customLayoutId, customLayout)

    viewToolBar = slicer.util.mainWindow().findChild('QToolBar', 'ViewToolBar')
    layoutMenu = viewToolBar.widgetForAction(viewToolBar.actions()[0]).menu()
    layoutSwitchActionParent = layoutMenu
    layoutSwitchAction = layoutSwitchActionParent.addAction("red + 3D side by side")  # add inside layout list
    layoutSwitchAction.setData(customLayoutId)
    # layoutSwitchAction.setIcon(qt.QIcon(':Icons/Go.png'))
    layoutSwitchAction.setToolTip('3D and slice view')
    layoutSwitchAction.connect('triggered()', lambda layoutId=customLayoutId: slicer.app.layoutManager().setLayout(layoutId))

    customLayout = """
    <layout type="horizontal" split="true">
      <item>
       <view class="vtkMRMLSliceNode" singletontag="Red">
        <property name="orientation" action="default">Axial</property>
        <property name="viewlabel" action="default">R</property>
        <property name="viewcolor" action="default">#F34A33</property>
       </view>
      </item>
      <item>
        <layout type=\"horizontal\">
          <item>
            <view class="vtkMRMLViewNode" singletontag="1">
              <property name="viewlabel" action="default">1</property>
            </view>
          </item>
          <item>
            <view class="vtkMRMLViewNode" singletontag="2" type="secondary">
              <property name="viewlabel" action="default">2</property>
            </view>
          </item>
        </layout>
      </item>
    </layout>
    """

    customLayoutId = 502

    layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(customLayoutId, customLayout)

    layoutSwitchActionParent = layoutMenu
    layoutSwitchAction = layoutSwitchActionParent.addAction("red + stacked dual 3D")  # add inside layout list
    layoutSwitchAction.setData(customLayoutId)
    # layoutSwitchAction.setIcon(qt.QIcon(':Icons/Go.png'))
    layoutSwitchAction.setToolTip('Dual 3D and slice view')
    layoutSwitchAction.connect('triggered()',
                               lambda layoutId=customLayoutId: slicer.app.layoutManager().setLayout(layoutId))

  def cleanup(self):
    self.effectFactorySingleton.disconnect('effectRegistered(QString)', self.editorEffectRegistered)

  def onInputBrowserChanged(self, currentNode):
    browserNodes = slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')
    for browser in browserNodes:
      browser.SetAttribute(self.INPUT_BROWSER, "False")

    if currentNode is None:
      return

    currentNode.SetAttribute(self.INPUT_BROWSER, "True")
    
    savedSkip = currentNode.GetAttribute(self.INPUT_SKIP_NUMBER)
    if savedSkip is not None:
      slicer.modules.singleslicesegmentation.widgetRepresentation().self().ui.skipImagesSpinBox.value = int(savedSkip)
    else:
      numSkip = slicer.modules.singleslicesegmentation.widgetRepresentation().self().ui.skipImagesSpinBox.value
      currentNode.SetAttribute(self.INPUT_SKIP_NUMBER, str(numSkip))
    
    logging.debug("onSequenceBrowserSelected: {}".format(currentNode.GetName()))


  def onInputVolumeChanged(self, currentNode):
    # todo: delete after testing parameter node
    # volumeNodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
    # for volume in volumeNodes:
    #   volume.SetAttribute(self.INPUT_IMAGE, "False")

    if currentNode is None:
      self.parameterSetNode.RemoveAttribute(self.INPUT_IMAGE_ID)
    else:
      self.parameterSetNode.SetAttribute(self.INPUT_IMAGE_ID, currentNode.GetID())


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

  def onImageChange(self, browserNode):
    inputImageIndex = browserNode.GetSelectedItemNumber()

    if inputImageIndex is not None:
      inputBrowserNode.SetAttribute(self.ORIGINAL_IMAGE_INDEX, str(inputImageIndex))
  
  def onSkipImagesNumChanged(self, value):
    inputBrowserNode = self.ui.inputSequenceBrowserSelector.currentNode()
    if inputBrowserNode is not None:
      inputBrowserNode.SetAttribute(self.INPUT_SKIP_NUMBER, str(value))
  

  def onCaptureButton(self):
    inputBrowserNode = self.ui.inputSequenceBrowserSelector.currentNode()
    inputImage = self.ui.inputVolumeSelector.currentNode()
    selectedSegmentationBrowser = self.ui.segmentationBrowserSelector.currentNode()
    selectedSegmentation = self.ui.inputSegmentationSelector.currentNode()
    numSkip = slicer.modules.singleslicesegmentation.widgetRepresentation().self().ui.skipImagesSpinBox.value

    if inputBrowserNode is None:
      logging.error("No browser node selected!")
      return
    if selectedSegmentation is None:
      logging.error("No segmentation selected!")
      return
    if selectedSegmentationBrowser is None:
      logging.error("No segmentation sequence browser selected!")
      return

    inputImageIndex = inputBrowserNode.GetSelectedItemNumber()
    
    original_index_str = selectedSegmentation.GetAttribute(self.ORIGINAL_IMAGE_INDEX)
    if original_index_str is None or original_index_str == "None" or original_index_str == "":
      selectedSegmentation.SetAttribute(self.ORIGINAL_IMAGE_INDEX, str(inputImageIndex))
      self.logic.captureSlice(selectedSegmentationBrowser, selectedSegmentation, inputImage)
    else:
      self.logic.captureSlice(selectedSegmentationBrowser, selectedSegmentation, inputImage)

    self.logic.eraseCurrentSegmentation(selectedSegmentation)
    selectedSegmentation.SetAttribute(self.ORIGINAL_IMAGE_INDEX, "None")
    inputBrowserNode.SelectNextItem(numSkip)

  def onClearButton(self):
    selectedSegmentation = self.ui.inputSegmentationSelector.currentNode()
    if selectedSegmentation is None:
      logging.error("No segmentation selected!")
      return
    self.logic.eraseCurrentSegmentation(selectedSegmentation)
    selectedSegmentation.SetAttribute(self.ORIGINAL_IMAGE_INDEX, "None")
  
  
  def onSkipButton(self):
    inputBrowserNode = self.ui.inputSequenceBrowserSelector.currentNode()
    selectedSegmentation = self.ui.inputSegmentationSelector.currentNode()
    numSkip = slicer.modules.singleslicesegmentation.widgetRepresentation().self().ui.skipImagesSpinBox.value
    if inputBrowserNode is None:
      logging.error("No browser node selected!")
      return
    if selectedSegmentation is None:
      logging.error("No segmentation selected!")
      return
    self.logic.eraseCurrentSegmentation(selectedSegmentation)
    inputBrowserNode.SelectNextItem(numSkip)
  
  
  def onExportButton(self):
    selectedSegmentationSequence = self.ui.segmentationBrowserSelector.currentNode()
    selectedSegmentation = self.ui.inputSegmentationSelector.currentNode()
    selectedImage = self.ui.inputVolumeSelector.currentNode()
    selectedImageSequence = self.ui.inputSequenceBrowserSelector.currentNode()
    outputFolder = self.ui.outputDirectoryButton.directory
    baseName = self.ui.filenamePrefixEdit.text

    if selectedSegmentation is None:
      logging.error("No segmentation selected!")
      return
    if selectedSegmentationSequence is None:
      logging.error("No segmentation sequence browser selected!")
      return
    if selectedImage is None:
      logging.error("No image selected!")
      return
    if selectedImage is None:
      logging.error("No image sequence browser selected!")
      return

    self.logic.exportPngSequence(selectedImage,
                                 selectedImageSequence,
                                 selectedSegmentation,
                                 selectedSegmentationSequence,
                                 outputFolder,
                                 baseName)

    if self.ui.exportTransformCheckBox.checked:
      self.logic.exportNumpySlice(selectedImage,
                                  selectedSegmentation,
                                  selectedSegmentationSequence,
                                  outputFolder)
      self.logic.exportTransformToWorldSequence(selectedSegmentation,
                                                selectedSegmentationSequence,
                                                outputFolder,
                                                baseName)


  def onLayoutSelectButton(self):
    layoutManager = slicer.app.layoutManager()
    currentLayout = layoutManager.layout

    # place skeleton model in first 3d view
    skeleton_volume = slicer.util.getFirstNodeByName("SkeletonModel")
    viewNode = layoutManager.threeDWidget(0).mrmlViewNode()

    if skeleton_volume is not None:
      displayNode = skeleton_volume.GetDisplayNode()
      displayNode.SetViewNodeIDs([viewNode.GetID()])

    # place reconstructed volume in second 3d view

    # hacky but necessary way to ensure that we grab the correct browser node
    i = 1
    found = False
    browser = slicer.util.getFirstNodeByClassByName('vtkMRMLSequenceBrowserNode', 'LandmarkingScan')
    while not found and i < 6:
      if browser == None:
        browser = slicer.util.getFirstNodeByClassByName('vtkMRMLSequenceBrowserNode', 'LandmarkingScan_{}'.format(str(i)))
        if browser != None:
          found = True
      else:
        found = True

      i += 1

    if browser is not None:
      spine_volume = slicer.util.getFirstNodeByName(browser.GetName() + 'ReconstructionResults')
      if layoutManager.threeDWidget(1) is not None:
        viewNode = layoutManager.threeDWidget(1).mrmlViewNode()
      else:
        newView = slicer.vtkMRMLViewNode()
        newView = slicer.mrmlScene.AddNode(newView)
        newWidget = slicer.qMRMLThreeDWidget()
        newWidget.setMRMLScene(slicer.mrmlScene)
        newWidget.setMRMLViewNode(newView)

      if spine_volume is not None:
        displayNode = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(spine_volume)
        displayNode.SetViewNodeIDs([viewNode.GetID()])
        spine_volume.SetDisplayVisibility(1)


    if currentLayout == 501:
      layoutManager.setLayout(502) # switch to dual 3d + red slice layout
    elif currentLayout == 502:
      layoutManager.setLayout(6) # switch to red slice only layout
    else:
      layoutManager.setLayout(501) # switch to custom layout with 3d viewer


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
    self.ui.editor.setMRMLSegmentEditorNode(self.parameterSetNode)


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
    logging.debug('Entered SingleSliceSegmentation module widget')

    # Prevent segmentation master volume to be created in the wrong position.
    # Todo: Instead of this, the input image should be kept transformed, and the segmentation also transformed

    inputImageNode = slicer.util.getFirstNodeByName(DEFAULT_INPUT_IMAGE_NAME)
    if inputImageNode is not None and inputImageNode.GetClassName() == 'vtkMRMLScalarVolumeNode':
      inputImageNode.SetAndObserveTransformNodeID(None)

    if self.ui.editor.turnOffLightboxes():
      slicer.util.warningDisplay('Segment Editor is not compatible with slice viewers in light box mode.'
                                 'Views are being reset.', windowTitle='Segment Editor')

    # Allow switching between effects and selected segment using keyboard shortcuts
    self.ui.editor.installKeyboardShortcuts()

    # Set parameter set node if absent
    self.selectParameterNode()
    self.ui.editor.updateWidgetFromMRML()
    
    # Update UI
    slicer.modules.sequences.setToolBarVisible(True)
    self.updateSelections()
    self.connectKeyboardShortcuts()

    # Collapse input group if all is selected

    if self.ui.inputSequenceBrowserSelector.currentNode() is not None and\
      self.ui.inputVolumeSelector.currentNode() is not None and\
      self.ui.inputSegmentationSelector.currentNode() is not None and\
      self.ui.segmentationBrowserSelector.currentNode() is not None:
      self.ui.inputCollapsibleButton.collapsed = True
    else:
      self.ui.inputCollapsibleButton.collapsed = False

    # If no segmentation node exists then create one so that the user does not have to create one manually

    if not self.ui.editor.segmentationNodeID():
      segmentationNode = slicer.mrmlScene.GetFirstNode(None, "vtkMRMLSegmentationNode")
      if not segmentationNode:
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
      self.ui.editor.setSegmentationNode(segmentationNode)
      if not self.ui.editor.masterVolumeNodeID():
        masterVolumeNodeID = self.getDefaultMasterVolumeNodeID()
        self.ui.editor.setMasterVolumeNodeID(masterVolumeNodeID)

    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(6)

    redController = slicer.app.layoutManager().sliceWidget('Red').sliceController()
    redController.fitSliceToBackground()
  
  
  def connectKeyboardShortcuts(self):
    self.shortcutS.connect('activated()', self.onSkipButton)
    self.shortcutD.connect('activated()', self.onClearButton)
    self.shortcutC.connect('activated()', self.onCaptureButton)
  
  
  def disconnectKeyboardShortcuts(self):
    self.shortcutS.activated.disconnect()
    self.shortcutD.activated.disconnect()
    self.shortcutC.activated.disconnect()


  def exit(self):
    self.ui.editor.setActiveEffect(None)
    self.ui.editor.uninstallKeyboardShortcuts()
    self.ui.editor.removeViewObservations()
    
    self.disconnectKeyboardShortcuts()


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

    for browser in browserNodes:
      if browser.GetAttribute(self.INPUT_BROWSER) == "True":
        self.ui.inputSequenceBrowserSelector.setCurrentNode(browser)
        slicer.modules.sequences.setToolBarActiveBrowserNode(browser)
        self.ui.inputCollapsibleButton.collapsed = True
        selectedItem = browser.GetAttribute(self.INPUT_LAST_INDEX)
        if selectedItem is not None:
          browser.SetSelectedItemNumber(int(selectedItem))
        skipNumber = browser.GetAttribute(self.INPUT_SKIP_NUMBER)
        if skipNumber is None:
          skipNumber = self.DEFAULT_INPUT_SKIP_NUMBER
        self.ui.skipImagesSpinBox.value = int(skipNumber)
      if browser.GetAttribute(self.OUTPUT_BROWSER) == "True":
        self.ui.segmentationBrowserSelector.setCurrentNode(browser)

    segmentationNodes = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
    for segmentation in segmentationNodes:
      if segmentation.GetAttribute(self.SEGMENTATION) == "True":
        self.ui.inputSegmentationSelector.setCurrentNode(segmentation)
        self.logic.eraseCurrentSegmentation(segmentation)
        self.ui.editor.setSegmentationNode(segmentation)

    inputImageAttribute = self.parameterSetNode.GetAttribute(self.INPUT_IMAGE_ID)
    if inputImageAttribute is not None:
      inputImageNode = slicer.mrmlScene.GetNodeByID(inputImageAttribute)
      if inputImageNode is not None:
        self.ui.inputVolumeSelector.setCurrentNode(inputImageNode)
        layoutManager = slicer.app.layoutManager()
        sliceLogic = layoutManager.sliceWidget('Red').sliceLogic()
        compositeNode = sliceLogic.GetSliceCompositeNode()
        compositeNode.SetBackgroundVolumeID(inputImageNode.GetID())
        self.ui.editor.setMasterVolumeNode(inputImageNode)


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
                        selectedImageSequence,
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
      originalIndex = selectedSegmentation.GetAttribute(SingleSliceSegmentationWidget.ORIGINAL_IMAGE_INDEX)
      try:
        selectedImageSequence.SetSelectedItemNumber(int(originalIndex))
      except:
        logging.error(f"Original image not found for segmentation {i} - skipping")
        continue
      slicer.modules.sequences.logic().UpdateAllProxyNodes()

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
      slicer.modules.sequences.logic().UpdateAllProxyNodes()


  def exportTransformToWorldSequence(self,
                                     selectedSegmentation,
                                     selectedSegmentationSequence,
                                     outputFolder,
                                     baseName):
    if not os.path.exists(outputFolder):
      logging.error("Export folder does not exist {}".format(outputFolder))
      return

    # Grab the Landmarking Scan Sequence - we will get transforms from here.
    landmarkingScanSequence = None
    sequenceBrowserNodesList = slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')
    for sequenceBrowserNode in sequenceBrowserNodesList:
      if 'LandmarkingScan' in sequenceBrowserNode.GetName():
        landmarkingScanSequence = sequenceBrowserNode
    if landmarkingScanSequence is None:
      logging.error("Landmarking Scan sequence does not exist.")
      return

    imageToTrans = slicer.util.getFirstNodeByName("ImageToTransd*")
    if imageToTrans is None:
      logging.error("ImageToTransducer transform does not exist.")
      return

    num_items = selectedSegmentationSequence.GetNumberOfItems()
    selectedSegmentationSequence.SelectFirstItem()
    landmarkingScanSequence.SelectFirstItem()

    for i in range(num_items):

      # Get landmarking scan transform index, set it, get TransformToWorld
      transformIndex = int(selectedSegmentation.GetAttribute(SingleSliceSegmentationWidget.ORIGINAL_IMAGE_INDEX))
      landmarkingScanSequence.SetSelectedItemNumber(transformIndex)
      transformToWorld = vtk.vtkMatrix4x4()
      imageToTrans.GetMatrixTransformToWorld(transformToWorld)
      transformToWorld_numpy = slicer.util.arrayFromVTKMatrix(transformToWorld)
      
      if i == 0:
        transforms_seq_numpy = transformToWorld_numpy[np.newaxis, :]
      else:
        transforms_seq_numpy = np.concatenate((transforms_seq_numpy, transformToWorld_numpy[np.newaxis, :]), axis=0)

      selectedSegmentationSequence.SelectNextItem()
      slicer.app.processEvents()
    
    # Save stacked transforms as output numpy array.
    transformFileName = baseName + "_transform.npy"
    transformFullname = os.path.join(outputFolder, transformFileName)
    np.save(transformFullname, transforms_seq_numpy)


  def captureSlice(self, selectedSegmentationSequence, selectedSegmentation, inputImage):
    originalIndex = selectedSegmentation.GetAttribute(SingleSliceSegmentationWidget.ORIGINAL_IMAGE_INDEX)
    
    #todo overwrite existing segmentation of the same image
    
    # Figure out which sequence of the selected browser has the proxy node inputImage and selectedSegmentation
    
    segmentedImageSequenceNode = selectedSegmentationSequence.GetSequenceNode(selectedSegmentation)
    if segmentedImageSequenceNode is None:
      logging.error("Sequence not found for input image: {}".format(inputImage.GetName()))
      return
    
    segmentationSequenceNode = selectedSegmentationSequence.GetSequenceNode(selectedSegmentation)
    if segmentationSequenceNode is None:
      logging.error("Sequence not found for segmentation: {}".format(selectedSegmentation.GetName()))
      return
    
    # Check all nodes saved in sequence if anyone has the same attribute (this image already recorded)
    
    numDataNodes = segmentedImageSequenceNode.GetNumberOfDataNodes()
    recordedOriginalIndex = None
    for i in range(numDataNodes):
      segmentationNode = segmentedImageSequenceNode.GetNthDataNode(i)
      savedIndex = segmentationNode.GetAttribute(SingleSliceSegmentationWidget.ORIGINAL_IMAGE_INDEX)
      if originalIndex == savedIndex:
        recordedOriginalIndex = i
        break
    
    # If this image has been saved previously, overwrite instead of add new
    
    try:
      recordedOriginalIndex = int(recordedOriginalIndex)
    except:
      recordedOriginalIndex = None
    
    if recordedOriginalIndex is None:
      selectedSegmentationSequence.SaveProxyNodesState()
    else:
      recordedIndexValue = segmentedImageSequenceNode.GetNthIndexValue(recordedOriginalIndex)
      segmentationSequenceNode.SetDataNodeAtValue(selectedSegmentation, recordedIndexValue)


  def eraseCurrentSegmentation(self, selectedSegmentation):
    num_segments = selectedSegmentation.GetSegmentation().GetNumberOfSegments()
    for i in range(num_segments):
      segmentId = selectedSegmentation.GetSegmentation().GetNthSegmentID(i)

      import vtkSegmentationCorePython as vtkSegmentationCore
      try:
        labelMapRep = selectedSegmentation.GetBinaryLabelmapRepresentation(segmentId)
      except:
        labelMapRep = selectedSegmentation.GetBinaryLabelmapInternalRepresentation(segmentId)
      slicer.vtkOrientedImageDataResample.FillImage(labelMapRep, 0, labelMapRep.GetExtent())
      slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
        labelMapRep, selectedSegmentation, segmentId, slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE)


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
    pass
