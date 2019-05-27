import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import logging

#
# SingleSliceSegmentation
#

class SingleSliceSegmentation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Single Slice Segmentation" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Ultrasound"]
    self.parent.dependencies = []
    self.parent.contributors = ["Tamas Ungi (Queen's University), Victoria Wu (Queen's University)"] # replace with "Firstname Lastname (Organization)"
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

class SingleSliceSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def __init__(self, parent):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)

    # Members
    self.parameterSetNode = None
    self.editor = None

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Export parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input selector
    #

    self.sequenceBrowserSelector = slicer.qMRMLNodeComboBox()
    self.sequenceBrowserSelector.nodeTypes = ["vtkMRMLSequenceBrowserNode"]
    self.sequenceBrowserSelector.selectNodeUponCreation = True
    self.sequenceBrowserSelector.addEnabled = False
    self.sequenceBrowserSelector.removeEnabled = False
    self.sequenceBrowserSelector.noneEnabled = False
    self.sequenceBrowserSelector.setMRMLScene( slicer.mrmlScene )
    parametersFormLayout.addRow("Input sequence browser: ", self.sequenceBrowserSelector)

    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input image" )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    self.segmentationSelector = slicer.qMRMLNodeComboBox()
    self.segmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.segmentationSelector.selectNodeUponCreation = True
    self.segmentationSelector.addEnabled = False
    self.segmentationSelector.removeEnabled = False
    self.segmentationSelector.noneEnabled = False
    self.segmentationSelector.showHidden = False
    self.segmentationSelector.showChildNodeTypes = False
    self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
    self.segmentationSelector.setToolTip("Pick the segmentation")
    parametersFormLayout.addRow("Segmentation: ", self.segmentationSelector)

    self.filenamePrefixEdit = qt.QLineEdit()
    parametersFormLayout.addRow("File name prefix", self.filenamePrefixEdit)
    #
    # output volume selector
    #
    self.outputDirButton = ctk.ctkDirectoryButton()

    parametersFormLayout.addRow("Output folder: ", self.outputDirButton)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Export slice")
    self.applyButton.toolTip = "Run the algorithm."
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)

    #
    # Segmentation Sequence  Area
    #
    segmentationCollapsibleButton = ctk.ctkCollapsibleButton()
    segmentationCollapsibleButton.text = "Segmentation Sequence"
    self.layout.addWidget( segmentationCollapsibleButton )

    # Layout within the dummy collapsible button
    segmentationFormLayout = qt.QFormLayout(segmentationCollapsibleButton)

    #
    # Segmentation sequence selector
    #
    self.segmentationSequenceSelector = slicer.qMRMLNodeComboBox()
    self.segmentationSequenceSelector.nodeTypes = ["vtkMRMLSequenceBrowserNode"]
    self.segmentationSequenceSelector.selectNodeUponCreation = True
    self.segmentationSequenceSelector.addEnabled = False
    self.segmentationSequenceSelector.removeEnabled = False
    self.segmentationSequenceSelector.noneEnabled = False
    self.segmentationSequenceSelector.setMRMLScene(slicer.mrmlScene)
    segmentationFormLayout.addRow("Segmentation sequence browser: ", self.segmentationSequenceSelector)


    #
    # Capture Frame Button
    #
    self.captureFrame = qt.QPushButton("Capture slice")
    self.captureFrame.toolTip = "Run the algorithm."
    segmentationFormLayout.addRow(self.captureFrame)

    # connections
    self.captureFrame.connect('clicked(bool)', self.onCaptureFrame)

    #
    # Segmentation Editor  Area
    #
    segmentEditorCollapsibleButton = ctk.ctkCollapsibleButton()
    segmentEditorCollapsibleButton.text = "Segment Editor"
    self.layout.addWidget( segmentEditorCollapsibleButton )

    # Layout within the dummy collapsible button
    segmentEditorFormLayout = qt.QFormLayout(segmentEditorCollapsibleButton)

    import qSlicerSegmentationsModuleWidgetsPythonQt
    self.editor = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
    self.editor.setMaximumNumberOfUndoStates(10)
    # Set parameter node first so that the automatic selections made when the scene is set are saved
    self.selectParameterNode()
    self.editor.setMRMLScene(slicer.mrmlScene)
    segmentEditorFormLayout.addWidget(self.editor)

    import qSlicerSegmentationsEditorEffectsPythonQt
    #TODO: For some reason the instance() function cannot be called as a class function although it's static
    factory = qSlicerSegmentationsEditorEffectsPythonQt.qSlicerSegmentEditorEffectFactory()
    self.effectFactorySingleton = factory.instance()
    self.effectFactorySingleton.connect('effectRegistered(QString)', self.editorEffectRegistered)

    # Connect observers to scene events
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)


  #Segment Editor Functionalities
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
    for n in xrange(count):
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
    """Runs whenever the module is reopened
    """
    if self.editor.turnOffLightboxes():
      slicer.util.warningDisplay('Segment Editor is not compatible with slice viewers in light box mode.'
        'Views are being reset.', windowTitle='Segment Editor')

    # Allow switching between effects and selected segment using keyboard shortcuts
    self.editor.installKeyboardShortcuts()

    # Set parameter set node if absent
    self.selectParameterNode()
    self.editor.updateWidgetFromMRML()

    # If no segmentation node exists then create one so that the user does not have to create one manually
    if not self.editor.segmentationNodeID():
      segmentationNode = slicer.mrmlScene.GetFirstNode(None, "vtkMRMLSegmentationNode")
      if not segmentationNode:
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
      self.editor.setSegmentationNode(segmentationNode)
      if not self.editor.masterVolumeNodeID():
        masterVolumeNodeID = self.getDefaultMasterVolumeNodeID()
        self.editor.setMasterVolumeNodeID(masterVolumeNodeID)

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

  def exit(self):
    self.editor.setActiveEffect(None)
    self.editor.uninstallKeyboardShortcuts()
    self.editor.removeViewObservations()

  def onSceneStartClose(self, caller, event):
    self.parameterSetNode = None
    self.editor.setSegmentationNode(None)
    self.editor.removeViewObservations()

  def onSceneEndClose(self, caller, event):
    if self.parent.isEntered:
      self.selectParameterNode()
      self.editor.updateWidgetFromMRML()

  def onSceneEndImport(self, caller, event):
    if self.parent.isEntered:
      self.selectParameterNode()
      self.editor.updateWidgetFromMRML()

  def cleanup(self):
    self.removeObservers()
    self.effectFactorySingleton.disconnect('effectRegistered(QString)', self.editorEffectRegistered)

  #Export Slice
  def onApplyButton(self):
    selectedImage = self.inputSelector.currentNode()
    selectedSegmentation = self.segmentationSelector.currentNode()
    outputFolder = str(self.outputDirButton.directory)
    filenamePrefix = str(self.filenamePrefixEdit.text)
    browserNode = self.sequenceBrowserSelector.currentNode()

    if selectedImage is None:
      logging.error("No input image selected!")
      return
    if selectedSegmentation is None:
      logging.error("No segmentation selected!")
      return
    if browserNode is None:
      logging.error("No browser node selected!")
      return

    itemNumber = browserNode.GetSelectedItemNumber()

    logic = SingleSliceSegmentationLogic()
    logic.exportSlice(selectedImage, selectedSegmentation, outputFolder, filenamePrefix, itemNumber)

  #Capture Slice
  def onCaptureFrame(self):
    browserNode = self.sequenceBrowserSelector.currentNode() #The original sequence we are capturing the image from
    selectedSegmentationSequence = self.segmentationSequenceSelector.currentNode() #The segmentation sequence we want to add the image to
    selectedImage = self.inputSelector.currentNode()
    selectedSegmentation = self.segmentationSelector.currentNode()

    if selectedImage is None:
      logging.error("No input image selected!")
      return
    if selectedSegmentation is None:
      logging.error("No segmentation selected!")
      return
    if selectedSegmentationSequence is None:
      logging.error("No segmentation sequence browser selected!")
      return
    if browserNode is None:
      logging.error("No browser node selected!")
      return

    logic = SingleSliceSegmentationLogic()
    logic.captureSlice(selectedSegmentationSequence, selectedSegmentation)

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

    # Assuming we are working with one (or the first) segment

    segmentId = selectedSegmentation.GetSegmentation().GetNthSegmentID(0)
    labelMapRep = selectedSegmentation.GetBinaryLabelmapRepresentation(segmentId)
    labelMapRep.Initialize()
    labelMapRep.Modified()
    selectedSegmentation.Modified()

  def captureSlice(self, selectedSegmentationSequence, selectedSegmentation):

    #Capture image into selectedSegmentationSequence
    #Make sure in the sequence browser GUI to create segmentation proxy node and save changes
    selectedSegmentationSequence.SaveProxyNodesState()

    #Assuming we are working with one (or the first) segment
    #Erases the current segmentation
    segmentId = selectedSegmentation.GetSegmentation().GetNthSegmentID(0)
    labelMapRep = selectedSegmentation.GetBinaryLabelmapRepresentation(segmentId)
    labelMapRep.Initialize()
    labelMapRep.Modified()
    selectedSegmentation.Modified()



  def hasImageData(self,volumeNode):
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
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
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
      uris='http://slicer.kitware.com/midas3/download?items=5767')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = SingleSliceSegmentationLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
