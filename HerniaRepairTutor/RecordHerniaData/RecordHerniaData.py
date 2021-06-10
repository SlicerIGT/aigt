import os
import unittest
import logging
import time
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

#
# RecordHerniaData
#

class RecordHerniaData(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Record Hernia Data"
    self.parent.categories = ["Training"]
    self.parent.dependencies = []
    self.parent.contributors = ["Rebecca Hisey (Queen's University)"]
    self.parent.helpText = """
This module is used to record RGB and depth video for Inguinal Hernia Repair using 2 Intel Realsense cameras.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

    # Additional initialization step after application startup is complete
    #slicer.app.connect("startupCompleted()", registerSampleData)

#
# RecordHerniaDataWidget
#

class RecordHerniaDataWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = RecordHerniaDataLogic()
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/RecordHerniaData.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = RecordHerniaDataLogic()
    self.recordingStarted = False
    self.logic.setupScene()
    self.logic.setParameterNode(self._parameterNode)

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # Buttons
    self.ui.StartStopRecordingButton.connect('clicked(bool)', self.onStartStopRecordingClicked)

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
    if not self._parameterNode.GetParameter('SavedScenesDirectory'):
      savedScenesDir = self.resourcePath('SavedScenes')
      if (not os.path.exists(savedScenesDir)):
        os.makedirs(savedScenesDir)
      self._parameterNode.SetParameter('SavedScenesDirectory',savedScenesDir)

    self.logic.setParameterNode(self._parameterNode)

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

    self._parameterNode.EndModify(wasModified)

  def onStartStopRecordingClicked(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:
      rgbPort = int(self.ui.rGBPortLineEdit.text)
      depthPort = int(self.ui.depthPortLineEdit.text)
      secondRgbPort = int(self.ui.secondCameraRGBPortLineEdit.text)
      secondDepthPort = int(self.ui.secondCameraDepthPortLineEdit.text)
      if not self.recordingStarted:
        self.ui.StartStopRecordingButton.setText("Stop Recording")
        self.logic.StartRecording(self.ui.userIDLineEdit.text)
        self.recordingStarted = True
      else:
        self.logic.StopRecording()
        self.logic.StopRecording()
        self.recordingStarted = False
        self.ui.StartStopRecordingButton.setText("Start Recording")
        self.logic.setupScene()

    except ValueError:
      logging.info("Ports must have numeric values")


#
# RecordHerniaDataLogic
#

class RecordHerniaDataLogic(ScriptedLoadableModuleLogic):
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

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("RGBPort"):
      parameterNode.SetParameter("RGBPort", "18944")
    if not parameterNode.GetParameter("DepthPort"):
      parameterNode.SetParameter("DepthPort", "18945")
    if not parameterNode.GetParameter("SecondRGBPort"):
      parameterNode.SetParameter("SecondRGBPort", "18946")
    if not parameterNode.GetParameter("SecondDepthPort"):
      parameterNode.SetParameter("SecondDepthPort", "18947")

  def setParameterNode(self,parameterNode):
    self.parameterNode = parameterNode


  def setupOpenIGTLinkConnectors(self, rgbPort,depthPort,secondRGBPort,secondDepthPort):
    try:
      self.rgbConnectorNode = slicer.util.getNode('RGBConnector')
      self.rgbConnectorNode.SetTypeClient('localhost',int(rgbPort))
    except slicer.util.MRMLNodeNotFoundException:
      self.rgbConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      self.rgbConnectorNode.SetName('RGBConnector')
      slicer.mrmlScene.AddNode(self.rgbConnectorNode)
      self.rgbConnectorNode.SetTypeClient('localhost',int(rgbPort))
      logging.debug('RGB Connector Created')
    self.rgbConnectorNode.Start()

    try:
      self.depthConnectorNode = slicer.util.getNode('DepthConnector')
      self.depthConnectorNode.SetTypeClient('localhost',int(depthPort))
    except slicer.util.MRMLNodeNotFoundException:
      self.depthConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      self.depthConnectorNode.SetName('DepthConnector')
      slicer.mrmlScene.AddNode(self.depthConnectorNode)
      self.depthConnectorNode.SetTypeClient('localhost',int(depthPort))
      logging.debug('Depth Connector Created')
    self.depthConnectorNode.Start()

    try:
      self.secondRGBConnectorNode = slicer.util.getNode('SecondRGBConnector')
      self.secondRGBConnectorNode.SetTypeClient('localhost',int(secondRGBPort))
    except slicer.util.MRMLNodeNotFoundException:
      self.secondRGBConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      self.secondRGBConnectorNode.SetName('SecondRGBConnector')
      slicer.mrmlScene.AddNode(self.secondRGBConnectorNode)
      self.secondRGBConnectorNode.SetTypeClient('localhost',int(secondRGBPort))
      logging.debug('Second RGB Connector Created')
    self.secondRGBConnectorNode.Start()

    try:
      self.secondDepthConnectorNode = slicer.util.getNode('SecondDepthConnector')
      self.secondDepthConnectorNode.SetTypeClient('localhost',int(secondDepthPort))
    except slicer.util.MRMLNodeNotFoundException:
      self.secondDepthConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      self.secondDepthConnectorNode.SetName('SecondDepthConnector')
      slicer.mrmlScene.AddNode(self.secondDepthConnectorNode)
      self.secondDepthConnectorNode.SetTypeClient('localhost',int(secondDepthPort))
      logging.debug('Second Depth Connector Created')
    self.secondDepthConnectorNode.Start()

  def setupScene(self):
    self.setupOpenIGTLinkConnectors(18944,18945,18946,18947)

    try:
      self.rgbCamera1 = slicer.util.getNode("ImageRGB")
    except slicer.util.MRMLNodeNotFoundException:
      self.rgbCamera1 = slicer.vtkMRMLStreamingVolumeNode()
      self.rgbCamera1.SetName("ImageRGB_ImageRGB")
      slicer.mrmlScene.AddNode(self.rgbCamera1)
    self.setupVolumeResliceDriver(self.rgbCamera1,"Red")

    try:
      self.rgbCamera2 = slicer.util.getNode("Image1RGB")
    except slicer.util.MRMLNodeNotFoundException:
      self.rgbCamera2 = slicer.vtkMRMLStreamingVolumeNode()
      self.rgbCamera2.SetName("Image1RGB_Image1RGB")
      slicer.mrmlScene.AddNode(self.rgbCamera2)
    self.setupVolumeResliceDriver(self.rgbCamera2, "Yellow")

    try:
      self.depthCamera1 = slicer.util.getNode("ImageDEPTH")
    except slicer.util.MRMLNodeNotFoundException:
      self.depthCamera1 = slicer.vtkMRMLStreamingVolumeNode()
      self.depthCamera1.SetName("ImageDEPTH_ImageDEPT")
      slicer.mrmlScene.AddNode(self.depthCamera1)

    try:
      self.depthCamera2 = slicer.util.getNode("Image1DEPTH")
    except slicer.util.MRMLNodeNotFoundException:
      self.depthCamera2 = slicer.vtkMRMLStreamingVolumeNode()
      self.depthCamera2.SetName("Image1DEPTH_Image1DE")
      slicer.mrmlScene.AddNode(self.depthCamera2)



  def setupVolumeResliceDriver(self,cameraNode,sliceColor):

    layoutManager = slicer.app.layoutManager()
    slice = layoutManager.sliceWidget(sliceColor)
    sliceLogic = slice.sliceLogic()
    sliceLogic.GetSliceCompositeNode().SetBackgroundVolumeID(cameraNode.GetID())

    resliceLogic = slicer.modules.volumereslicedriver.logic()
    if resliceLogic:
      sliceNode = slicer.util.getNode('vtkMRMLSliceNode'+sliceColor)
      sliceNode.SetSliceResolutionMode(slicer.vtkMRMLSliceNode.SliceResolutionMatchVolumes)
      resliceLogic.SetDriverForSlice(cameraNode.GetID(), sliceNode)
      resliceLogic.SetModeForSlice(6, sliceNode)
      resliceLogic.SetFlipForSlice(False, sliceNode)
      # resliceLogic.SetRotationForSlice(180, yellowNode)
      sliceLogic.FitSliceToAll()

  def StartRecording(self,fileName):
    self.fileName = fileName + "-" + time.strftime("%Y%m%d-%H%M%S")
    self.recordingStartTime = vtk.vtkTimerLog.GetUniversalTime()
    self.herniaSequenceBrowserNode = slicer.vtkMRMLSequenceBrowserNode()
    self.startSequenceBrowserRecording(self.herniaSequenceBrowserNode)

  def StopRecording(self):
    self.stopSequenceBrowserRecording(self.herniaSequenceBrowserNode)
    self.saveRecording()
    self.removeRecordingFromScene()

  def startSequenceBrowserRecording(self, browserNode):
    if (browserNode is None):
      return

    # Indicate that this node was recorded, not loaded from file
    browserNode.SetName(slicer.mrmlScene.GetUniqueNameByString("Recording"))
    browserNode.SetAttribute("Recorded", "True")
    # Create and populate a sequence browser node if the recording started
    browserNode.SetScene(slicer.mrmlScene)
    slicer.mrmlScene.AddNode(browserNode)
    sequenceBrowserLogic = slicer.modules.sequences.logic()


    modifiedFlag = browserNode.StartModify()
    sequenceBrowserLogic.AddSynchronizedNode(None, self.rgbCamera1, browserNode)
    sequenceBrowserLogic.AddSynchronizedNode(None, self.depthCamera1, browserNode)
    sequenceBrowserLogic.AddSynchronizedNode(None, self.rgbCamera2, browserNode)
    sequenceBrowserLogic.AddSynchronizedNode(None, self.depthCamera2, browserNode)

    # Stop overwriting and saving changes to all nodes
    browserNode.SetRecording(None, True)
    browserNode.SetOverwriteProxyName(None, False)
    browserNode.SetSaveChanges(None, False)
    browserNode.EndModify(modifiedFlag)

    browserNode.SetRecordingActive(True)

    #self.StartRecordingSeekWidget.setMRMLSequenceBrowserNode(browserNode)

  def stopSequenceBrowserRecording(self, browserNode):
    if (browserNode is None):
      return
    browserNode.SetRecordingActive(False)
    browserNode.SetRecording( None, False )
    self.saveRecording()

  def saveRecording(self):
    savedScenesDirectory = self.parameterNode.GetParameter('SavedScenesDirectory')

    recordingCollection = slicer.mrmlScene.GetNodesByClass( "vtkMRMLSequenceBrowserNode" )
    for nodeNumber in range( recordingCollection.GetNumberOfItems() ):
      browserNode = recordingCollection.GetItemAsObject( nodeNumber )
      filename = self.fileName + os.extsep + "sqbr"
      filename = os.path.join( savedScenesDirectory, filename )
      slicer.util.saveNode(browserNode, filename)

  def removeRecordingFromScene(self):
    slicer.mrmlScene.Clear()



#
# RecordHerniaDataTest
#

class RecordHerniaDataTest(ScriptedLoadableModuleTest):
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