import os
import unittest
import numpy
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import subprocess
import pandas
import cv2

#
# DataCollection
#

class DataCollection(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Data Collection" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Deep Learn Live"]
    self.parent.dependencies = []
    self.parent.contributors = ["Rebecca Hisey (Perk Lab)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is a module to collect training images from videos"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# DataCollectionWidget
#

class DataCollectionWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.logic = DataCollectionLogic()

    self.moduleDir = os.path.dirname(slicer.modules.datacollection.path)
    datasetDirectory = os.path.join(self.moduleDir, os.pardir, "Datasets")
    try:
      os.listdir(datasetDirectory)
    except FileNotFoundError:
      os.mkdir(datasetDirectory)


    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #self.imageSaveDirectory = qt.QLineEdit("Select directory to save images")
    #parametersFormLayout.addRow(self.imageSaveDirectory)

    self.selectRecordingNodeComboBox = qt.QComboBox()
    self.selectRecordingNodeComboBox.addItems(["Select Image Node"])
    self.recordingNodes = slicer.util.getNodesByClass("vtkMRMLVolumeNode")
    recordingNodeNames = []
    for recordingNode in self.recordingNodes:
        recordingNodeNames.append(recordingNode.GetName())
    self.selectRecordingNodeComboBox.addItems(recordingNodeNames)
    parametersFormLayout.addRow(self.selectRecordingNodeComboBox)


    self.datasetSelector = ctk.ctkDirectoryButton()
    self.datasetSelector.directory = os.path.join(self.moduleDir,os.pardir,"Datasets")
    parametersFormLayout.addRow(self.datasetSelector)

    self.videoIDComboBox = qt.QComboBox()
    self.videoIDComboBox.addItems(["Select video ID","Create new video ID"])
    self.onDatasetSelected()
    parametersFormLayout.addRow(self.videoIDComboBox)

    self.imageSubtypeComboBox = qt.QComboBox()
    self.imageSubtypeComboBox.addItems(["Select image subtype (optional)","Create new image subtype"])
    parametersFormLayout.addRow(self.imageSubtypeComboBox)
    self.logic.setImageSubtype("")

    self.fileTypeComboBox = qt.QComboBox()
    self.fileTypeComboBox.addItems([".jpg",".png",".bmp",".tiff"])
    parametersFormLayout.addRow(self.fileTypeComboBox)
    self.fileType = self.fileTypeComboBox.currentText

    self.collectFromSequenceCheckBox = qt.QCheckBox("Collect from Sequence")
    self.collectingFromSequence = False
    parametersFormLayout.addRow(self.collectFromSequenceCheckBox)

    self.problemTypeComboBox = qt.QComboBox()
    self.problemTypeComboBox.addItems(["Select problem type","Classification","Detection","Segmentation"])
    parametersFormLayout.addRow(self.problemTypeComboBox)
    self.classificationFrame = qt.QFrame()
    self.classificationLayout()
    parametersFormLayout.addRow(self.classificationFrame)
    self.classificationFrame.visible = False
    self.detectionFrame = qt.QFrame()
    self.detectionLayout()
    parametersFormLayout.addRow(self.detectionFrame)
    self.detectionFrame.visible = False
    self.segmentationFrame = qt.QFrame()
    self.segmentationLayout()
    parametersFormLayout.addRow(self.segmentationFrame)
    self.segmentationFrame.visible = False

    #
    # Start/Stop Image Collection Button
    #
    self.startStopCollectingImagesButton = qt.QPushButton("Start Image Collection")
    self.startStopCollectingImagesButton.toolTip = "Collect images."
    self.startStopCollectingImagesButton.enabled = False
    parametersFormLayout.addRow(self.startStopCollectingImagesButton)


    self.infoLabel = qt.QLabel("")
    parametersFormLayout.addRow(self.infoLabel)
    # connections
    self.nodeAddedToSceneObserver = slicer.mrmlScene.AddObserver(slicer.mrmlScene.NodeAddedEvent,self.addNodesToRecordingCombobox)
    self.nodeRemovedFromSceneObserver = slicer.mrmlScene.AddObserver(slicer.mrmlScene.NodeRemovedEvent, self.removeNodesFromRecordingCombobox)
    self.fileTypeComboBox.connect('currentIndexChanged(int)',self.onFileTypeSelected)
    self.inputSegmentationSelector.connect('currentIndexChanged(int)', self.onSegmentationInputSelected)
    self.selectRecordingNodeComboBox.connect("currentIndexChanged(int)",self.onRecordingNodeSelected)
    self.datasetSelector.connect('directorySelected(QString)',self.onDatasetSelected)
    self.startStopCollectingImagesButton.connect('clicked(bool)', self.onStartStopCollectingImagesButton)
    self.videoIDComboBox.connect('currentIndexChanged(int)',self.onVideoIDSelected)
    self.imageSubtypeComboBox.connect('currentIndexChanged(int)',self.onImageSubtypeSelected)
    self.collectFromSequenceCheckBox.connect('stateChanged(int)',self.onCollectFromSequenceChecked)
    self.problemTypeComboBox.connect('currentIndexChanged(int)',self.onProblemTypeSelected)


    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Start/Stop Collecting Images Button state
    self.onSelect()
    try:
      self.webcamReference = slicer.util.getNode('Webcam_Reference')
    except slicer.util.MRMLNodeNotFoundException:
    #if not self.webcamReference:
      imageSpacing = [0.2, 0.2, 0.2]
      imageData = vtk.vtkImageData()
      imageData.SetDimensions(640, 480, 1)
      imageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
      thresholder = vtk.vtkImageThreshold()
      thresholder.SetInputData(imageData)
      thresholder.SetInValue(0)
      thresholder.SetOutValue(0)
      # Create volume node
      self.webcamReference = slicer.vtkMRMLVectorVolumeNode()
      self.webcamReference.SetName('Webcam_Reference')
      self.webcamReference.SetSpacing(imageSpacing)
      self.webcamReference.SetImageDataConnection(thresholder.GetOutputPort())
      # Add volume to scene
      slicer.mrmlScene.AddNode(self.webcamReference)
      displayNode = slicer.vtkMRMLVectorVolumeDisplayNode()
      slicer.mrmlScene.AddNode(displayNode)
      self.webcamReference.SetAndObserveDisplayNodeID(displayNode.GetID())

    self.webcamConnectorNode = self.createWebcamPlusConnector()
    self.webcamConnectorNode.Start()
    self.setupWebcamResliceDriver()

  def addNodesToRecordingCombobox(self,caller,eventID):
    self.recordingNodes = slicer.util.getNodesByClass("vtkMRMLVolumeNode")
    recordingNodeNames = []
    for recordingNode in self.recordingNodes:
      nodeName = recordingNode.GetName()
      if self.selectRecordingNodeComboBox.findText(nodeName) == -1:
        recordingNodeNames.append(nodeName)
    self.selectRecordingNodeComboBox.addItems(recordingNodeNames)

  def removeNodesFromRecordingCombobox(self,caller,eventID):
    self.recordingNodes = slicer.util.getNodesByClass("vtkMRMLVolumeNode")
    recordingNodeNames = []
    for recordingNode in self.recordingNodes:
      nodeName = recordingNode.GetName()
      if self.selectRecordingNodeComboBox.findText(nodeName) == -1:
        recordingNodeNames.append(nodeName)
    for i in range(1,self.selectRecordingNodeComboBox.count):
      if not self.selectRecordingNodeComboBox.itemText(i) in recordingNodeNames:
        self.selectRecordingNodeComboBox.removeItem(i)

  def classificationLayout(self):
    classificationFormLayout = qt.QFormLayout(self.classificationFrame)
    self.classificationLabellingMethodComboBox = qt.QComboBox()
    self.classificationLabellingMethodComboBox.addItems(["Unlabelled", "Single Label"])
    self.labellingMethod = self.classificationLabellingMethodComboBox.currentText
    classificationFormLayout.addWidget(self.classificationLabellingMethodComboBox)

    self.classificationLabelTypeLineEdit = qt.QLineEdit("Label Title")
    classificationFormLayout.addWidget(self.classificationLabelTypeLineEdit)
    self.classificationLabelTypeLineEdit.visible = False

    self.classificationLabelNameLineEdit = qt.QLineEdit("Label")
    classificationFormLayout.addWidget(self.classificationLabelNameLineEdit)
    self.classificationLabelNameLineEdit.visible = False

    self.autoLabelFilePathSelector = ctk.ctkPathLineEdit()
    self.autoLabelFilePathSelector.showBrowseButton = True
    self.autoLabelPath = os.path.join(self.moduleDir,os.pardir,"Datasets")
    self.autoLabelFilePathSelector.setCurrentPath(self.autoLabelPath)
    classificationFormLayout.addWidget(self.autoLabelFilePathSelector)
    self.autoLabelFilePathSelector.visible = False

    '''self.autoLabelFileNameLineEdit = qt.QLineEdit("Auto_Labels.csv")
    classificationFormLayout.addWidget(self.autoLabelFileNameLineEdit)
    self.autoLabelFileNameLineEdit.visible=False'''

    self.classificationLabellingMethodComboBox.connect('currentIndexChanged(int)', self.onLabellingMethodSelected)
    self.autoLabelFilePathSelector.connect('currentPathChanged(QString)',self.onAutoLabelFileChanged)
    #self.autoLabelFileNameLineEdit.connect('textChanged(QString)', self.onAutoLabelFileChanged)

  def detectionLayout(self):
    self.detectionFrame = qt.QFrame()
    detectionFormLayout = qt.QFormLayout(self.detectionFrame)
    self.detectionLabel = qt.QLabel()
    self.detectionLabel.setText("This problem type is not yet supported")
    detectionFormLayout.addWidget(self.detectionLabel)

  def segmentationLayout(self):
    self.segmentationFrame = qt.QFrame()
    segmentationFormLayout = qt.QFormLayout(self.segmentationFrame)
    self.segmentationLabellingMethodComboBox = qt.QComboBox()
    self.segmentationLabellingMethodComboBox.addItems(["Unlabelled","From Segmentation"])
    self.labellingMethod = self.segmentationLabellingMethodComboBox.currentText
    segmentationFormLayout.addWidget(self.segmentationLabellingMethodComboBox)

    self.segmentationLabelTypeLineEdit = qt.QLineEdit("Label Title")
    segmentationFormLayout.addWidget(self.segmentationLabelTypeLineEdit)
    self.segmentationLabelTypeLineEdit.visible = False

    self.inputSegmentationSelector = qt.QComboBox()
    self.inputSegmentationSelector.addItems(["Select Input Segmentation"])
    segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    segmentationNodeNames = []
    for segNode in segmentationNodes:
      segmentationNodeNames.append(segNode.GetName())
    self.inputSegmentationSelector.addItems(segmentationNodeNames)
    segmentationFormLayout.addWidget(self.inputSegmentationSelector)
    self.inputSegmentationSelector.visible = False

    self.segmentationLabellingMethodComboBox.connect('currentIndexChanged(int)', self.onLabellingMethodSelected)


  def createWebcamPlusConnector(self):
    try:
      webcamConnectorNode = slicer.util.getNode('WebcamPlusConnector')
    except slicer.util.MRMLNodeNotFoundException:
      webcamConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      webcamConnectorNode.SetName('WebcamPlusConnector')
      slicer.mrmlScene.AddNode(webcamConnectorNode)
      hostNamePort = "localhost:18944"
      [hostName, port] = hostNamePort.split(':')
      webcamConnectorNode.SetTypeClient(hostName, int(port))
      logging.debug('Webcam PlusConnector Created')
    return webcamConnectorNode

  def setupWebcamResliceDriver(self):
    # Setup the volume reslice driver for the webcam.
    self.webcamReference = slicer.util.getNode('Webcam_Reference')

    layoutManager = slicer.app.layoutManager()
    yellowSlice = layoutManager.sliceWidget('Yellow')
    yellowSliceLogic = yellowSlice.sliceLogic()
    yellowSliceLogic.GetSliceCompositeNode().SetBackgroundVolumeID(self.webcamReference.GetID())

    resliceLogic = slicer.modules.volumereslicedriver.logic()
    if resliceLogic:
      yellowNode = slicer.util.getNode('vtkMRMLSliceNodeYellow')
      yellowNode.SetSliceResolutionMode(slicer.vtkMRMLSliceNode.SliceResolutionMatchVolumes)
      resliceLogic.SetDriverForSlice(self.webcamReference.GetID(), yellowNode)
      resliceLogic.SetModeForSlice(6, yellowNode)
      resliceLogic.SetFlipForSlice(False, yellowNode)
      resliceLogic.SetRotationForSlice(180, yellowNode)
      yellowSliceLogic.FitSliceToAll()

  def cleanup(self):
    pass

  def openCreateNewDatasetWindow(self):
    self.createNewDatasetWidget = qt.QDialog()
    self.createNewDatasetWidget.setModal(True)
    self.createNewDatasetFrame = qt.QFrame(self.createNewDatasetWidget)
    self.createNewDatasetFrame.setFrameStyle(0x0006)
    self.createNewDatasetWidget.setWindowTitle('Create New Dataset')
    self.createNewDatasetPopupGeometry = qt.QRect()
    mainWindow = slicer.util.mainWindow()
    if mainWindow:
      mainWindowGeometry = mainWindow.geometry
      self.windowWidth = mainWindow.width * 0.35
      self.windowHeight = mainWindow.height * 0.35
      self.createNewDatasetPopupGeometry.setWidth(self.windowWidth)
      self.createNewDatasetPopupGeometry.setHeight(self.windowHeight)
      self.popupPositioned = False
      self.createNewDatasetWidget.setGeometry(self.createNewDatasetPopupGeometry)
      self.createNewDatasetFrame.setGeometry(self.createNewDatasetPopupGeometry)
      self.createNewDatasetWidget.move(mainWindow.width / 2.0 - self.windowWidth,
                                     mainWindow.height / 2 - self.windowHeight)
    self.createNewDatasetLayout = qt.QVBoxLayout()
    self.createNewDatasetLayout.setContentsMargins(12, 4, 4, 4)
    self.createNewDatasetLayout.setSpacing(4)

    self.createNewDatasetButtonLayout = qt.QFormLayout()
    self.createNewDatasetButtonLayout.setContentsMargins(12, 4, 4, 4)
    self.createNewDatasetButtonLayout.setSpacing(4)

    self.datasetNameLineEdit = qt.QLineEdit("Dataset Name")
    self.createNewDatasetButtonLayout.addRow(self.datasetNameLineEdit)

    self.createNewDatasetButton = qt.QPushButton("Add Dataset")
    self.createNewDatasetButtonLayout.addRow(self.createNewDatasetButton)

    self.errorLabel = qt.QLabel("")
    self.createNewDatasetButtonLayout.addRow(self.errorLabel)

    self.createNewDatasetButton.connect('clicked(bool)', self.onNewDatasetAdded)

    self.createNewDatasetLayout.addLayout(self.createNewDatasetButtonLayout)
    self.createNewDatasetFrame.setLayout(self.createNewDatasetLayout)

  def openCreateNewVideoIDWindow(self):
    self.createNewVideoIDWidget = qt.QDialog()
    self.createNewVideoIDWidget.setModal(True)
    self.createNewVideoIDFrame = qt.QFrame(self.createNewVideoIDWidget)
    self.createNewVideoIDFrame.setFrameStyle(0x0006)
    self.createNewVideoIDWidget.setWindowTitle('Create New Dataset')
    self.createNewVideoIDPopupGeometry = qt.QRect()
    mainWindow = slicer.util.mainWindow()
    if mainWindow:
      mainWindowGeometry = mainWindow.geometry
      self.windowWidth = mainWindow.width * 0.35
      self.windowHeight = mainWindow.height * 0.35
      self.createNewVideoIDPopupGeometry.setWidth(self.windowWidth)
      self.createNewVideoIDPopupGeometry.setHeight(self.windowHeight)
      self.popupPositioned = False
      self.createNewVideoIDWidget.setGeometry(self.createNewVideoIDPopupGeometry)
      self.createNewVideoIDFrame.setGeometry(self.createNewVideoIDPopupGeometry)
      self.createNewVideoIDWidget.move(mainWindow.width / 2.0 - self.windowWidth,
                                     mainWindow.height / 2 - self.windowHeight)
    self.createNewVideoIDLayout = qt.QVBoxLayout()
    self.createNewVideoIDLayout.setContentsMargins(12, 4, 4, 4)
    self.createNewVideoIDLayout.setSpacing(4)

    self.createNewVideoIDButtonLayout = qt.QFormLayout()
    self.createNewVideoIDButtonLayout.setContentsMargins(12, 4, 4, 4)
    self.createNewVideoIDButtonLayout.setSpacing(4)

    self.videoIDLineEdit = qt.QLineEdit("Video ID")
    self.createNewVideoIDButtonLayout.addRow(self.videoIDLineEdit)

    self.createNewVideoIDButton = qt.QPushButton("Add Video ID")
    self.createNewVideoIDButtonLayout.addRow(self.createNewVideoIDButton)

    self.videoIDErrorLabel = qt.QLabel("")
    self.createNewVideoIDButtonLayout.addRow(self.videoIDErrorLabel)

    self.createNewVideoIDButton.connect('clicked(bool)', self.onNewVideoIDAdded)

    self.createNewVideoIDLayout.addLayout(self.createNewVideoIDButtonLayout)
    self.createNewVideoIDFrame.setLayout(self.createNewVideoIDLayout)

  def openCreateNewImageSubtypeWindow(self):
    self.createNewImageSubtypeWidget = qt.QDialog()
    self.createNewImageSubtypeWidget.setModal(True)
    self.createNewImageSubtypeFrame = qt.QFrame(self.createNewImageSubtypeWidget)
    self.createNewImageSubtypeFrame.setFrameStyle(0x0006)
    self.createNewImageSubtypeWidget.setWindowTitle('Create New Image Subtype')
    self.createNewImageSubtypePopupGeometry = qt.QRect()
    mainWindow = slicer.util.mainWindow()
    if mainWindow:
      mainWindowGeometry = mainWindow.geometry
      self.windowWidth = mainWindow.width * 0.35
      self.windowHeight = mainWindow.height * 0.35
      self.createNewImageSubtypePopupGeometry.setWidth(self.windowWidth)
      self.createNewImageSubtypePopupGeometry.setHeight(self.windowHeight)
      self.popupPositioned = False
      self.createNewImageSubtypeWidget.setGeometry(self.createNewImageSubtypePopupGeometry)
      self.createNewImageSubtypeFrame.setGeometry(self.createNewImageSubtypePopupGeometry)
      self.createNewImageSubtypeWidget.move(mainWindow.width / 2.0 - self.windowWidth,
                                     mainWindow.height / 2 - self.windowHeight)
    self.createNewImageSubtypeLayout = qt.QVBoxLayout()
    self.createNewImageSubtypeLayout.setContentsMargins(12, 4, 4, 4)
    self.createNewImageSubtypeLayout.setSpacing(4)

    self.createNewImageSubtypeButtonLayout = qt.QFormLayout()
    self.createNewImageSubtypeButtonLayout.setContentsMargins(12, 4, 4, 4)
    self.createNewImageSubtypeButtonLayout.setSpacing(4)

    self.ImageSubtypeNameLineEdit = qt.QLineEdit("Image Subtype Name")
    self.createNewImageSubtypeButtonLayout.addRow(self.ImageSubtypeNameLineEdit)

    self.createNewImageSubtypeButton = qt.QPushButton("Add Image Subtype")
    self.createNewImageSubtypeButtonLayout.addRow(self.createNewImageSubtypeButton)

    self.errorLabel = qt.QLabel("")
    self.createNewImageSubtypeButtonLayout.addRow(self.errorLabel)

    self.createNewImageSubtypeButton.connect('clicked(bool)', self.onNewImageSubtypeAdded)

    self.createNewImageSubtypeLayout.addLayout(self.createNewImageSubtypeButtonLayout)
    self.createNewImageSubtypeFrame.setLayout(self.createNewImageSubtypeLayout)

  def onRecordingNodeSelected(self):
    if self.selectRecordingNodeComboBox.currentText != "Select Image Node":
      self.recordingNode = self.selectRecordingNodeComboBox.currentText

  def onDatasetSelected(self):
    if self.datasetSelector.directory == "Create New Dataset":
      try:
        self.createNewDatasetWidget.show()
      except AttributeError:
        self.openCreateNewDatasetWindow()
        self.createNewDatasetWidget.show()
    elif self.datasetSelector.directory != "Select Dataset":
      self.currentDatasetName = os.path.basename(self.datasetSelector.directory)
      self.videoPath = self.datasetSelector.directory
      self.addVideoIDsToComboBox()
    else:
      for i in range(2, self.videoIDComboBox.count + 1):
        self.videoIDComboBox.removeItem(i)


  def addVideoIDsToComboBox(self):
    for i in range(2,self.videoIDComboBox.count + 1):
      self.videoIDComboBox.removeItem(i)
    videoIDList = os.listdir(self.videoPath)
    self.videoIDList = [dir for dir in videoIDList if dir.rfind(".") == -1] #get only directories
    self.videoIDComboBox.addItems(self.videoIDList)

  def addImageSubtypesToComboBox(self):
    for i in range(2,self.imageSubtypeComboBox.count + 1):
      self.imageSubtypeComboBox.removeItem(i)
    if self.videoIDComboBox.currentText != "Select video ID":
      subtypePath = os.path.join(self.videoPath,self.videoIDComboBox.currentText)
      imageSubtypeList = os.listdir(subtypePath)
      self.imageSubtypeList = [dir for dir in imageSubtypeList if dir.rfind(".") == -1] #get only directories
      self.imageSubtypeComboBox.addItems(self.imageSubtypeList)

  def onNewDatasetAdded(self):
    self.currentDatasetName = self.datasetNameLineEdit.text
    try:
      datasetPath = os.path.join(self.moduleDir,os.pardir,"Datasets",self.currentDatasetName)
      os.mkdir(datasetPath)
      self.datasetSelector.addItems([self.currentDatasetName])
      datasetIndex = self.datasetSelector.findText(self.currentDatasetName)
      self.datasetSelector.currentIndex = datasetIndex
      self.createNewDatasetWidget.hide()
      self.datasetNameLineEdit.setText("Dataset Name")
      self.errorLabel.setText("")
    except WindowsError:
      self.datasetNameLineEdit.setText("Dataset Name")
      self.errorLabel.setText("A dataset with the name " + self.currentDatasetName + " already exists")


  def onNewVideoIDAdded(self):
    self.currentVideoID = self.videoIDLineEdit.text
    try:
      videoIDPath = os.path.join(self.datasetSelector.directory,self.currentVideoID)
      os.mkdir(videoIDPath)
      self.videoIDComboBox.addItems([self.currentVideoID])
      videoIDIndex = self.videoIDComboBox.findText(self.currentVideoID)
      self.videoIDComboBox.currentIndex = videoIDIndex
      self.createNewVideoIDWidget.hide()
      self.videoIDLineEdit.setText("Video ID")
      self.videoIDErrorLabel.setText("")
    except WindowsError:
      self.videoIDLineEdit.setText("Video ID")
      self.videoIDErrorLabel.setText("A video with ID " + self.currentVideoID + " already exists")

  def onNewImageSubtypeAdded(self):
    self.currentImageSubtypeName = self.ImageSubtypeNameLineEdit.text
    try:
      imageSubtypePath = os.path.join(self.datasetSelector.directory,self.currentVideoID,self.currentImageSubtypeName)
      os.mkdir(imageSubtypePath)
      self.imageSubtypeComboBox.addItems([self.currentImageSubtypeName])
      imageSubtypeIndex = self.imageSubtypeComboBox.findText(self.currentImageSubtypeName)
      self.imageSubtypeComboBox.currentIndex = imageSubtypeIndex
      self.createNewImageSubtypeWidget.hide()
      self.ImageSubtypeNameLineEdit.setText("Image subtype Name")
      self.errorLabel.setText("")
    except WindowsError:
      self.ImageSubtypeNameLineEdit.setText("Image subtype Name")
      self.errorLabel.setText("An image subtype with the name " + self.currentImageSubtypeName + " already exists")

  def onImageSubtypeSelected(self):
    self.currentImageSubtypeName = self.imageSubtypeComboBox.currentText
    if self.currentImageSubtypeName == "Create new image subtype":
      try:
        self.createNewImageSubtypeWidget.show()
      except AttributeError:
        self.openCreateNewImageSubtypeWindow()
        self.createNewImageSubtypeWidget.show()
    elif self.currentImageSubtypeName != "Select image subtype (optional)":
      self.logic.setImageSubtype(self.currentImageSubtypeName)
      self.csvFilePath = os.path.join(self.currentVideoIDFilePath, self.currentImageSubtypeName, self.currentVideoID + "_" + self.currentImageSubtypeName + "_Labels.csv")
    else:
      self.logic.setImageSubtype("")

  def onVideoIDSelected(self):
    if self.videoIDComboBox.currentText == "Create new video ID":
      try:
        self.createNewVideoIDWidget.show()
      except AttributeError:
        self.openCreateNewVideoIDWindow()
        self.createNewVideoIDWidget.show()
    elif self.videoIDComboBox.currentText != "Select video ID":
      self.currentVideoID = self.videoIDComboBox.currentText
      self.currentVideoIDFilePath = os.path.join(self.datasetSelector.directory,self.currentVideoID)
      self.startStopCollectingImagesButton.enabled = True
      self.csvFilePath = os.path.join(self.currentVideoIDFilePath, self.currentVideoID + "_Labels.csv")
      self.addImageSubtypesToComboBox()


  def onProblemTypeSelected(self):
    self.problemType = self.problemTypeComboBox.currentText
    if self.problemType == "Classification":
      if self.collectFromSequenceCheckBox.checked:
        autoFromFilePresent = self.classificationLabellingMethodComboBox.findText("Auto from file")
        if autoFromFilePresent == -1:
          self.classificationLabellingMethodComboBox.addItems(["Auto from file"])
        self.autoLabelPath = os.path.join(self.autoLabelFilePathSelector.currentPath)
      else:
        self.classificationLabellingMethodComboBox.removeItem(self.classificationLabellingMethodComboBox.findText("Auto from file"))
      self.classificationFrame.visible = True
      self.detectionFrame.visible = False
      self.segmentationFrame.visible = False
    elif self.problemType == "Detection":
      self.classificationFrame.visible = False
      self.detectionFrame.visible = True
      self.segmentationFrame.visible = False
    elif self.problemType == "Segmentation":
      self.classificationFrame.visible = False
      self.detectionFrame.visible = False
      self.segmentationFrame.visible = True

  def onAutoLabelFileChanged(self):
    self.autoLabelPath = os.path.join(self.autoLabelFilePathSelector.currentPath)
    return

  def onSelect(self):
    self.startStopCollectingImagesButton.enabled =  self.videoIDComboBox.currentText!= "Select video ID" and self.videoIDComboBox.currentText!= "Create new video ID" and self.selectRecordingNodeComboBox.currentText != "Select Image Node"


  def onStartStopCollectingImagesButton(self):
    try:
      self.imageLabels = pandas.read_csv(self.csvFilePath, index_col=0)
      # self.imageLabels.drop("Unnamed: 0",axis=1)
    except FileNotFoundError:
      self.imageLabels = pandas.DataFrame(columns=["FileName", "Time Recorded"])
    if self.startStopCollectingImagesButton.text == "Start Image Collection":
      self.collectingImages = False
      self.startStopCollectingImagesButton.setText("Stop Image Collection")
    else:
      self.collectingImages = True
      self.startStopCollectingImagesButton.setText("Start Image Collection")
    if self.labellingMethod == "Single Label":
      self.labelName = self.classificationLabelNameLineEdit.text
      self.labelType = self.classificationLabelTypeLineEdit.text
    elif self.labellingMethod == "Auto from file":
      self.labelName = None
      self.labelType = None
    elif self.labellingMethod == "From Segmentation":
      self.labelName = self.inputSegmentation
      self.labelType = self.inputSegmentation
    else:
      self.labelName = None
      self.labelType = None
    self.logic.startImageCollection(self.recordingNode, self.fileType, self.collectingImages, self.currentVideoID,self.currentVideoIDFilePath, self.imageLabels,self.csvFilePath,self.labellingMethod,self.collectingFromSequence,self.labelName,self.labelType,self.autoLabelPath)

  def onLabellingMethodSelected(self):
    if self.problemType == "Classification":
      self.labellingMethod = self.classificationLabellingMethodComboBox.currentText
      if self.labellingMethod == "Single Label":
        self.classificationLabelNameLineEdit.visible = True
        self.classificationLabelTypeLineEdit.visible = True
        self.autoLabelFilePathSelector.visible = False
        #self.autoLabelFileNameLineEdit.visible = False
      elif self.labellingMethod == "Auto from file":
        self.classificationLabelNameLineEdit.visible = False
        self.classificationLabelTypeLineEdit.visible = False
        self.autoLabelFilePathSelector.visible = True
        #self.autoLabelFileNameLineEdit.visible = True
      else:
        self.classificationLabelNameLineEdit.visible = False
        self.classificationLabelTypeLineEdit.visible = False
        self.autoLabelFilePathSelector.visible = False
        #self.autoLabelFileNameLineEdit.visible = False
    elif self.problemType == "Segmentation":
      self.labellingMethod = self.segmentationLabellingMethodComboBox.currentText
      if self.labellingMethod == "Unlabelled":
        self.inputSegmentationSelector.visible = False
      else:
        self.inputSegmentationSelector.visible = True
    else:
      self.labellingMethod = "Unlabelled"
    return

  def onFileTypeSelected(self):
    self.fileType = self.fileTypeComboBox.currentText

  def onSegmentationInputSelected(self):
    if self.inputSegmentationSelector.currentText != "Select Input Segmentation":
      self.inputSegmentation = self.inputSegmentationSelector.currentText

  def onCollectFromSequenceChecked(self):
    if self.collectFromSequenceCheckBox.checked:
      if self.problemTypeComboBox.currentText == "Classification":
        self.classificationLabellingMethodComboBox.addItems(["Auto from file"])
      self.collectingFromSequence = True
    else:
      self.classificationLabellingMethodComboBox.removeItem(self.classificationLabellingMethodComboBox.findText("Auto from file"))
      self.collectingFromSequence = False


#
# DataCollectionLogic
#

class DataCollectionLogic(ScriptedLoadableModuleLogic):
  def startImageCollection(self,recordingNode,fileType, imageCollectionStarted,videoID, videoIDFilePath, imageLabels, labelFilePath, labellingMethod = "Unlabelled", fromSequence = False,labelName = None,labelType = None,autolabelFilePath = None):
    try:
      # the module is in the python path
      import cv2
    except ImportError:
      # for the build directory, load from the file
      import imp, platform
      if platform.system() == 'Windows':
        cv2File = 'cv2.pyd'
        cv2Path = '../../../../OpenCV-build/lib/Release/' + cv2File
      else:
        cv2File = 'cv2.so'
        cv2Path = '../../../../OpenCV-build/lib/' + cv2File
      scriptPath = os.path.dirname(os.path.abspath(__file__))
      cv2Path = os.path.abspath(os.path.join(scriptPath, cv2Path))
      # in the build directory, this path should exist, but in the installed extension
      # it should be in the python pat, so only use the short file name
      if not os.path.isfile(cv2Path):
        cv2Path = cv2File
      cv2 = imp.load_dynamic('cv2', cv2File)
    self.recordingVolumeNode = slicer.util.getNode(recordingNode)
    self.fileType = fileType
    self.collectingImages = imageCollectionStarted
    self.continueRecording = not(self.collectingImages)
    self.videoID = videoID
    self.videoIDFilePath = videoIDFilePath
    self.labelName = labelName
    self.labelType = labelType
    self.imageLabels = imageLabels
    self.autoLabelFilePath = autolabelFilePath
    if (not self.labelType in self.imageLabels.columns) and self.labelType != None and not self.imageLabels.empty:
      self.imageLabels[self.labelType] = ['None' for i in range(self.imageLabels.index.max()+1)]
    '''if "Time Recorded" in self.imageLabels.columns:
      self.imageLabels = self.imageLabels.astype({'Time Recorded':'float64'})'''
    self.labelFilePath = labelFilePath
    self.labellingMethod = labellingMethod
    self.lastRecordedTime = 0.0
    if self.labellingMethod == "From Segmentation":
      self.segmentationNodeName = labelName
    self.fromSequence = fromSequence
    if self.labellingMethod == 'Auto from file':
      self.autoLabels = pandas.read_csv(self.autoLabelFilePath)
      self.labelType = self.autoLabels.columns[0]

    if self.collectingImages == False:
      if self.recordingVolumeNode.GetClassName() == "vtkMRMLStreamingVolumeNode":
        self.recordingVolumeNodeObserver = self.recordingVolumeNode.AddObserver(slicer.vtkMRMLStreamingVolumeNode.FrameModifiedEvent,self.onStartCollectingImages)
      elif self.recordingVolumeNode.GetClassName() == "vtkMRMLVectorVolumeNode":
        self.recordingVolumeNodeObserver = self.recordingVolumeNode.AddObserver(slicer.vtkMRMLVectorVolumeNode.ImageDataModifiedEvent, self.onStartCollectingImages)
      elif self.recordingVolumeNode.GetClassName() == "vtkMRMLScalarVolumeNode":
        self.recordingVolumeNodeObserver = self.recordingVolumeNode.AddObserver(slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent, self.onStartCollectingImages)
      else:
        logging.info(self.recordingVolumeNode.GetClassName() + " is not a supported recording volume type")
      logging.info("Start collecting images")

    else:
      self.recordingVolumeNode.RemoveObserver(self.recordingVolumeNodeObserver)
      self.recordingVolumeNodeObserver = None
      self.numImagesInFile = len(os.listdir(os.path.dirname(self.labelFilePath)))
      logging.info("Saved " + str(self.numImagesInFile) + " to directory : " + str(os.path.dirname(self.labelFilePath)))

    if self.fromSequence:
      try:
        if not self.finishedVideo:
          playWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserPlayWidget")
          playWidgetButtons = playWidget[0].findChildren('QPushButton')
          playWidgetButtons[2].click()
        else:
          logging.info("Video processing complete")
      except AttributeError:
        playWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserPlayWidget")
        playWidgetButtons = playWidget[0].findChildren('QPushButton')
        playWidgetButtons[2].click()

  def onStartCollectingImages(self,caller,eventID):
    import numpy
    try:
      # the module is in the python path
      import cv2
    except ModuleNotFoundError:
      # for the build directory, load from the file
      import imp, platform
      if platform.system() == 'Windows':
        cv2File = 'cv2.pyd'
        cv2Path = '../../../../OpenCV-build/lib/Release/' + cv2File
      else:
        cv2File = 'cv2.so'
        cv2Path = '../../../../OpenCV-build/lib/' + cv2File
      scriptPath = os.path.dirname(os.path.abspath(__file__))
      cv2Path = os.path.abspath(os.path.join(scriptPath, cv2Path))
      # in the build directory, this path should exist, but in the installed extension
      # it should be in the python pat, so only use the short file name
      if not os.path.isfile(cv2Path):
        cv2Path = cv2File
      cv2 = imp.load_dynamic('cv2', cv2File)
    if self.fromSequence:
      seekWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserSeekWidget")
      seekWidget = seekWidget[0]
      seekSlider = seekWidget.findChildren("QSlider")
      seekSlider = seekSlider[0]
      timeLabel = seekWidget.findChildren("QLabel")
      timeLabel = timeLabel[1]
      recordingTime = float(timeLabel.text)
      if seekSlider.value < seekSlider.maximum and recordingTime >= self.lastRecordedTime:
        self.continueRecording = True
      else:
        self.continueRecording = False
    # Get the vtkImageData as an np.array.
    if (not self.fromSequence) or self.continueRecording:
      allFiles = os.listdir(os.path.dirname(self.labelFilePath))
      imgFiles = [x for x in allFiles if not "segmentation" in x and not ".csv" in x]
      self.numImagesInFile = len(imgFiles)
      logging.info(self.numImagesInFile)
      imData = self.getVtkImageDataAsOpenCVMat(self.recordingVolumeNode.GetName())
      fileName = self.videoID + "_" + self.imageSubtype + "_" + str(self.numImagesInFile).zfill(5) + self.fileType
      if self.fromSequence:
        if not self.imageLabels.empty:
          dataframeEntry = self.imageLabels.loc[(abs(self.imageLabels["Time Recorded"] - recordingTime) <= 0.2)]
          if not dataframeEntry.empty:
            addingtoexisting = True
            entry = dataframeEntry.index[-1]
          else:
            addingtoexisting = False
            imagePath = os.path.dirname(self.labelFilePath)
            cv2.imwrite(os.path.join(imagePath, fileName), imData)
        else:
          addingtoexisting = False
          imagePath = os.path.dirname(self.labelFilePath)
          cv2.imwrite(os.path.join(imagePath, fileName), imData)
      if self.labellingMethod == "Unlabelled":
        if self.fromSequence:
          recordingTime = timeLabel.text
          self.lastRecordedTime = float(recordingTime)
          self.imageLabels = self.imageLabels.append({'FileName': fileName,'Time Recorded':recordingTime}, ignore_index=True)
        else:
          self.imageLabels = self.imageLabels.append({'FileName':fileName},ignore_index=True)
      else:
        if self.labellingMethod == 'Auto from file':
          self.labelType = self.autoLabels.columns[0]
          self.labelName = self.getClassificationLabelFromFile()
        elif self.labellingMethod == 'From Segmentation':
          (labelImData, self.labelName) = self.getSegmentationLabel(fileName)
          imagePath = os.path.dirname(self.labelFilePath)
          cv2.imwrite(os.path.join(imagePath,self.labelName),labelImData)
        if self.fromSequence:
          recordingTime = timeLabel.text
          self.lastRecordedTime = float(recordingTime)
          if not addingtoexisting:
            self.imageLabels = self.imageLabels.append({'FileName': fileName, 'Time Recorded':self.lastRecordedTime, self.labelType: self.labelName},
                                                     ignore_index=True)
          else:
            self.imageLabels[self.labelType].iloc[entry] = self.labelName
        else:
          self.imageLabels = self.imageLabels.append({'FileName': fileName, self.labelType: self.labelName},
                                                     ignore_index=True)
      self.imageLabels.to_csv(self.labelFilePath)
    elif not self.continueRecording:
      playWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserPlayWidget")
      playWidgetButtons = playWidget[0].findChildren('QPushButton')
      playWidgetButtons[2].click()
      self.finishedVideo = True

  def setImageSubtype(self,subtypeName):
    self.imageSubtype = subtypeName

  def getClassificationLabelFromFile(self):
    seekWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserSeekWidget")
    seekWidget = seekWidget[0]
    timeStamp = seekWidget.findChildren("QLabel")
    timeStamp = float(timeStamp[1].text)
    task = self.autoLabels.loc[(self.autoLabels["Start"]<=timeStamp) & (self.autoLabels["End"]>timeStamp)]
    labelName = task.iloc[0][self.labelType]
    return labelName

  def getSegmentationLabel(self,fileName):
    segmentationNode = slicer.util.getNode(self.segmentationNodeName)
    labelMapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    imageNode = slicer.util.getNode(self.recordingVolumeNode.GetName())
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, labelMapNode)
    labelmapOriented_Reference = self.generateMergedLabelmapInReferenceGeometry(segmentationNode, imageNode)
    slicer.modules.segmentations.logic().CreateLabelmapVolumeFromOrientedImageData(labelmapOriented_Reference,
                                                                                   labelMapNode)
    labelMapNode.SetAndObserveTransformNodeID(imageNode.GetParentTransformNode().GetID())

    labelFileName = fileName.replace(".","_segmentation.")
    output_array = slicer.util.arrayFromVolume(labelMapNode)
    #output_array = numpy.where(output_array > 0, 255, output_array)
    output_array[output_array > 0] = 255
    shape = output_array.shape
    if shape[0] == 1:
      shape = [shape[1],shape[2],shape[0]]
    output_array = output_array.reshape(shape)
    #output_array = cv2.flip(output_array,1)
    slicer.mrmlScene.RemoveNode(labelMapNode)
    return (output_array,labelFileName)

  def generateMergedLabelmapInReferenceGeometry(self,segmentationNode, referenceVolumeNode):
    if segmentationNode is None:
      logging.error("Invalid segmentation node")
      return None
    if referenceVolumeNode is None:
      logging.error("Invalid reference volume node")
      return None

    # Get reference geometry in the segmentation node's coordinate system
    referenceGeometry_Reference = slicer.vtkOrientedImageData()  # reference geometry in reference node coordinate system
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
                                                           segmentationNode.GetParentTransformNode(),
                                                           referenceGeometryToSegmentationTransform)
      slicer.vtkOrientedImageDataResample.TransformOrientedImage(referenceGeometry_Segmentation,
                                                                 referenceGeometryToSegmentationTransform, True)

    # Generate shared labelmap for the exported segments in segmentation coordinates
    sharedImage_Segmentation = slicer.vtkOrientedImageData()
    if (not segmentationNode.GenerateMergedLabelmapForAllSegments(sharedImage_Segmentation, 0, None)):
      logging.error("ExportSegmentsToLabelmapNode: Failed to generate shared labelmap")
      return None

    # Transform shared labelmap to reference geometry coordinate system
    segmentationToReferenceGeometryTransform = referenceGeometryToSegmentationTransform.GetInverse()
    segmentationToReferenceGeometryTransform.Update()
    slicer.vtkOrientedImageDataResample.ResampleOrientedImageToReferenceOrientedImage(sharedImage_Segmentation,
                                                                                      referenceGeometry_Reference,
                                                                                      mergedLabelmap_Reference,
                                                                                      False, False,
                                                                                      segmentationToReferenceGeometryTransform)

    return mergedLabelmap_Reference

  def getVtkImageDataAsOpenCVMat(self, volumeNodeName):
    try:
      # the module is in the python path
      import cv2
    except ModuleNotFoundError:
      # for the build directory, load from the file
      import imp, platform
      if platform.system() == 'Windows':
        cv2File = 'cv2.pyd'
        cv2Path = '../../../../OpenCV-build/lib/Release/' + cv2File
      else:
        cv2File = 'cv2.so'
        cv2Path = '../../../../OpenCV-build/lib/' + cv2File
      scriptPath = os.path.dirname(os.path.abspath(__file__))
      cv2Path = os.path.abspath(os.path.join(scriptPath, cv2Path))
      # in the build directory, this path should exist, but in the installed extension
      # it should be in the python pat, so only use the short file name
      if not os.path.isfile(cv2Path):
        cv2Path = cv2File
      cv2 = imp.load_dynamic('cv2', cv2File)
    cameraVolume = slicer.util.getNode(volumeNodeName)
    image = cameraVolume.GetImageData()
    shape = list(cameraVolume.GetImageData().GetDimensions())
    components = image.GetNumberOfScalarComponents()
    if components > 1:
      shape.reverse()
      shape.append(components)
      shape.remove(1)
    imageMat = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape)
    if components > 1:
      imageMat = cv2.cvtColor(imageMat, cv2.COLOR_RGB2BGR)
    return imageMat

class DataCollectionTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)