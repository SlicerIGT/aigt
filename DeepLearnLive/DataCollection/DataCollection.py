import os
import unittest
import numpy
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import subprocess
import time
import datetime
from sequenceSpinBox import sequenceSpinBox

try:
  import cv2
except ModuleNotFoundError:
  slicer.util.pip_install("opencv-python")
  import cv2

try:
  import pandas
except ModuleNotFoundError:
  slicer.util.pip_install("pandas")
  import pandas

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
    self.labelSequenceCollapsibleButton = ctk.ctkCollapsibleButton()
    self.labelSequenceCollapsibleButton.text = "Label Sequence"
    self.layout.addWidget(self.labelSequenceCollapsibleButton)

    labelSequenceLayout = qt.QFormLayout(self.labelSequenceCollapsibleButton)
    self.setupLabelSequenceLayout(labelSequenceLayout)
    self.labelSequenceCollapsibleButton.collapsed = True

    reviewLabelsCollapsibleButton = ctk.ctkCollapsibleButton()
    reviewLabelsCollapsibleButton.text = "Review Labels"
    self.layout.addWidget(reviewLabelsCollapsibleButton)
    reviewLabelsCollapsibleButton.collapsed = True

    reviewLabelsFormLayout = qt.QFormLayout(reviewLabelsCollapsibleButton)
    self.setupReviewLayout(reviewLabelsFormLayout)

    #self.infoLabel = qt.QLabel("")
    #parametersFormLayout.addRow(self.infoLabel)

    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Export Images"
    self.layout.addWidget(parametersCollapsibleButton)
    parametersCollapsibleButton.collapsed = False

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    self.selectRecordingNodeComboBox = qt.QComboBox()
    self.selectRecordingNodeComboBox.addItems(["Select Image Node"])
    self.recordingNodes = slicer.util.getNodesByClass("vtkMRMLVolumeNode")
    recordingNodeNames = []
    for recordingNode in self.recordingNodes:
        recordingNodeNames.append(recordingNode.GetName())
    self.selectRecordingNodeComboBox.addItems(recordingNodeNames)
    parametersFormLayout.addRow("Select Image Node: ",self.selectRecordingNodeComboBox)

    self.datasetSelector = ctk.ctkDirectoryButton()
    self.datasetSelector.directory = os.path.join(self.moduleDir,os.pardir,"Datasets")
    parametersFormLayout.addRow("Select dataset: ",self.datasetSelector)

    self.videoIDComboBox = qt.QComboBox()
    self.videoIDComboBox.addItems(["Select video ID","Create new video ID"])
    self.onDatasetSelected()
    parametersFormLayout.addRow("Select videoID: ",self.videoIDComboBox)

    self.imageSubtypeComboBox = qt.QComboBox()
    self.imageSubtypeComboBox.addItems(["Select image subtype (optional)","Create new image subtype"])
    parametersFormLayout.addRow("Select subtype: ",self.imageSubtypeComboBox)
    self.logic.setImageSubtype("")

    self.fileTypeComboBox = qt.QComboBox()
    self.fileTypeComboBox.addItems([".jpg",".png",".bmp",".tiff"])
    parametersFormLayout.addRow("File type: ",self.fileTypeComboBox)
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

    #Create a live webcam stream
    try:
      self.webcamReference = slicer.util.getNode('Live_Webcam_Reference')
    except slicer.util.MRMLNodeNotFoundException:
      imageSpacing = [0.2, 0.2, 0.2]
      imageData = vtk.vtkImageData()
      imageData.SetDimensions(640, 480, 3)
      imageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
      thresholder = vtk.vtkImageThreshold()
      thresholder.SetInputData(imageData)
      thresholder.SetInValue(0)
      thresholder.SetOutValue(0)
      # Create volume node
      self.webcamReference = slicer.vtkMRMLVectorVolumeNode()
      self.webcamReference.SetName('Live_Webcam_Reference')
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

  def setupLabelSequenceLayout(self,layout):
    """
    lays out the module to label a sequence, user chooses image node and problem type
    :param layout: the interface layout depending on specified problem type
    :return:
    """
    self.selectSequenceBox = slicer.qMRMLNodeComboBox()
    self.selectSequenceBox.selectNodeUponCreation = True
    self.selectSequenceBox.nodeTypes = ["vtkMRMLSequenceBrowserNode"]
    self.selectSequenceBox.addEnabled = False
    self.selectSequenceBox.removeEnabled = False
    self.selectSequenceBox.editEnabled = False
    self.selectSequenceBox.renameEnabled = False
    self.selectSequenceBox.noneEnabled = False
    self.selectSequenceBox.showHidden = False
    self.selectSequenceBox.showChildNodeTypes = False
    self.selectSequenceBox.setMRMLScene(slicer.mrmlScene)
    layout.addRow(self.selectSequenceBox)

    self.logic.setSequenceBrowserNode(self.selectSequenceBox.currentNode())

    self.selectImageNodeBox = qt.QComboBox()
    self.selectImageNodeBox.addItem("Select Image Node")
    layout.addRow(self.selectImageNodeBox)

    self.labelproblemTypeComboBox = qt.QComboBox()
    self.labelproblemTypeComboBox.addItems(["Select problem type","Classification","Segmentation","Detection"])
    layout.addRow(self.labelproblemTypeComboBox)

    self.selectLabelTypeCombobox = slicer.qMRMLNodeComboBox()
    self.selectLabelTypeCombobox.selectNodeUponCreation = False
    self.selectLabelTypeCombobox.nodeTypes = ["vtkMRMLTextNode"]
    self.selectLabelTypeCombobox.addEnabled = True
    self.selectLabelTypeCombobox.removeEnabled = False
    self.selectLabelTypeCombobox.editEnabled = False
    self.selectLabelTypeCombobox.renameEnabled = True
    self.selectLabelTypeCombobox.noneEnabled = False
    self.selectLabelTypeCombobox.showHidden = False
    self.selectLabelTypeCombobox.showChildNodeTypes = False
    self.selectLabelTypeCombobox.setMRMLScene(slicer.mrmlScene)
    layout.addRow("Label node: ",self.selectLabelTypeCombobox)

    self.labelTableWidget = qt.QTableWidget(0,3)
    self.labelTableWidget.setColumnWidth(0, 80)
    self.labelTableWidget.setColumnWidth(1,300)
    self.labelTableWidget.setColumnWidth(2,50)
    self.labelTableWidget.setHorizontalHeaderLabels(["Label name","Range","Remove"])
    self.labelTableWidget.setGeometry(qt.QRect(0,0,10,10))

    layout.addRow(self.labelTableWidget)
    self.labelTableWidget.visible = False

    self.addRowButton = qt.QPushButton("Add row")
    self.addRowButton.visible = False

    layout.addRow(self.addRowButton)

    self.removeRowButton = qt.QPushButton("Remove selected rows")
    self.removeRowButton.visible = False

    layout.addRow(self.removeRowButton)

    self.labelSequenceButton = qt.QPushButton("Label sequence")
    layout.addRow(self.labelSequenceButton)

    self.addImageNodeNamesToComboBox()

    self.selectSequenceBox.connect("currentNodeChanged(vtkMRMLNode*)",self.onSequenceBrowserSelected)
    self.labelproblemTypeComboBox.connect('currentIndexChanged(int)',self.onlabelProblemTypeSelected)
    self.selectImageNodeBox.connect('currentIndexChanged(int)',self.onImageNodeSelected)
    self.selectLabelTypeCombobox.connect("currentNodeChanged(vtkMRMLNode*)",self.onLabelNodeSelected)
    self.removeRowButton.connect('clicked(bool)',self.onRemoveRowClicked)
    self.addRowButton.connect('clicked(bool)',self.onAddRowClicked)
    self.labelSequenceButton.connect('clicked(bool)',self.onLabelSequenceClicked)

    if self.selectLabelTypeCombobox.currentNode() != None:
      self.onLabelNodeSelected()

    self.logic.setLabelNode(self.selectLabelTypeCombobox.currentNode())

  def onlabelProblemTypeSelected(self):
    """
    adjusts module layout based on the specified problem
    :return:
    """
    if self.labelproblemTypeComboBox.currentText == "Classification":
      self.labelTableWidget.visible = True
      self.addRowButton.visible = True
      self.removeRowButton.visible = True
      self.labelSequenceButton.visible = True
      self.selectLabelTypeCombobox.nodeTypes = ["vtkMRMLTextNode"]
    elif self.labelproblemTypeComboBox.currentText == "Segmentation":
      self.labelTableWidget.visible = False
      self.addRowButton.visible = False
      self.removeRowButton.visible = False
      self.labelSequenceButton.visible = False
      self.selectLabelTypeCombobox.nodeTypes = ["vtkMRMLSegmentationNode"]
    elif self.labelproblemTypeComboBox.currentText == "Detection":
      self.labelTableWidget.visible = False
      self.addRowButton.visible = False
      self.removeRowButton.visible = False
      self.labelSequenceButton.visible = False
    else:
      self.labelTableWidget.visible = False
      self.addRowButton.visible = False
      self.removeRowButton.visible = False
      self.labelSequenceButton.visible = False
      print(self.labelproblemTypeComboBox.currentText)

  def onSequenceBrowserSelected(self):
    """
    gets the sequence browser nodes
    :return:
    """
    self.sequenceBrowserNode = self.selectSequenceBox.currentNode()
    self.logic.setSequenceBrowserNode(self.selectSequenceBox.currentNode())
    self.addImageNodeNamesToComboBox()

  def addImageNodeNamesToComboBox(self):
    """
    retrieves all images nodes from the selected sequence
    :return:
    """
    imageNodeNames = []
    proxyNodes = vtk.vtkCollection()
    try:
      self.selectSequenceBox.currentNode().GetAllProxyNodes(proxyNodes)
      numProxyNodes = proxyNodes.GetNumberOfItems()
      for i in range(numProxyNodes):
        proxyNode = proxyNodes.GetItemAsObject(i)
        proxyNodeClass = proxyNode.GetClassName()
        if proxyNodeClass in ["vtkMRMLVectorVolumeNode", "vtkMRMLScalarVolumeNode", "vtkMRMLStreamingVolumeNode"]:
          imageNodeNames.append(proxyNode.GetName())
      for i in range(self.selectImageNodeBox.count - 1, 0, -1):
        self.selectImageNodeBox.removeItem(i)
      self.selectImageNodeBox.addItems(imageNodeNames)
    except AttributeError:
      pass

  def onImageNodeSelected(self):
    """
    reads the user's selected image node
    :return:
    """
    if self.selectImageNodeBox.currentText != "Select Image Node":
      self.logic.setImageNode(self.selectImageNodeBox.currentText)
      self.sequenceNode = self.selectSequenceBox.currentNode().GetSequenceNode(slicer.util.getNode(self.selectImageNodeBox.currentText))
      self.numDataNodes = self.sequenceNode.GetNumberOfDataNodes()

  def onLabelNodeSelected(self):
    # 
    for i in range(self.labelTableWidget.rowCount, -1, -1): # QUESTION: why -1
      self.labelTableWidget.removeRow(i)
    if self.selectLabelTypeCombobox.currentNode() != None:
      self.logic.setLabelNode(self.selectLabelTypeCombobox.currentNode())
      currentNodeName = self.selectLabelTypeCombobox.currentNode().GetName()
      if "Sequence" in currentNodeName:
        sequenceBrowserNode = slicer.util.getNode(currentNodeName + " browser")
        self.sequenceNode = sequenceBrowserNode.GetSequenceNode(self.selectLabelTypeCombobox.currentNode())
        self.numDataNodes = self.sequenceNode.GetNumberOfDataNodes()
        if self.sequenceNode.GetNumberOfDataNodes() != 0:
          labels = self.logic.getLabelsFromSequence(self.sequenceNode)
          numTableRows = labels.index.max() + 1
          maxValues = []
          for i in range(numTableRows):
            minValue = self.sequenceNode.GetItemNumberFromIndexValue(str(labels["Start"][i]))
            maxValue = self.sequenceNode.GetItemNumberFromIndexValue(str(labels["End"][i]))
            maxValues.append(maxValue)
            labelTypeName = currentNodeName.replace("-Sequence","")
            rowLabel = labels[labelTypeName][i]
            self.onAddRowClicked(minValue,rowLabel)
          maxLabels = [x for x in self.labelTableWidget.findChildren('QDoubleSpinBox')]
          for i in range(len(maxLabels)):
            maxLabels[i].setValue(maxValues[i])

  def onAddRowClicked(self,minValue=None,label=None):
    """
    add a slider row to the labelling interface
    :param minValue: minimum numerical value for slider
    :param label: the label with which to create the row
    :return:
    """
    numRows = self.labelTableWidget.rowCount
    self.labelTableWidget.insertRow(numRows)
    labelNameLineEdit = qt.QLineEdit()
    if label == None:
      labelNameLineEdit.setText("Label")
    else:
      labelNameLineEdit.setText(label)
    labelNameLineEdit.setObjectName("labelEdit_"+str(numRows))
    self.labelTableWidget.setCellWidget(numRows, 0, labelNameLineEdit)

    RangeSelectorWidget = qt.QWidget()
    RangeSelectorWidget.setObjectName("RangeSelectorWidget_"+str(numRows))
    layout = qt.QHBoxLayout()
    minTimeLabel = qt.QLabel("0.00 s")
    if minValue != None and minValue > 0:
      minTimeLabel.text = "%.2f" % float(self.sequenceNode.GetNthIndexValue(self.numDataNodes-1))
    minTimeLabel.setObjectName("minLabel_"+str(numRows))
    maxTimeLabel = qt.QLabel("%.2f" % float(self.sequenceNode.GetNthIndexValue(self.numDataNodes-1)))
    maxTimeLabel.setObjectName("maxLabel_"+str(numRows))
    #maxTimeBox = qt.QDoubleSpinBox()
    maxTimeBox = sequenceSpinBox()
    valueRange = ["%.2f" % float(self.sequenceNode.GetNthIndexValue(x)) for x in range(self.numDataNodes)]
    maxTimeBox.setValueRange(valueRange)
    maxTimeBox.setMinimum(0)
    maxTimeBox.setMaximum(self.numDataNodes-1)
    maxTimeBox.setSuffix(" s")
    maxTimeBox.setValue(int(self.numDataNodes-1))
    maxLineEdit = maxTimeBox.findChildren("QLineEdit")
    maxLineEdit[0].setReadOnly(True)
    rangeSlider = ctk.ctkRangeSlider()
    rangeSlider.setOrientation(qt.Qt.Horizontal)
    rangeSlider.setObjectName("rangeSlider_"+str(numRows))
    rangeSlider.setMinimum(0)
    rangeSlider.setMaximum(self.numDataNodes-1)
    layout.addWidget(minTimeLabel)
    layout.addWidget(rangeSlider)
    layout.addWidget(maxTimeBox)
    RangeSelectorWidget.setLayout(layout)
    self.labelTableWidget.setCellWidget(numRows, 1, RangeSelectorWidget)

    removeButton = qt.QCheckBox()
    removeButton.setGeometry(qt.QRect(0, 0, 10, 10))
    self.labelTableWidget.setCellWidget(numRows, 2, removeButton)
    maxTimeBox.connect('valueChanged(double)',self.maxValueChanged)
    rangeSlider.connect('positionsChanged(int,int)',self.onSliderPositionChanged)
    rangeSlider.connect('sliderReleased()', self.onSliderReleased)
    rangeSlider.connect('sliderPressed()', self.onSliderClicked)


  def onSliderClicked(self):
    maxLabels = [x for x in self.labelTableWidget.findChildren('QDoubleSpinBox')]
    for widget in maxLabels:
      widget.blockSignals(True)

  def onSliderReleased(self):
    # 
    maxLabels = [x for x in self.labelTableWidget.findChildren('QDoubleSpinBox')]
    for widget in maxLabels:
      widget.blockSignals(False)

  def maxValueChanged(self):
    minLabels = [x for x in self.labelTableWidget.findChildren('QLabel') if "minLabel" in x.name]
    rangeSliders = self.labelTableWidget.findChildren('ctkRangeSlider')
    maxLabels = [x for x in self.labelTableWidget.findChildren('QDoubleSpinBox')]
    for widget in rangeSliders:
      widget.blockSignals(True)
    for i in range(len(maxLabels)):
      if i < len(maxLabels)-1:
        minLabels[i+1].text = maxLabels[i].text
        rangeSliders[i + 1].minimumPosition = maxLabels[i].value
      rangeSliders[i].maximumPosition = maxLabels[i].value
    for widget in rangeSliders:
      widget.blockSignals(False)

  def onSliderPositionChanged(self):
    """
    change min or max val of label whose slider changed
    :return:
    """
    rangeSliders = self.labelTableWidget.findChildren('ctkRangeSlider')
    minLabels = [x for x in self.labelTableWidget.findChildren('QLabel') if "minLabel" in x.name]
    maxLabels = [x for x in self.labelTableWidget.findChildren('QDoubleSpinBox')]
    rangeSliders[0].minimumPosition = 0
    maxLabels[0].setValue(rangeSliders[0].maximumPosition)
    for i in range(1,len(rangeSliders)):
      minLabels[i].text = "%5.2f s" % float(self.sequenceNode.GetNthIndexValue(rangeSliders[i-1].maximumPosition))
      rangeSliders[i].minimumPosition = rangeSliders[i-1].maximumPosition
      maxLabels[i].setValue(rangeSliders[i].maximumPosition)
    rangeSliders[len(rangeSliders)-1].maximumPosition = self.numDataNodes
    maxLabels[len(rangeSliders)-1].setValue(self.numDataNodes-1)


  def onRemoveRowClicked(self):
    """
    deletes a row from the interface
    :return:
    """
    numRows = self.labelTableWidget.rowCount
    rowsToRemove = []
    for i in range(numRows):
      checkBox = self.labelTableWidget.cellWidget(i,2)
      if checkBox.checked:
        rowsToRemove.append(i)
    rowsToRemove.sort(reverse=True)
    for row in rowsToRemove:
      self.labelTableWidget.removeRow(row)
    rangeSliders = self.labelTableWidget.findChildren('ctkRangeSlider')
    minLabels = [x for x in self.labelTableWidget.findChildren('QLabel') if "minLabel" in x.name]
    maxLabels = [x for x in self.labelTableWidget.findChildren('QLabel') if "maxLabel" in x.name]
    rangeSliders[0].minimumPosition = 0
    maxLabels[0].text = "%5.2f s" % float(self.sequenceNode.GetNthIndexValue(rangeSliders[0].maximumPosition))
    for i in range(1, len(rangeSliders)):
      minLabels[i].text = "%5.2f s" % float(self.sequenceNode.GetNthIndexValue(rangeSliders[i - 1].maximumPosition))
      rangeSliders[i].minimumPosition = rangeSliders[i - 1].maximumPosition
      maxLabels[i].text = "%5.2f s" % float(self.sequenceNode.GetNthIndexValue(rangeSliders[i].maximumPosition))
    rangeSliders[len(rangeSliders) - 1].maximumPosition = self.numDataNodes
    maxLabels[len(rangeSliders) - 1].text = "%5.2f s" % float(self.sequenceNode.GetNthIndexValue(self.numDataNodes - 1))


  def onLabelSequenceClicked(self):
    """
    labels sequence
    :return:
    """
    self.logic.labelSequence(self.labelTableWidget)

  def setupReviewLayout(self,layout):
    """
    sets up the interface to review labels
    :param layout: the GUI layout
    :return:
    """
    self.reviewNodeSelector = slicer.qMRMLNodeComboBox()
    self.reviewNodeSelector.selectNodeUponCreation = True
    self.reviewNodeSelector.nodeTypes = (
    ("vtkMRMLScalarVolumeNode", "vtkMRMLVectorVolumeNode", "vtkMRMLStreamingVolumeNode"))
    self.reviewNodeSelector.addEnabled = True
    self.reviewNodeSelector.removeEnabled = False
    self.reviewNodeSelector.editEnabled = True
    self.reviewNodeSelector.renameEnabled = True
    self.reviewNodeSelector.noneEnabled = False
    self.reviewNodeSelector.showHidden = False
    self.reviewNodeSelector.showChildNodeTypes = False
    self.reviewNodeSelector.setMRMLScene(slicer.mrmlScene)
    layout.addRow("Image Node: ", self.reviewNodeSelector)

    self.reviewDatasetSelector = ctk.ctkDirectoryButton()
    self.reviewDatasetSelector.directory = os.path.join(self.moduleDir, os.pardir, "Datasets")
    layout.addRow("Select dataset: ", self.reviewDatasetSelector)

    self.reviewVideoIDComboBox = qt.QComboBox()
    self.reviewVideoIDComboBox.addItem("Select video ID")
    layout.addRow(self.reviewVideoIDComboBox)

    self.reviewImageSubtypeBox = qt.QComboBox()
    self.reviewImageSubtypeBox.addItem("Select image subtype (optional)")
    self.reviewImageNode = self.reviewNodeSelector.currentNode()
    layout.addRow(self.reviewImageSubtypeBox)

    self.reviewLabelTypeBox = qt.QComboBox()
    self.reviewLabelTypeBox.addItems(["Select label type","Display all"])
    layout.addRow(self.reviewLabelTypeBox)

    self.startReviewButton = qt.QPushButton("Start Review")
    layout.addRow(self.startReviewButton)
    self.reviewing = False

    self.reviewNodeSelector.connect('currentNodeChanged(vtkMRMLNode*)',self.onReviewImageNodeSelected)
    self.reviewDatasetSelector.connect('directorySelected(QString)',self.onReviewDatasetSelected)
    self.reviewVideoIDComboBox.connect('currentIndexChanged(int)', self.onReviewVideoIDSelected)
    self.reviewImageSubtypeBox.connect('currentIndexChanged(int)',self.onReviewImageSubtypeSelected)
    self.reviewLabelTypeBox.connect('currentIndexChanged(int)', self.onReviewLabelTypeSelected)
    self.startReviewButton.connect('clicked(bool)',self.onStartReviewClicked)

  def onReviewImageNodeSelected(self):
    self.reviewImageNode = self.reviewNodeSelector.currentNode()

  def onReviewDatasetSelected(self):
    self.reviewDataset = self.reviewDatasetSelector.directory
    self.reviewVideoIDComboBox.currentIndex = 0
    self.reviewImageSubtypeBox.currentIndex = 0
    self.reviewLabelTypeBox.currentIndex = 0
    for i in range(self.reviewVideoIDComboBox.count-1, 0, -1):
      self.reviewVideoIDComboBox.removeItem(i)
    videoIDs = [x for x in os.listdir(self.reviewDataset) if not '.' in x]
    self.reviewVideoIDComboBox.addItems(videoIDs)

  def onReviewVideoIDSelected(self):
    self.reviewVideoID = self.reviewVideoIDComboBox.currentText
    self.reviewImageSubtypeBox.currentIndex = 0
    self.reviewLabelTypeBox.currentIndex = 0
    for i in range(self.reviewImageSubtypeBox.count - 1, 0, -1):
      self.reviewImageSubtypeBox.removeItem(i)
    for i in range(self.reviewLabelTypeBox.count - 1, 1, -1):
      self.reviewLabelTypeBox.removeItem(i)
    if self.reviewVideoID != "Select video ID":
      imageSubtypes = [x for x in os.listdir(os.path.join(self.reviewDataset,self.reviewVideoID)) if not '.' in x]
      if imageSubtypes == []:
        self.labelFileName = self.reviewVideoID + '_Labels.csv'
        self.labelCSV = pandas.read_csv(os.path.join(self.reviewDataset,self.reviewVideoID,self.labelFileName))
        labelCSVHeadings = self.labelCSV.columns
        if "Time Recorded" in labelCSVHeadings:
          self.labelTypes = labelCSVHeadings[3:]
        else:
          self.labelTypes = labelCSVHeadings[2:]
        self.reviewLabelTypeBox.addItems(self.labelTypes)
      else:
        self.reviewImageSubtypeBox.addItems(imageSubtypes)

  def onReviewImageSubtypeSelected(self):
    self.reviewImageSubtype = self.reviewImageSubtypeBox.currentText
    self.reviewLabelTypeBox.currentIndex = 0
    for i in range(self.reviewLabelTypeBox.count - 1, 1, -1):
      self.reviewLabelTypeBox.removeItem(i)
    if self.reviewImageSubtype != "Select image subtype (optional)":
      self.labelFileName = self.reviewVideoID+'_'+self.reviewImageSubtype+'_Labels.csv'
      self.labelCSV = pandas.read_csv(os.path.join(self.reviewDataset, self.reviewVideoID,self.reviewImageSubtype,self.labelFileName))
      labelCSVHeadings = self.labelCSV.columns
      if "Time_Recorded" in labelCSVHeadings:
        self.labelTypes = labelCSVHeadings[3:]
      else:
        self.labelTypes = labelCSVHeadings[2:]
      self.reviewLabelTypeBox.addItems(self.labelTypes)

  def onReviewLabelTypeSelected(self):
    if self.reviewLabelTypeBox.currentText == "Display all":
      self.labelType = self.labelTypes
    elif self.reviewLabelTypeBox.currentText != "Select label type":
      self.labelType = [self.reviewLabelTypeBox.currentText]

  def onStartReviewClicked(self):
    if not self.reviewing:
      self.logic.StartReview(self.labelCSV,self.labelType,self.reviewImageNode)
      self.reviewing = True
      self.startReviewButton.setText("Stop Review")
    else:
      self.logic.StopReview()
      self.reviewing = False
      self.startReviewButton.setText("Start Review")

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
    for i in range(self.selectRecordingNodeComboBox.count-1,0,-1):
      if not self.selectRecordingNodeComboBox.itemText(i) in recordingNodeNames:
        self.selectRecordingNodeComboBox.removeItem(i)

  def classificationLayout(self):
    """
    sets up the module interface for classification
    :return:
    """
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

    self.classificationSequenceSelector = slicer.qMRMLNodeComboBox()
    self.classificationSequenceSelector.selectNodeUponCreation = True
    self.classificationSequenceSelector.nodeTypes = ["vtkMRMLSequenceNode"]
    self.classificationSequenceSelector.addEnabled = False
    self.classificationSequenceSelector.removeEnabled = False
    self.classificationSequenceSelector.editEnabled = False
    self.classificationSequenceSelector.renameEnabled = False
    self.classificationSequenceSelector.noneEnabled = False
    self.classificationSequenceSelector.showHidden = False
    self.classificationSequenceSelector.showChildNodeTypes = False
    self.classificationSequenceSelector.setMRMLScene(slicer.mrmlScene)
    classificationFormLayout.addWidget(self.classificationSequenceSelector)
    self.classificationSequenceSelector.visible = False
    self.classificationLabelNode = self.classificationSequenceSelector.currentNode()

    self.classificationLabellingMethodComboBox.connect('currentIndexChanged(int)', self.onLabellingMethodSelected)
    self.autoLabelFilePathSelector.connect('currentPathChanged(QString)',self.onAutoLabelFileChanged)
    self.classificationSequenceSelector.connect('currentNodeChanged(vtkMRMLNode*)',self.onClassificationLabelNodeSelected)

  def detectionLayout(self):
    """
    sets up the module interface for detection (not yet supported)
    :return:
    """
    self.detectionFrame = qt.QFrame()
    detectionFormLayout = qt.QFormLayout(self.detectionFrame)
    self.detectionLabel = qt.QLabel()
    self.detectionLabel.setText("This problem type is not yet supported")
    detectionFormLayout.addWidget(self.detectionLabel)

  def segmentationLayout(self):
    """
    sets up the module interface for segmentation
    :return:
    """
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
    """
    creates the webcam connection with slicer through Plus server
    :return: webcam connector node
    """
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
    """
    Setup the volume reslice driver for the webcam.
    :return:
    """
    self.webcamReference = slicer.util.getNode('Live_Webcam_Reference')

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
    """
    opens a window to create a new dataset
    :return:
    """
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
    """
    creates new popup window through which user can specify a new video ID
    :return:
    """
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
    """
    opens popup window through which user can specify a new image subtype
    :return:
    """
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
    """
    sets recording node based on user choice from combo box
    :return:
    """
    if self.selectRecordingNodeComboBox.currentText != "Select Image Node":
      self.recordingNode = self.selectRecordingNodeComboBox.currentText


  def onDatasetSelected(self):
      for i in range(self.videoIDComboBox.count,2,-1):
        self.videoIDComboBox.removeItem(i)
      self.currentDatasetName = os.path.basename(self.datasetSelector.directory)
      self.videoPath = self.datasetSelector.directory
      self.addVideoIDsToComboBox()


  def addVideoIDsToComboBox(self):
    """
    when a new video ID is created, add it to the combo box
    :return:
    """
    for i in range(2,self.videoIDComboBox.count + 1):
      self.videoIDComboBox.removeItem(i)
    videoIDList = os.listdir(self.videoPath)
    self.videoIDList = [dir for dir in videoIDList if dir.rfind(".") == -1] #get only directories
    self.videoIDComboBox.addItems(self.videoIDList)

  def addImageSubtypesToComboBox(self):
    """
    when a new image subtype is created, add it to the combo box
    :return:
    """
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
    """
    prompts user to add a new video ID throught the popup window
    :return:
    """
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
    """
    prompts user to add a new image subtype throught the popup window
    :return:
    """
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
    """
    when image subtype is selected, create/open the window for user to input the subtype and make the csv file's path match the subtype
    :return:
    """
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
    """
    when image subtype is selected, create/open the window for user to input the ID and make the csv file's path match the ID
    :return:
    """
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
    """
    setup combo boxes to match choices that correspond to the selected problem type
    :return:
    """
    self.problemType = self.problemTypeComboBox.currentText
    if self.problemType == "Classification":
      if self.collectFromSequenceCheckBox.checked:
        autoFromFilePresent = self.classificationLabellingMethodComboBox.findText("Auto from file")
        if autoFromFilePresent == -1:
          self.classificationLabellingMethodComboBox.addItems(["Auto from file"])
        fromSequencePresent = self.classificationLabellingMethodComboBox.findText("From Sequence")
        if fromSequencePresent == -1:
          self.classificationLabellingMethodComboBox.addItems(["From Sequence"])
        self.autoLabelPath = os.path.join(self.autoLabelFilePathSelector.currentPath)
      else:
        self.classificationLabellingMethodComboBox.removeItem(self.classificationLabellingMethodComboBox.findText("Auto from file"))
        self.classificationLabellingMethodComboBox.removeItem(self.classificationLabellingMethodComboBox.findText("From Sequence"))
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
    """
    update the path to auto label file
    :return:
    """
    self.autoLabelPath = os.path.join(self.autoLabelFilePathSelector.currentPath)
    self.logic.setAutolabelPath(self.autoLabelPath)

  def onSelect(self):
    self.startStopCollectingImagesButton.enabled =  self.videoIDComboBox.currentText!= "Select video ID" and self.videoIDComboBox.currentText!= "Create new video ID" and self.selectRecordingNodeComboBox.currentText != "Select Image Node"


  def onStartStopCollectingImagesButton(self):
    """
    stops/starts collecting images and updates the button text accordingly
    :return:
    """
    self.logic.setRecordingNode(self.recordingNode)
    self.logic.setDatasetNameAndPath(self.videoPath,self.currentDatasetName)
    self.logic.setVideoIDAndPath(self.currentVideoID, self.currentVideoIDFilePath)
    self.logic.setAutolabelPath(self.autoLabelPath)
    self.logic.setLabellingMethod(self.labellingMethod)

    try:
      self.imageLabels = pandas.read_csv(self.csvFilePath, index_col=0)
    except FileNotFoundError:
      self.imageLabels = pandas.DataFrame(columns=["FileName", "Time Recorded"])
    if self.startStopCollectingImagesButton.text == "Start Image Collection":
      self.collectingImages = False
      self.startStopCollectingImagesButton.setText("Stop Image Collection")
    else:
      self.collectingImages = True
      self.startStopCollectingImagesButton.setText("Start Image Collection")
    if self.labellingMethod == "Single Label":
      self.logic.setLabelName(self.classificationLabelNameLineEdit.text)
      self.logic.setLabelType(self.classificationLabelTypeLineEdit.text)
    elif self.labellingMethod == "From Segmentation":
      self.logic.setLabelName(self.inputSegmentation)
      self.logic.setLabelType(self.inputSegmentation)
    elif self.labellingMethod == "From Sequence":
      self.logic.setLabelSequence(self.classificationLabelNode)
      self.logic.setLabelType(self.classificationLabelTypeLineEdit.text)
    self.logic.startImageCollection (self.collectingImages, self.imageLabels,self.csvFilePath)

  def onLabellingMethodSelected(self):
    """
    updates the interface choices based on which labeling method and problem type
    :return:
    """
    if self.problemType == "Classification":
      self.labellingMethod = self.classificationLabellingMethodComboBox.currentText
      if self.labellingMethod == "Single Label":
        self.classificationLabelNameLineEdit.visible = True
        self.classificationLabelTypeLineEdit.visible = True
        self.autoLabelFilePathSelector.visible = False
        self.classificationSequenceSelector.visible = False
      elif self.labellingMethod == "Auto from file":
        self.classificationLabelNameLineEdit.visible = False
        self.classificationLabelTypeLineEdit.visible = False
        self.autoLabelFilePathSelector.visible = True
        self.classificationSequenceSelector.visible = False
      elif self.labellingMethod == "From Sequence":
        self.classificationLabelTypeLineEdit.visible = True
        self.classificationLabelNameLineEdit.visible = False
        self.autoLabelFilePathSelector.visible = False
        self.classificationSequenceSelector.visible = True
      else:
        self.classificationLabelNameLineEdit.visible = False
        self.classificationLabelTypeLineEdit.visible = False
        self.autoLabelFilePathSelector.visible = False
        self.classificationSequenceSelector.visible = False
    elif self.problemType == "Segmentation":
      self.labellingMethod = self.segmentationLabellingMethodComboBox.currentText
      if self.labellingMethod == "Unlabelled":
        self.inputSegmentationSelector.visible = False
      else:
        self.inputSegmentationSelector.visible = True
    else:
      self.labellingMethod = "Unlabelled"
    self.logic.setLabellingMethod(self.labellingMethod)

  def onFileTypeSelected(self):
    """
    sets file type for output images i.e jpg, png
    :return:
    """
    self.fileType = self.fileTypeComboBox.currentText
    self.logic.setFileType(self.fileType)

  def onSegmentationInputSelected(self):
    if self.inputSegmentationSelector.currentText != "Select Input Segmentation":
      self.inputSegmentation = self.inputSegmentationSelector.currentText
      self.logic.setInputSegmentationNode(self.inputSegmentation)

  def onCollectFromSequenceChecked(self):
    """
    collects training images from sequence
    :return:
    """
    if self.collectFromSequenceCheckBox.checked:
      if self.problemTypeComboBox.currentText == "Classification":
        self.classificationLabellingMethodComboBox.addItems(["Auto from file","From Sequence"])
      self.collectingFromSequence = True
    else:
      self.classificationLabellingMethodComboBox.removeItem(self.classificationLabellingMethodComboBox.findText("Auto from file"))
      self.classificationLabellingMethodComboBox.removeItem(self.classificationLabellingMethodComboBox.findText("From Sequence"))
      self.collectingFromSequence = False
    self.logic.setCollectingFromSequence(self.collectingFromSequence)

  def onClassificationLabelNodeSelected(self):
    self.classificationLabelNode = self.classificationSequenceSelector.currentNode()
#
# DataCollectionLogic
#

class DataCollectionLogic(ScriptedLoadableModuleLogic):
  def __init__(self):
    self.imageSubtype = ''
    self.fileType = '.jpg'
    self.fromSequence = False
    self.labellingMethod = "Unlabelled"
    self.labelType = None
    self.labelName = None
    self.finishedVideo = False
    self.dataSetName = None
    self.videoID = None
    self.recordingVolumeNode = None
    self.autoLabelFilePath = None
    self.segmentationNodeName = None
    self.labelSequenceNode = None

  def labelSequence(self,labelTable):
    """
    creates a new label data node with the set labels
    :param labelTable: the table used for labelling the sequence
    :return:
    """
    self.labelTable = labelTable
    self.labels = self.getLabels(self.labelTable)
    self.sequenceNode = self.sequenceBrowserNode.GetSequenceNode(self.imageNode)
    try:
      self.labelSequenceNode = slicer.util.getNode(self.labelNode.GetName() + "-Sequence")
    except slicer.util.MRMLNodeNotFoundException:
      self.labelSequenceNode = slicer.vtkMRMLSequenceNode()
      self.labelSequenceNode.SetName(self.labelNode.GetName() + "-Sequence")
      slicer.mrmlScene.AddNode(self.labelSequenceNode)
    for i in range(self.sequenceNode.GetNumberOfDataNodes()):
      index = self.sequenceNode.GetNthIndexValue(i)
      indexFloat = round(float(index),2)
      newTextNode = slicer.vtkMRMLTextNode()
      newTextNode.SetName(self.labelNode.GetName()+"-Sequence_"+str(i).zfill(4))
      label = self.labels.loc[(self.labels["Start"] <= indexFloat) & (self.labels["End"] > indexFloat)]
      if label.empty:
        maxIndex = self.labels.index.max()
        print(self.labels)
        labelName = self.labels[self.labelNode.GetName()][maxIndex]
      else:
        labelName = label.iloc[0][self.labelNode.GetName()]
      newTextNode.SetText(labelName)
      self.labelSequenceNode.SetDataNodeAtValue(newTextNode,index)
      if i != self.labelSequenceNode.GetNumberOfDataNodes():
        logging.info("Created data node " + str(i+1) + " with label " + labelName)
      #logging.info(str(self.labelSequenceNode.GetNumberOfDataNodes()) + " / " + str(self.sequenceNode.GetNumberOfDataNodes()))
    #self.sequenceBrowserNode.AddSynchronizedSequenceNodeID(self.labelSequenceNode.GetID())

  def getLabels(self,labelTable):
    """
    retrives the labels from the slider rows
    :param labelTable: labels from GUI
    :return: labels for the sequence in a pandas dataframe
    """
    labels = pandas.DataFrame(columns=[self.labelNode.GetName(),"Start","End"])
    lineEdits = [x for x in labelTable.findChildren("QLineEdit") if "labelEdit" in x.name]
    minLabels = [x for x in labelTable.findChildren('QLabel') if "minLabel" in x.name]
    maxLabels = [x for x in labelTable.findChildren('QDoubleSpinBox')]
    for i in range(len(maxLabels)):
      labelName = lineEdits[i].text
      minValue = minLabels[i].text
      minValue = minValue.replace(" s","")
      minValue = float(minValue)
      maxValue = maxLabels[i].text
      maxValue = maxValue.replace(" s", "")
      maxValue = float(maxValue)
      labels = labels.append({self.labelNode.GetName():labelName,"Start":minValue,"End":maxValue},ignore_index=True)
    return labels


  def setSequenceBrowserNode(self,sequenceBrowser):
    """
    sets sequence browser node
    :param sequenceBrowser:
    :return:
    """
    self.sequenceBrowserNode = sequenceBrowser

  def setImageNode(self,imageNodeName):
    """
    sets image node
    :param imageNodeName: name of the image node
    :return:
    """
    self.imageNode = slicer.util.getNode(imageNodeName)

  def setLabelNode(self,labelNode):
    """
    set label node
    :param labelNode:
    :return:
    """
    self.labelNode = labelNode

  def startImageCollection(self, imageCollectionStarted, imageLabels, labelFilePath):
    """
     begins to separate sequence into individual frames
    :param imageCollectionStarted: bool, denotes if the image collection has started
    :param imageLabels: the labels in a pandas dataframe
    :param labelFilePath: the directory of the label file
    :return:
    """
    self.collectingImages = imageCollectionStarted
    self.continueRecording = not(self.collectingImages)
    self.imageLabels = imageLabels
    self.labelFilePath = labelFilePath

    self.lastRecordedTime = 0.0

    if self.labellingMethod == "From Sequence":
      self.exportImagesFromSequence()
    elif self.labellingMethod == "From Segmentation" and self.fromSequence:
      self.exportSegmentationsFromSequence()
    else:
      if self.labellingMethod == 'Auto from file':
        self.autoLabels = pandas.read_csv(self.autoLabelFilePath)
        self.labelType = self.autoLabels.columns[0]
      if (not self.labelType in self.imageLabels.columns) and self.labelType != None and not self.imageLabels.empty:
        if self.labellingMethod == "Auto from file":
          self.imageLabels = self.labelExistingEntries(self.imageLabels,self.autoLabels)
        elif "Time Recorded" in self.imageLabels.columns:
          self.imageLabels[self.labelType] = ['None' for i in range(self.imageLabels.index.max()+1)]
        else:
          logging.info("Cannot relabel images recorded from live sequence")

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
          pass
          '''playWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserPlayWidget")
          playWidgetButtons = playWidget[0].findChildren('QPushButton')
          playWidgetButtons[2].click()'''

  def onStartCollectingImages(self,caller,eventID):
    """
    :param caller:
    :param eventID:
    :return:
    """
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
      else:
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
        elif self.labellingMethod == "From Sequence":
          labelDataNode = self.labelSequenceNode.GetDataNodeAtValue(str(recordingTime))
          self.labelName = labelDataNode.GetText()
        if self.fromSequence:
          recordingTime = timeLabel.text
          self.lastRecordedTime = float(recordingTime)
          if not addingtoexisting:
            self.imageLabels = self.imageLabels.append({'FileName': fileName, 'Time Recorded':self.lastRecordedTime, self.labelType: self.labelName},
                                                     ignore_index=True)
          else:
            #self.imageLabels[self.labelType].iloc[entry] = self.labelName
            #self.imageLabels[self.labelType].iloc._setitem_with_indexer(entry, self.labelName)
            self.imageLabels.loc[entry,self.labelType] = self.labelName
        else:
          self.imageLabels = self.imageLabels.append({'FileName': fileName, self.labelType: self.labelName},
                                                     ignore_index=True)
      self.imageLabels.to_csv(self.labelFilePath)
    elif not self.continueRecording:
      playWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserPlayWidget")
      playWidgetButtons = playWidget[0].findChildren('QPushButton')
      playWidgetButtons[2].click()
      self.finishedVideo = True

  def exportImagesFromSequence(self):
    """
    export the sequence frames as individual frames
    :return:
    """
    sequenceName = self.recordingVolumeNode.GetName().split(sep="_")
    sequenceNode = slicer.util.getFirstNodeByName(sequenceName[0])
    if sequenceNode == None or sequenceNode.GetClassName() != 'vtkMRMLSequenceNode':
      sequenceNodeID = sequenceNode.GetID()
      IDNumbers = [x for x in sequenceNodeID if x.isnumeric()]
      sequenceID = 'vtkMRMLSequenceNode'
      for i in IDNumbers:
        sequenceID += str(i)
      sequenceNode = slicer.util.getNode(sequenceID)

    numDataNodes = sequenceNode.GetNumberOfDataNodes()
    addingToExisting = False
    all_labels = self.getLabelsFromSequence()
    if (not self.labelType in self.imageLabels.columns) and not self.imageLabels.empty:
      self.imageLabels[self.labelType] = ['None' for i in range(self.imageLabels.index.max() + 1)]
      addingToExisting = True
    elif not self.imageLabels.empty:
      addingToExisting = True
    prevTimeRecorded = 0
    for i in range(numDataNodes):
      logging.info(str(i) + " / " + str(numDataNodes) + " written")
      dataNode = sequenceNode.GetNthDataNode(i)
      timeRecorded = float(sequenceNode.GetNthIndexValue(i))
      roundedtimeRecorded = "%.2f" % timeRecorded
      if timeRecorded - prevTimeRecorded > 0.01:
        prevTimeRecorded = timeRecorded
        roundedtimeRecorded = float(roundedtimeRecorded)
        if self.labellingMethod == "From Sequence":
          labelName = self.getClassificationLabelFromSequence(all_labels,timeRecorded)
        if addingToExisting:
          entry = self.imageLabels.loc[(self.imageLabels["Time Recorded"] == roundedtimeRecorded)]
          if entry.empty:
            entry = self.imageLabels.loc[(abs(self.imageLabels["Time Recorded"] - roundedtimeRecorded) <= 0.05)]
          for j in entry.index:
            self.imageLabels.loc[j, self.labelType] = labelName
        else:
          imData = self.getVtkImageDataAsOpenCVMat(dataNode,True)
          if self.imageSubtype == '':
            fileName = self.videoID + "_" + str(i).zfill(5) + self.fileType
          else:
            fileName = self.videoID + "_" + self.imageSubtype + "_" + str(i).zfill(5) + self.fileType
          imagePath = os.path.dirname(self.labelFilePath)
          cv2.imwrite(os.path.join(imagePath, fileName), imData)
          self.imageLabels = self.imageLabels.append({"FileName":fileName,"Time Recorded":timeRecorded,self.labelType:labelName},ignore_index=True)
      prevTimeRecorded = timeRecorded
    self.imageLabels.to_csv(self.labelFilePath)

  def exportSegmentationsFromSequence(self):
    sequenceName = self.recordingVolumeNode.GetName()
    print(sequenceName)
    sequenceNode = slicer.util.getFirstNodeByName(sequenceName)
    print(sequenceNode.GetClassName())
    if sequenceNode == None or sequenceNode.GetClassName() != 'vtkMRMLSequenceNode':
      sequenceNodeID = sequenceNode.GetID()
      IDNumbers = [x for x in sequenceNodeID if x.isnumeric()]
      sequenceID = 'vtkMRMLSequenceNode'
      for i in IDNumbers:
        sequenceID += str(i)
      sequenceNode = slicer.util.getNode(sequenceID)

    numDataNodes = sequenceNode.GetNumberOfDataNodes()
    addingToExisting = False
    if (not self.labelType in self.imageLabels.columns) and not self.imageLabels.empty:
      self.imageLabels[self.labelType] = ['None' for i in range(self.imageLabels.index.max() + 1)]
      addingToExisting = True
    elif not self.imageLabels.empty:
      addingToExisting = True
    prevTimeRecorded = 0
    for i in range(numDataNodes):
      logging.info(str(i) + " / " + str(numDataNodes) + " written")
      dataNode = sequenceNode.GetNthDataNode(i)
      timeRecorded = float(sequenceNode.GetNthIndexValue(i))
      roundedtimeRecorded = "%.2f" % timeRecorded
      if timeRecorded - prevTimeRecorded > 0.01:
        prevTimeRecorded = timeRecorded
        roundedtimeRecorded = float(roundedtimeRecorded)
        if self.imageSubtype == '':
          fileName = self.videoID + "_" + str(i).zfill(5) + self.fileType
        else:
          fileName = self.videoID + "_" + self.imageSubtype + "_" + str(i).zfill(5) + self.fileType
        labelName = self.getSegmentationLabelFromSequence(timeRecorded,i,fileName)
        if addingToExisting:
          entry = self.imageLabels.loc[(self.imageLabels["Time Recorded"] == roundedtimeRecorded)]
          if entry.empty:
            entry = self.imageLabels.loc[(abs(self.imageLabels["Time Recorded"] - roundedtimeRecorded) <= 0.05)]
          for j in entry.index:
            self.imageLabels.loc[j, self.labelType] = labelName
        else:
          imData = self.getVtkImageDataAsOpenCVMat(dataNode, True)
          imagePath = os.path.dirname(self.labelFilePath)
          cv2.imwrite(os.path.join(imagePath, fileName), imData)
          self.imageLabels = self.imageLabels.append(
            {"FileName": fileName, "Time Recorded": timeRecorded, self.labelType: labelName}, ignore_index=True)
      prevTimeRecorded = timeRecorded
    self.imageLabels.to_csv(self.labelFilePath)

  def getSegmentationLabelFromSequence(self, timeRecorded,index,fileName):
    seekWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserSeekWidget")
    seekWidget = seekWidget[0]
    seekSlider = seekWidget.findChildren("QSlider")
    seekSlider = seekSlider[0]
    timeLabel = seekWidget.findChildren("QLabel")
    timeLabel = timeLabel[1]
    recordingTime = float(timeLabel.text)
    if seekSlider.value <= seekSlider.maximum and recordingTime >= self.lastRecordedTime:
      seekSlider.setValue(index)
      slicer.mrmlScene.Modified()
      segImage, labelFileName = self.getSegmentationLabel(fileName)
      imagePath = os.path.dirname(self.labelFilePath)
      cv2.imwrite(os.path.join(imagePath, labelFileName), segImage)
    return labelFileName

  def getClassificationLabelFromSequence(self,labels,timeRecorded):
    label = labels.loc[(labels["Start"] <= timeRecorded) & (labels["End"] > timeRecorded)]
    if label.empty:
      maxIndex = labels.index.max()
      labelName = labels[self.labelType][maxIndex]
    else:
      labelName = label.iloc[0][self.labelType]
    return labelName

  def getLabelsFromSequence(self,labelSequenceNode=None):
    """
    retrieves label data from a sequence node
    :param labelSequenceNode: QUESTION: why default to None?
    :return: retrieved labels
    """
    if labelSequenceNode != None:
      self.labelSequenceNode = labelSequenceNode
      self.labelType = labelSequenceNode.GetName().replace("-Sequence","")
    labels = pandas.DataFrame(columns = [self.labelType,"Start","End"])
    numDataNodes = self.labelSequenceNode.GetNumberOfDataNodes()
    startIndex = float(self.labelSequenceNode.GetNthIndexValue(0))
    startLabel = self.labelSequenceNode.GetNthDataNode(0).GetText()
    endIndex = float(self.labelSequenceNode.GetNthIndexValue(0))
    for i in range(1,numDataNodes):
      currentLabel = self.labelSequenceNode.GetNthDataNode(i).GetText()
      if currentLabel == startLabel and i != numDataNodes-1:
        endIndex = float(self.labelSequenceNode.GetNthIndexValue(i))
      else:
        endIndex = float(self.labelSequenceNode.GetNthIndexValue(i))
        labels = labels.append({self.labelType:startLabel,"Start":startIndex,"End":endIndex},ignore_index=True)
        startIndex = endIndex
        startLabel = currentLabel
        #endIndex = float(self.labelSequenceNode.GetNthIndexValue(i))
    return labels

  def labelExistingEntries(self,imageLabels,autolabels):
    """
    labels a frame that already has a label
    :param imageLabels: the existing image labels in pandas dataframe
    :param autolabels: the autolabels in a pandas dataframe
    :return: the updates image labels
    """
    imageLabels[self.labelType] = ['None' for i in range(len(imageLabels.index))]
    for i in range(0,len(autolabels.index)):
      entriesToLabel = imageLabels.loc[(imageLabels["Time Recorded"] >= autolabels["Start"][i]) & (imageLabels["Time Recorded"] < autolabels["End"][i])]
      for j in entriesToLabel.index:
        imageLabels[self.labelType].iloc._setitem_with_indexer(j,autolabels[self.labelType][i])
    return imageLabels

  def setImageSubtype(self,subtypeName):
    """
    sets the image subtype
    :param subtypeName: name of image subtype i.e RGB, depth, etc.
    :return:
    """
    self.imageSubtype = subtypeName
    self.finishedVideo = False

  def setLabelSequence(self,sequenceNode):
    self.labelSequenceNode = sequenceNode

  def getClassificationLabelFromFile(self):
    """
    retrieves the labels from label file
    :return: label names in dataframe
    """
    seekWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserSeekWidget")
    seekWidget = seekWidget[0]
    timeStamp = seekWidget.findChildren("QLabel")
    timeStamp = float(timeStamp[1].text)
    task = self.autoLabels.loc[(self.autoLabels["Start"]<=timeStamp) & (self.autoLabels["End"]>timeStamp)]
    labelName = task.iloc[0][self.labelType]
    return labelName

  def getSegmentationLabel(self,fileName):
    """
    retrieves the segmentation labels
    :param fileName: the name of the label file that will be outputted
    :return: the filename with _segmentation appended, label map node array
    """
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
    """

    :param segmentationNode:
    :param referenceVolumeNode:
    :return:
    """
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

  def getVtkImageDataAsOpenCVMat(self, volumeNodeName,NodeGiven=False):
    """
    converts vtk image data to an openCV matrix
    :param volumeNodeName: name volume node to be converted
    :param NodeGiven: bool, volumeNodeName defaults to current volume node if set to False
    :return: openCV matrix
    """
    import cv2
    if not NodeGiven:
      cameraVolume = slicer.util.getNode(volumeNodeName)
    else:
      cameraVolume = volumeNodeName
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

  def setDatasetNameAndPath(self,videoPath,datasetName):
    """
    sets name and path of dataset
    :param videoPath: path to the video
    :param datasetName: name of the dataset
    :return: path and dataset name
    """
    if datasetName != self.dataSetName:
      self.finishedVideo = False
    self.videoPath = videoPath
    self.dataSetName = datasetName


  def setRecordingNode(self,recordingNodeName):
    """
    sets the recording node
    :param recordingNodeName: the name of the node that will be set to the recording node
    :return:
    """
    if self.recordingVolumeNode!= None and recordingNodeName != self.recordingVolumeNode.GetName():
      self.finishedVideo = False
    self.recordingVolumeNode = slicer.util.getNode(recordingNodeName)

  def setFileType(self,fileType):
    """
    sets image extension for output files
    :param fileType: jpg, png, etc.
    :return:
    """
    if fileType != self.fileType:
      self.finishedVideo = False
    self.fileType = fileType

  def setCollectingFromSequence(self,collectingFromSequence):
    self.fromSequence = collectingFromSequence

  def setLabellingMethod(self,labellingMethod):
    if labellingMethod != self.labellingMethod:
      self.finishedVideo = False
    self.labellingMethod = labellingMethod

  def setLabelName(self,labelName):
    self.labelName = labelName

  def setLabelType(self,labelType):
    if labelType != self.labelType:
      self.finishedVideo = False
    self.labelType = labelType

  def setAutolabelPath(self,autolabelPath):
    if autolabelPath != self.autoLabelFilePath:
      self.finishedVideo = False
    self.autoLabelFilePath = autolabelPath

  def setVideoIDAndPath(self,videoID,videoPath):
    if videoID != self.videoID:
      self.finishedVideo = False
    self.videoPath = videoPath
    self.videoID = videoID

  def setInputSegmentationNode(self,segmentationNodeName):
    self.segmentationNodeName = segmentationNodeName

  def StartReview(self,labelCSV,labelType,reviewImageNode):
    """
    puts the time and label in the bottom corner of the image node so sequence labels can be reviewed.
    :param labelCSV: the CSV with the labels
    :param labelType: the problem type, i.e segmentation, object detection, etc
    :param reviewImageNode:
    :return:
    """
    self.labelCSV = labelCSV
    self.labelCSV = self.labelCSV.astype({'Time Recorded': 'float64'})
    self.labelType = labelType
    self.reviewImageNode = reviewImageNode
    self.seekWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserSeekWidget")
    self.seekWidget = self.seekWidget[0]
    self.timeLabel = self.seekWidget.findChildren("QLabel")
    self.timeLabel = self.timeLabel[1]
    self.sliceView = self.getSliceView(self.reviewImageNode.GetID())
    self.annotation = self.sliceView.cornerAnnotation()
    self.annotation.GetTextProperty().SetColor(1,1,1)
    self.annotation.GetTextProperty().SetFontSize(100)
    self.lastDisplayedTime = time.time()
    if self.reviewImageNode.GetClassName() == "vtkMRMLStreamingVolumeNode":
      self.reviewImageNodeObserver = self.reviewImageNode.AddObserver(slicer.vtkMRMLStreamingVolumeNode.FrameModifiedEvent, self.onStartReview)
    elif self.reviewImageNode.GetClassName() == "vtkMRMLVectorVolumeNode":
      self.reviewImageNodeObserver = self.reviewImageNode.AddObserver(slicer.vtkMRMLVectorVolumeNode.ImageDataModifiedEvent, self.onStartReview)
    elif self.reviewImageNode.GetClassName() == "vtkMRMLScalarVolumeNode":
      self.reviewImageNodeObserver = self.reviewImageNode.AddObserver(slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent, self.onStartReview)

  def StopReview(self):
    """
    once reviewing is over, remove the observer
    :return:
    """
    self.reviewImageNode.RemoveObserver(self.reviewImageNodeObserver)
    self.reviewImageNodeObserver = None

  def onStartReview(self,caller,eventid):
    """
    begins the review of the labels
    :param caller:
    :param eventid:
    :return:
    """
    recordingTime = float(self.timeLabel.text)
    if time.time() - self.lastDisplayedTime > 0.1:
      labelIndex = self.labelCSV.iloc[(self.labelCSV["Time Recorded"]-recordingTime).abs().argsort()[:1]]
      labelIndex = labelIndex.index[0]
      self.labelString = ''
      for label in self.labelType:
        self.labelString = self.labelString + str(label) + ': '+ str(self.labelCSV[label][labelIndex]) + '\n'
      self.lastDisplayedTime
    self.annotation.SetText(vtk.vtkCornerAnnotation.LowerRight,self.labelString)


  def getSliceView(self,reviewImageNodeID):
    """
    changes layout of the scene to slice view
    :param reviewImageNodeID: the image node that is being reviewed
    :return: slice view
    """
    layoutManager = slicer.app.layoutManager()
    sliceNodes = ["Red","Green","Yellow"]
    activeSliceWidget = None
    for slice in sliceNodes:
      SliceWidget = layoutManager.sliceWidget(slice)
      sliceLogic = SliceWidget.sliceLogic()
      sliceID = sliceLogic.GetSliceCompositeNode().GetBackgroundVolumeID()
      if sliceID == reviewImageNodeID:
        activeSliceWidget = SliceWidget
    if activeSliceWidget != None:
      sliceView = activeSliceWidget.sliceView()
      return(sliceView)
    else:
      return None

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