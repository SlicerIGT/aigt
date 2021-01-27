import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from pathlib import Path
import subprocess
import time

try:
  import pandas
except ModuleNotFoundError:
  slicer.util.pip_install("pandas")
  import pandas


#
# TrainNeuralNet
#

class TrainNeuralNet(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Train Neural Net"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Deep Learn Live"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Rebecca Hisey (Perk Lab)"]  # TODO: replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This module provides utilities for training a neural network for use with Run Neural Net module.
"""  # TODO: update with short description of the module
    self.parent.helpText += self.getDefaultModuleDocumentationLink()  # TODO: verify that the default URL is correct or change it to the actual documentation
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""  # TODO: replace with organization, grant and thanks.

#
# TrainNeuralNetWidget
#

class TrainNeuralNetWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.logic = TrainNeuralNetLogic()
    self.moduleDir = os.path.dirname(slicer.modules.trainneuralnet.path)

    self.selectDataCollapsibleButton = ctk.ctkCollapsibleButton()
    self.selectDataCollapsibleButton.text = "Select Data"
    self.layout.addWidget(self.selectDataCollapsibleButton)
    self.selectDataCollapsibleButton.collapsed = False

    # Layout within the dummy collapsible button
    selectDataFormLayout = qt.QFormLayout(self.selectDataCollapsibleButton)
    self.setupSelectDataLayout(selectDataFormLayout)

    self.trainScriptCollapsibleButton = ctk.ctkCollapsibleButton()
    self.trainScriptCollapsibleButton.text = "Create Training Script"
    self.layout.addWidget(self.trainScriptCollapsibleButton)
    self.trainScriptCollapsibleButton.collapsed=True

    # Layout within the dummy collapsible button
    trainScriptFormLayout = qt.QFormLayout(self.trainScriptCollapsibleButton)
    self.setupTrainScriptLayout(trainScriptFormLayout)

    self.trainNetworkCollapsibleButton = ctk.ctkCollapsibleButton()
    self.trainNetworkCollapsibleButton.text = "Train Network"
    self.layout.addWidget(self.trainNetworkCollapsibleButton)
    self.trainNetworkCollapsibleButton.collapsed = True

    # Layout within the dummy collapsible button
    trainNetworkFormLayout = qt.QFormLayout(self.trainNetworkCollapsibleButton)
    self.setupTrainNetworkLayout(trainNetworkFormLayout)


  def setupSelectDataLayout(self,layout):
    self.datasetDirectorySelector = ctk.ctkDirectoryButton()
    self.datasetDirectoryPath = os.path.join(self.moduleDir,os.pardir,"Datasets")
    self.datasetDirectorySelector.directory = self.datasetDirectoryPath
    layout.addRow(self.datasetDirectorySelector)
    if not os.path.isdir(self.datasetDirectoryPath):
      os.mkdir(self.datasetDirectoryPath)

    self.girderClientLineEdit = qt.QLineEdit()
    self.girderClientLineEdit.setText("https://pocus.cs.queensu.ca/api/v1")
    self.girderClientURL = self.girderClientLineEdit.text
    layout.addRow(self.girderClientLineEdit)
    self.girderClientLineEdit.visible = False

    self.girderUserNameLineEdit = qt.QLineEdit("Username")
    self.girderPasswordLineEdit = qt.QLineEdit("Password")
    self.girderPasswordLineEdit.setEchoMode(2) #hides password as they are typing
    self.signInToGirderButton = qt.QPushButton("Sign in")
    self.girderSignInLayout = qt.QHBoxLayout()
    self.girderSignInLayout.addWidget(self.girderUserNameLineEdit)
    self.girderSignInLayout.addWidget(self.girderPasswordLineEdit)
    self.girderSignInLayout.addWidget(self.signInToGirderButton)
    layout.addRow(self.girderSignInLayout)
    self.girderUserNameLineEdit.visible = False
    self.girderPasswordLineEdit.visible = False
    self.signInToGirderButton.visible = False

    self.girderSignInStatusLabel = qt.QLabel()
    self.girderSignInStatusLabel.visible = False
    layout.addRow(self.girderSignInStatusLabel)

    self.setGirderCollectionComboBox = qt.QComboBox()
    self.setGirderCollectionComboBox.addItem("Select collection")
    layout.addRow(self.setGirderCollectionComboBox)
    self.setGirderCollectionComboBox.visible = False

    self.videoIDSelector = qt.QComboBox()
    self.videoIDSelector.addItems(["Select video IDs","Use all video IDs","Manually select video IDs"])
    layout.addRow(self.videoIDSelector)

    self.labelSelector = qt.QComboBox()
    self.labelSelector.addItems(["Select image labels","Select multiple labels"])
    layout.addRow(self.labelSelector)

    self.label = qt.QLabel("\nFolds")
    layout.addRow(self.label)
    self.numberOfFoldsSelector = qt.QSpinBox()
    self.numberOfFoldsSelector.minimum = 1
    self.numberOfFoldsSelector.maximum = 50
    self.numberOfFoldsSelector.singleStep = 1
    self.numberOfFoldsSelector.value = 1
    self.numberOfFolds = 1

    self.trainValTestSplitComboBox = qt.QComboBox()
    self.trainValTestSplitComboBox.addItems(["Select train-validation-test split","Random percentage","Manual Selection"])
    layout.addRow(self.numberOfFoldsSelector,self.trainValTestSplitComboBox)

    self.trainPercentageSpinBox = qt.QSpinBox()
    self.trainPercentageSpinBox.minimum = 0
    self.trainPercentageSpinBox.maximum = 100
    self.trainPercentageSpinBox.singleStep = 10
    self.trainPercentageSpinBox.value = 70
    self.trainPercentage = 70
    self.trainPercentageSpinBox.visible = False
    self.trainLabel = qt.QLabel("Train percentage: ")
    self.trainLabel.visible = False
    layout.addRow(self.trainLabel,self.trainPercentageSpinBox)

    self.valPercentageSpinBox = qt.QSpinBox()
    self.valPercentageSpinBox.minimum = 0
    self.valPercentageSpinBox.maximum = 100
    self.valPercentageSpinBox.singleStep = 10
    self.valPercentageSpinBox.value = 15
    self.valPercentage = 15
    self.valPercentageSpinBox.visible = False
    self.valLabel = qt.QLabel("Validation percentage: ")
    self.valLabel.visible = False
    layout.addRow(self.valLabel,self.valPercentageSpinBox)

    self.testPercentageSpinBox = qt.QSpinBox()
    self.testPercentageSpinBox.minimum = 0
    self.testPercentageSpinBox.maximum = 100
    self.testPercentageSpinBox.singleStep = 10
    self.testPercentageSpinBox.value = 15
    self.testPercentage = 15
    self.testPercentageSpinBox.visible = False
    self.testLabel = qt.QLabel("Test percentage: ")
    self.testLabel.visible = False
    layout.addRow(self.testLabel,self.testPercentageSpinBox)

    self.fileNameLineEdit = qt.QLineEdit()
    self.fileNameLineEdit.setText("Csv File Name")
    layout.addRow(self.fileNameLineEdit)

    self.createDataCSVButton = qt.QPushButton("Create CSV")
    layout.addRow(self.createDataCSVButton)

    self.blankLabel = qt.QLabel("\n")
    layout.addRow(self.blankLabel)

    self.useDataFromGirderServerCheckBox = qt.QCheckBox("Get data from Girder server")
    self.useDataFromGirderServerCheckBox.checked = False

    layout.addRow(self.useDataFromGirderServerCheckBox)
    self.datasetDirectorySelector.connect('directorySelected(QString)',self.onDatasetSelected)
    self.useDataFromGirderServerCheckBox.connect('stateChanged(int)',self.onUseGirderChecked)
    self.signInToGirderButton.connect('clicked(bool)',self.onSignInClicked)
    self.setGirderCollectionComboBox.connect('currentIndexChanged(int)',self.ongirderCollectionSelected)
    self.videoIDSelector.connect('currentIndexChanged(int)',self.onVideoIDSelected)
    self.labelSelector.connect('currentIndexChanged(int)',self.onImageLabelSelected)
    self.numberOfFoldsSelector.connect('valueChanged(int)',self.onNumberOfFoldsChanged)
    self.trainValTestSplitComboBox.connect('currentIndexChanged(int)',self.onTrainValTestMethodSelected)
    self.trainPercentageSpinBox.connect('valueChanged(int)',self.onTrainPercentageChanged)
    self.valPercentageSpinBox.connect('valueChanged(int)',self.onValPercentageChanged)
    self.testPercentageSpinBox.connect('valueChanged(int)',self.onTestPercentageChanged)
    self.fileNameLineEdit.connect('textChanged(QString)',self.onFileNameChanged)
    self.createDataCSVButton.connect('clicked(bool)',self.onCreateCSVClicked)

  def onFileNameChanged(self):
    self.fileName = self.fileNameLineEdit.text
    if self.useDataFromGirderServerCheckBox.checked:
      self.csvFileSavePath = os.path.join(self.moduleDir,os.pardir,"Datasets",self.setGirderCollectionComboBox.currentText)
    else:
      self.csvFileSavePath = self.datasetDirectorySelector.directory

  def onTrainPercentageChanged(self):
    self.trainPercentage = self.trainPercentageSpinBox.value

  def onValPercentageChanged(self):
    self.valPercentage = self.valPercentageSpinBox.value

  def onTestPercentageChanged(self):
    self.testPercentage = self.testPercentageSpinBox.value

  def onTrainValTestMethodSelected(self):
    if self.trainValTestSplitComboBox.currentText == "Random percentage":
      if self.numberOfFolds == 1:
        self.trainLabel.visible = True
        self.trainPercentageSpinBox.visible = True
        self.valLabel.visible = True
        self.valPercentageSpinBox.visible = True
        self.testLabel.visible = True
        self.testPercentageSpinBox.visible = True
      else:
        self.trainLabel.visible = True
        self.trainPercentageSpinBox.visible = True
        self.valLabel.visible = True
        self.valPercentageSpinBox.visible = True
        self.testLabel.visible = False
        self.testPercentageSpinBox.visible = False
        self.testPercentage = None
    elif self.trainValTestSplitComboBox.currentText == "Manual Selection":
      self.trainLabel.visible = False
      self.trainPercentageSpinBox.visible = False
      self.valLabel.visible = False
      self.valPercentageSpinBox.visible = False
      self.testLabel.visible = False
      self.testPercentageSpinBox.visible = False
      self.openSelectFolds()
      self.selectFoldsWidget.show()
    else:
      self.trainLabel.visible = False
      self.trainPercentageSpinBox.visible = False
      self.valLabel.visible = False
      self.valPercentageSpinBox.visible = False
      self.testLabel.visible = False
      self.testPercentageSpinBox.visible = False

  def openSelectFolds(self):
    self.selectFoldsWidget = qt.QDialog()
    self.selectFoldsWidget.resize(589, 443)
    self.selectFoldsWidget.setWindowTitle("Select Folds")

    self.buttonBox = qt.QDialogButtonBox(self.selectFoldsWidget)
    self.buttonBox.setGeometry(qt.QRect(210, 390, 341, 32))
    self.buttonBox.setOrientation(qt.Qt.Horizontal)
    self.buttonBox.setStandardButtons(qt.QDialogButtonBox.Cancel | qt.QDialogButtonBox.Ok)

    self.scrollArea = qt.QScrollArea(self.selectFoldsWidget)
    self.scrollArea.setGeometry(qt.QRect(10, 10, 561, 381))
    self.scrollArea.setWidgetResizable(False)

    self.scrollAreaWidgetContents = qt.QWidget()
    self.scrollAreaWidgetContents.setGeometry(qt.QRect(0, 0, 10 + (210*self.numberOfFolds), 80*len(self.selectedVideoIDNames)))

    for i in range(0,self.numberOfFolds):

      verticalLayoutWidget = qt.QWidget(self.scrollAreaWidgetContents)
      verticalLayoutWidget.setGeometry(qt.QRect(10 +(i*210), 0, 200, 80*len(self.selectedVideoIDNames)))

      verticalLayout = qt.QVBoxLayout(verticalLayoutWidget)
      verticalLayout.setContentsMargins(0, 0, 0, 0)

      videoIDButtonGroup = qt.QGroupBox(verticalLayoutWidget)
      videoIDButtonGroup.setTitle("Fold "+str(i))

      for j in range(0,len(self.selectedVideoIDNames)):

        videoIDGroupBox = qt.QGroupBox(videoIDButtonGroup)
        videoIDGroupBox.setGeometry(qt.QRect(10, 20 + (j*60), 180, 50))
        videoIDGroupBox.setTitle(self.selectedVideoIDNames[j])

        trainRadioButton = qt.QRadioButton(videoIDGroupBox)
        trainRadioButton.setGeometry(qt.QRect(10, 20, 50, 20))
        trainRadioButton.setText("Train")
        valRadioButton = qt.QRadioButton(videoIDGroupBox)
        valRadioButton.setGeometry(qt.QRect(60, 20, 70, 20))
        valRadioButton.setText("Validation")
        testRadioButton = qt.QRadioButton(videoIDGroupBox)
        testRadioButton.setGeometry(qt.QRect(130, 20, 50, 20))
        testRadioButton.setText("Test")
        verticalLayout.addWidget(videoIDButtonGroup)

    self.scrollArea.setWidget(self.scrollAreaWidgetContents)

    self.buttonBox.accepted.connect(self.foldsSelected)
    self.buttonBox.rejected.connect(self.foldSelectionCancelled)


  def foldsSelected(self):
    allGroupBoxes = self.scrollAreaWidgetContents.findChildren('QGroupBox')
    foldGroupBox = 0
    self.trainList = []
    self.valList = []
    self.testList = []
    if self.useDataFromGirderServerCheckBox.checked:
      self.trainGirderIDs = []
      self.valGirderIDs = []
      self.testGirderIDs = []
    while foldGroupBox < len(allGroupBoxes):
      groupBoxTitle = allGroupBoxes[foldGroupBox].title
      if "Fold" in groupBoxTitle:
        FoldTrainList = []
        FoldValList = []
        FoldTestList = []
        if self.useDataFromGirderServerCheckBox.checked:
          foldTrainGirderIDs = []
          foldValGirderIDs = []
          foldTestGirderIDs = []
        childGroupBoxes = allGroupBoxes[foldGroupBox].findChildren('QGroupBox')
        for i in range(0,len(childGroupBoxes)):
          buttons = childGroupBoxes[i].findChildren("QRadioButton")
          if buttons[2].checked:
            FoldTestList.append(childGroupBoxes[i].title)
            if self.useDataFromGirderServerCheckBox.checked:
              foldTestGirderIDs.append(self.selectedVideoIDgirderIDs[i])
          elif buttons[1].checked:
            FoldValList.append(childGroupBoxes[i].title)
            if self.useDataFromGirderServerCheckBox.checked:
              foldValGirderIDs.append(self.selectedVideoIDgirderIDs[i])
          else:
            FoldTrainList.append(childGroupBoxes[i].title)
            if self.useDataFromGirderServerCheckBox.checked:
              foldTrainGirderIDs.append(self.selectedVideoIDgirderIDs[i])
          foldGroupBox +=1
      foldGroupBox += 1
      self.trainList.append(FoldTrainList)
      self.valList.append(FoldValList)
      self.testList.append(FoldTestList)
      if self.useDataFromGirderServerCheckBox.checked:
        self.trainGirderIDs.append(foldTrainGirderIDs)
        self.valGirderIDs.append(foldValGirderIDs)
        self.testGirderIDs.append(foldTestGirderIDs)
    self.selectFoldsWidget.hide()


  def foldSelectionCancelled(self):
    self.selectFoldsWidget.hide()
    self.trainValTestSplitComboBox.currentIndex = 0


  def onNumberOfFoldsChanged(self):
    self.numberOfFolds = self.numberOfFoldsSelector.value

  def onDatasetSelected(self):
    self.datasetDirectoryPath = self.datasetDirectorySelector.directory
    self.videoIDNames = [x for x in os.listdir(self.datasetDirectorySelector.directory) if not "." in x]
    self.onVideoIDSelected()

  def onImageLabelSelected(self):
    self.imageLabel = self.labelSelector.currentText
    if self.imageLabel == "Select multiple labels":
      self.openSelectLabelsWindow()
      self.selectLabelWidget.show()
    elif self.imageLabel != "Select image labels":
      self.selectedImageLabels = [self.imageLabel]

  def openSelectLabelsWindow(self):
    self.selectLabelWidget = qt.QDialog()
    self.selectLabelWidget.setModal(True)
    self.selectLabelFrame = qt.QFrame(self.selectLabelWidget)
    self.selectLabelFrame.setFrameStyle(0x0006)
    self.selectLabelWidget.setWindowTitle('Select image labels')
    self.selectLabelPopupGeometry = qt.QRect()
    mainWindow = slicer.util.mainWindow()
    if mainWindow:
      mainWindowGeometry = mainWindow.geometry
      self.windowWidth = mainWindow.width * 0.2
      self.windowHeight = mainWindow.height * 0.2
      self.selectLabelPopupGeometry.setWidth(self.windowWidth)
      self.selectLabelPopupGeometry.setHeight(self.windowHeight)
      self.popupPositioned = False
      self.selectLabelWidget.setGeometry(self.selectLabelPopupGeometry)
      self.selectLabelFrame.setGeometry(self.selectLabelPopupGeometry)
      self.selectLabelWidget.move(mainWindow.width / 2.0 - self.windowWidth,
                                     mainWindow.height / 2 - self.windowHeight)
    self.selectLabelsLayout = qt.QVBoxLayout()
    self.selectLabelsLayout.setContentsMargins(12, 4, 4, 4)
    self.selectLabelsLayout.setSpacing(10)
    for i in range(0,len(self.imageLabels)):
      self.selectLabelsLayout.addWidget(qt.QCheckBox(self.imageLabels[i]))
    self.selectImageLabelsPushButton = qt.QPushButton("Done")
    self.selectImageLabelsPushButton.connect('clicked(bool)',self.onMultipleImageLabelsSelected)
    self.selectLabelsLayout.addWidget(self.selectImageLabelsPushButton)
    self.selectLabelFrame.setLayout(self.selectLabelsLayout)

  def onMultipleImageLabelsSelected(self):
    self.selectLabelWidget.hide()
    allLabelCheckBoxes = self.selectLabelFrame.findChildren('QCheckBox')
    self.selectedImageLabels = []
    for checkBox in allLabelCheckBoxes:
      if checkBox.checked:
        self.selectedImageLabels.append(checkBox.text)

  def onVideoIDSelected(self):
    if self.videoIDSelector.currentText == "Use all video IDs":
      if self.useDataFromGirderServerCheckBox.checked:
        self.selectedVideoIDNames = []
        self.selectedVideoIDgirderIDs = []
        self.girderCollectionFolders = self.girderClient.listFolder(parentId=self.girderCollectionID,parentFolderType="collection")
        for folder in self.girderCollectionFolders:
          self.videoIDNames.append(folder["name"])
          self.videoIDgirderIDs.append(folder["_id"])
          self.selectedVideoIDNames.append(folder["name"])
          self.selectedVideoIDgirderIDs.append(folder["_id"])
      else:
        self.videoIDNames = [x for x in os.listdir(self.datasetDirectorySelector.directory) if not '.' in x]
        self.selectedVideoIDNames = self.videoIDNames
      self.addImageLabelsToComboBox()
    elif self.videoIDSelector.currentText == "Manually select video IDs":
      self.openSelectVideosWindow()
      self.selectVideoIDsWidget.show()

  def openSelectVideosWindow(self):
    self.selectVideoIDsWidget = qt.QDialog()
    self.selectVideoIDsWidget.setModal(True)
    self.selectVideoIDsFrame = qt.QFrame()
    self.selectVideoIDsFrame.setFrameStyle(0x0006)
    self.selectVideoIDsWidget.setWindowTitle('Select video IDs')
    self.selectVideoIDsScrollArea = qt.QScrollArea(self.selectVideoIDsWidget)
    self.selectVideoIDsScrollArea.setWidget(self.selectVideoIDsFrame)
    self.selectVideoIDsScrollArea.setWidgetResizable(True)
    self.selectVideoIDsPopupGeometry = qt.QRect()
    mainWindow = slicer.util.mainWindow()
    if mainWindow:
      mainWindowGeometry = mainWindow.geometry
      self.windowWidth = mainWindow.width * 0.35
      self.windowHeight = mainWindow.height * 0.35
      self.selectVideoIDsPopupGeometry.setWidth(self.windowWidth)
      self.selectVideoIDsPopupGeometry.setHeight(self.windowHeight)
      self.popupPositioned = False
      self.selectVideoIDsWidget.setGeometry(self.selectVideoIDsPopupGeometry)
      self.selectVideoIDsFrame.setGeometry(self.selectVideoIDsPopupGeometry)
      self.selectVideoIDsScrollArea.setGeometry(self.selectVideoIDsPopupGeometry)
      self.selectVideoIDsWidget.move(mainWindow.width / 2.0 - self.windowWidth,mainWindow.height / 2 - self.windowHeight)
    self.selectVideoIDsLayout = qt.QVBoxLayout()
    self.selectVideoIDsLayout.setContentsMargins(12, 4, 4, 4)
    self.selectVideoIDsLayout.setSpacing(10)
    self.selectVideoIDsLayout.addStrut(self.windowWidth)
    #self.selectVideoIDsLayout.setGeometry(self.selectVideoIDsPopupGeometry)
    self.selectableGirderIDs = []
    for video in range(0,len(self.videoIDNames)):
      if self.useDataFromGirderServerCheckBox.checked:
        videoID = self.girderClient.listFolder(self.videoIDgirderIDs[video])
        numImageSubtypes = sum(1 for _ in videoID)
      else:
        videoID = self.videoIDNames[video]
        imageSubtypes = os.listdir(os.path.join(self.datasetDirectoryPath,videoID))
        numImageSubtypes = len([x for x in imageSubtypes if not '.' in x])
      if numImageSubtypes == 0:
        videoCheckBox = qt.QCheckBox(self.videoIDNames[video])
        self.selectVideoIDsLayout.addWidget(videoCheckBox)
        if self.useDataFromGirderServerCheckBox.checked:
          self.selectableGirderIDs.append(self.videoIDgirderIDs[video])
      else:
        if self.useDataFromGirderServerCheckBox.checked:
          imageSubtypeGenerator = self.girderClient.listFolder(self.videoIDgirderIDs[video])
          self.selectableGirderIDs.append(self.videoIDgirderIDs[video])
        else:
          imageSubtypeGenerator = os.listdir(os.path.join(self.datasetDirectoryPath,videoID))
        videoCollapsibleButton = ctk.ctkCollapsibleButton()
        videoCollapsibleButton.text = self.videoIDNames[video]
        self.videoButtonLayout = qt.QFormLayout(videoCollapsibleButton)

        for imageSubtype in imageSubtypeGenerator:
          if self.useDataFromGirderServerCheckBox.checked:
            imageSubtypeButton = qt.QCheckBox(imageSubtype["name"])
            self.selectableGirderIDs.append(imageSubtype["_id"])
          else:
            imageSubtypeButton = qt.QCheckBox(imageSubtype)
          self.videoButtonLayout.addWidget(imageSubtypeButton)
        self.selectVideoIDsLayout.addWidget(videoCollapsibleButton)

    self.completedSelectionButton = qt.QPushButton("Done")
    self.completedSelectionButton.connect('clicked(bool)',self.onVideoIDSelectionComplete)
    self.selectVideoIDsLayout.addWidget(self.completedSelectionButton)

    self.selectVideoIDsFrame.setLayout(self.selectVideoIDsLayout)

  def onVideoIDSelectionComplete(self):
    self.selectVideoIDsWidget.hide()
    videoIDCheckBoxes = self.selectVideoIDsFrame.findChildren('QAbstractButton')
    self.selectedVideoIDNames = []
    self.selectedVideoIDgirderIDs = []
    i = 0
    while i < len(videoIDCheckBoxes):
      childWidgets = videoIDCheckBoxes[i].findChildren('QCheckBox')
      if len(childWidgets)==0:
        if videoIDCheckBoxes[i].checked:
            childName = videoIDCheckBoxes[i].text
            self.selectedVideoIDNames.append(childName)
            if self.useDataFromGirderServerCheckBox.checked:
              self.selectedVideoIDgirderIDs.append(self.selectableGirderIDs[i])
        i+=1
      else:
        parentName = videoIDCheckBoxes[i].text
        for subtype in range(0,len(childWidgets)):
          if childWidgets[subtype].checked:
            subtypeName = childWidgets[subtype].text
            childName = parentName + '/' + subtypeName
            self.selectedVideoIDNames.append(childName)
            if self.useDataFromGirderServerCheckBox.checked:
              self.selectedVideoIDgirderIDs.append(self.selectableGirderIDs[i+subtype + 1])
        i+=(1+ len(childWidgets))
    self.addImageLabelsToComboBox()

  def addImageLabelsToComboBox(self):
    labelFileName = self.selectedVideoIDNames[0].replace("/","_") + "_Labels.csv"
    tempFilePath = os.path.join(self.moduleDir, "temp")
    for i in range(self.labelSelector.count,1,-1):
      self.labelSelector.removeItem(i)
    try:
      os.mkdir(tempFilePath)
    except FileExistsError:
      localcsvFilePath = os.path.join(self.datasetDirectoryPath,self.selectedVideoIDNames[0],labelFileName)
    if self.useDataFromGirderServerCheckBox.checked:
      csvFile = self.girderClient.listItem(self.selectedVideoIDgirderIDs[0],name=labelFileName)
      for file in csvFile:
        csvFileID = file["_id"]
      self.girderClient.downloadItem(csvFileID,tempFilePath)
      localcsvFilePath = os.path.join(tempFilePath,labelFileName)
    else:
      localcsvFilePath = os.path.join(self.datasetDirectorySelector.directory,self.selectedVideoIDNames[0],labelFileName)
    labelFile = pandas.read_csv(localcsvFilePath)
    headings = labelFile.columns
    self.imageLabels = []
    for i in range(2,len(headings)):
      if headings[i] != "Time Recorded":
        self.labelSelector.addItem(headings[i])
        self.imageLabels.append(headings[i])

  def ongirderCollectionSelected(self):
    self.girderCollectionIndex = self.setGirderCollectionComboBox.currentIndex
    self.girderCollectionID = self.girderCollectionIDs[self.girderCollectionIndex-1]
    self.videoIDNames = []
    self.videoIDgirderIDs = []
    self.girderCollectionFolders = self.girderClient.listFolder(parentId=self.girderCollectionID,parentFolderType="collection")
    for folder in self.girderCollectionFolders:
      self.videoIDNames.append(folder["name"])
      self.videoIDgirderIDs.append(folder["_id"])

  def onSignInClicked(self):
    self.girderClient = girder_client.GirderClient(apiUrl=self.girderClientLineEdit.text)
    for i in range(self.setGirderCollectionComboBox.count,0,-1):
      self.setGirderCollectionComboBox.removeItem(i)
    try:
      self.girderClient.authenticate(username=self.girderUserNameLineEdit.text,password=self.girderPasswordLineEdit.text)
      self.girderSignInStatusLabel.setText("Successfully authenticated girder client")
      self.girderSignInStatusLabel.visible = True
      self.girderCollectionNames = []
      self.girderCollectionIDs = []
      self.girderCollections = self.girderClient.listCollection()
      for col in self.girderCollections:
        self.girderCollectionNames.append(col["name"])
        self.girderCollectionIDs.append(col["_id"])
      self.setGirderCollectionComboBox.addItems(self.girderCollectionNames)
    except girder_client.AuthenticationError:
      self.girderSignInStatusLabel.setText("Failed to authenticate: Incorrect username or password")
      self.girderSignInStatusLabel.visible = True

  def onUseGirderChecked(self):
    if self.useDataFromGirderServerCheckBox.checked:
      self.datasetDirectorySelector.visible = False
      self.girderClientLineEdit.visible = True
      self.girderUserNameLineEdit.visible = True
      self.girderPasswordLineEdit.visible = True
      self.signInToGirderButton.visible = True
      self.setGirderCollectionComboBox.visible = True
    else:
      self.datasetDirectorySelector.visible = True
      self.girderClientLineEdit.visible = False
      self.girderUserNameLineEdit.visible = False
      self.girderPasswordLineEdit.visible = False
      self.signInToGirderButton.visible = False
      self.setGirderCollectionComboBox.visible = False

  def onCreateCSVClicked(self):
    self.logic.setCSVSavePath(self.csvFileSavePath,self.fileName)
    self.logic.setLabels(self.selectedImageLabels)
    if self.trainValTestSplitComboBox.currentText == "Random percentage":
      if self.useDataFromGirderServerCheckBox.checked:
        self.logic.setSelectedVideoIDs(self.selectedVideoIDNames,self.selectedVideoIDgirderIDs,numberOfFolds = self.numberOfFolds,train=self.trainPercentage,val=self.valPercentage,test=self.testPercentage)
      else:
        self.logic.setSelectedVideoIDs(self.selectedVideoIDNames, numberOfFolds=self.numberOfFolds,train=self.trainPercentage, val=self.valPercentage, test=self.testPercentage)
    elif self.trainValTestSplitComboBox.currentText == "Manual Selection":
      self.logic.setTrainValTestSet(self.trainList,self.valList,self.testList)
      if self.useDataFromGirderServerCheckBox.checked:
        self.logic.setTrainValTestGirderIDs(self.trainGirderIDs,self.valGirderIDs,self.testGirderIDs)
    if self.useDataFromGirderServerCheckBox.checked:
      self.logic.setGirderClient(self.girderClient,os.path.join(self.moduleDir,"temp"))
      self.logic.setGirderAPIURL(self.girderClientURL)
      status = self.logic.createCSVFromGirder()
    else:
      status = self.logic.createCSV()
    if status ==0 :
      self.resetSelectDataWidget()

  def resetSelectDataWidget(self):
    self.videoIDSelector.currentIndex = 0
    self.datasetDirectorySelector.directory = os.path.join(self.moduleDir,os.pardir,"Datasets")
    self.setGirderCollectionComboBox.currentIndex = 0
    self.labelSelector.currentIndex = 0
    self.numberOfFoldsSelector.value = 1
    self.trainValTestSplitComboBox.currentIndex=0
    self.fileNameLineEdit.setText("CSV File Name")

  def setupTrainScriptLayout(self,layout):
    self.modelDirectoryFilePathSelector = ctk.ctkDirectoryButton()
    self.modelDirectoryFilePath = os.path.join(self.moduleDir, os.pardir, "Networks")
    self.modelDirectoryFilePathSelector.directory = self.modelDirectoryFilePath
    layout.addRow(self.modelDirectoryFilePathSelector)

    self.networkTypeSelector = qt.QComboBox()
    self.networkTypeSelector.addItems(["Select network type","Create new network"])
    networks = os.listdir(os.path.join(self.moduleDir, os.pardir, 'Networks'))
    networks = [x for x in networks if not '.' in x]
    self.networkTypeSelector.addItems(networks)
    self.networkType = "Select network type"
    layout.addRow(self.networkTypeSelector)

    self.scriptTypeSelector = qt.QComboBox()
    self.scriptTypeSelector.addItems(["Select training script type","Jupyter notebook","Python script"])
    self.scriptType = "Select training script type"
    layout.addRow(self.scriptTypeSelector)

    self.createTrainScriptButton = qt.QPushButton("Create training script")
    layout.addRow(self.createTrainScriptButton)

    self.modelDirectoryFilePathSelector.connect('directorySelected(QString)',self.onModelDirectorySelected)
    self.networkTypeSelector.connect('currentIndexChanged(int)',self.onNetworkTypeSelected)
    self.scriptTypeSelector.connect('currentIndexChanged(int)',self.onScriptTypeSelected)
    self.createTrainScriptButton.connect('clicked(bool)',self.onCreateTrainingScriptClicked)

  def onModelDirectorySelected(self):
    self.modelDirectoryFilePath = self.modelDirectoryFilePathSelector.directory
    for i in range(self.networkTypeSelector.count,1,-1):
      self.networkTypeSelector.removeItem(i)
    networkTypes = [x for x in os.listdir(self.modelDirectoryFilePath) if not '.' in x]
    self.networkTypeSelector.addItems(networkTypes)
    self.logic.setModelDirectoryPath(self.modelDirectoryFilePath)

  def onNetworkTypeSelected(self):
    self.networkType = self.networkTypeSelector.currentText
    if self.networkType == "Create new network":
      self.openCreateNewNetworkDialog()
      self.createNewNetworkDialog.show()
    elif self.networkType != "Select networkType":
      self.logic.setNetworkType(self.networkType)
      self.logic.setModelDirectoryPath(self.modelDirectoryFilePath)

  def openCreateNewNetworkDialog(self):
    self.createNewNetworkDialog = qt.QDialog()
    self.createNewNetworkDialog.setWindowTitle("Create New Network Type")
    self.createNewNetworkDialog.resize(224, 176)
    self.buttonBox = qt.QDialogButtonBox(self.createNewNetworkDialog)
    self.buttonBox.setGeometry(qt.QRect(10, 120, 211, 32))
    self.buttonBox.setOrientation(qt.Qt.Horizontal)
    self.buttonBox.setStandardButtons(qt.QDialogButtonBox.Cancel | qt.QDialogButtonBox.Ok)
    self.buttonBox.setCenterButtons(True)
    self.networkNamelineEdit = qt.QLineEdit(self.createNewNetworkDialog)
    self.networkNamelineEdit.setGeometry(qt.QRect(20, 50, 181, 20))
    self.networkNamelineEdit.setText("Model name")
    self.newNetworkErrorLabel = qt.QLabel(self.createNewNetworkDialog)
    self.newNetworkErrorLabel.setGeometry(qt.QRect(20, 80, 181, 13))
    self.newNetworkErrorLabel.setText("")

    self.buttonBox.accepted.connect(self.newNetworkNameSelected)
    self.buttonBox.rejected.connect(self.cancelCreateNewNetwork)

  def newNetworkNameSelected(self):
    newNetworkName = self.networkNamelineEdit.text
    if newNetworkName == "Model name":
      self.newNetworkErrorLabel.setText("Invalid name: Model name")
    elif os.path.isdir(os.path.join(self.modelDirectoryFilePath,newNetworkName)):
      self.newNetworkErrorLabel.setText("Model "+newNetworkName+" already exists")
    else:
      self.newNetworkErrorLabel.setText("")
      self.createNewNetworkDialog.hide()
      self.networkType = newNetworkName
      self.networkTypeSelector.addItem(newNetworkName)
      self.networkTypeSelector.currentText = newNetworkName
      self.logic.setModelDirectoryPath(self.modelDirectoryFilePath)
      self.logic.setNetworkType(self.networkType)
      import RunNeuralNet
      runNeuralNetLogic = RunNeuralNet.RunNeuralNetLogic()
      runNeuralNetLogic.createNewModel(self.networkType,self.modelDirectoryFilePath)

  def cancelCreateNewNetwork(self):
    self.networkTypeSelector.currentText = "Select network type"
    self.createNewNetworkDialog.hide()

  def onScriptTypeSelected(self):
    self.scriptType = self.scriptTypeSelector.currentText
    if self.scriptType != "Select training script type":
      self.logic.setTrainingScriptType(self.scriptType)

  def onCreateTrainingScriptClicked(self):
    if self.networkType == "Select network type":
      logging.info("No network type selected, could not create training script")
    elif self.scriptType == "Select training script type":
      logging.info("No script type selected, could not create training script")
    else:
      self.logic.createTrainingScript(self.moduleDir)

  def setupTrainNetworkLayout(self,layout):
    self.condaDirectoryPathSelector = ctk.ctkDirectoryButton()
    self.condaDirectoryPath = self.getCondaPath()
    self.condaDirectoryPathSelector.directory = self.condaDirectoryPath
    layout.addRow("Conda executable location: ", self.condaDirectoryPathSelector)
    self.logic.setCondaDirectory(self.condaDirectoryPath)

    self.environmentNameLineEdit = qt.QLineEdit("EnvironmentName")
    self.environmentName = "kerasGPUEnv"
    layout.addRow("Conda enviroment name: ", self.environmentNameLineEdit)
    self.logic.setCondaEnvironmentName(self.environmentName)

    self.dataCSVSelector = ctk.ctkPathLineEdit()
    self.dataCSVSelector.showBrowseButton = True
    self.dataCSV = os.path.join(self.moduleDir, os.pardir, "Datasets")
    self.dataCSVSelector.setCurrentPath(self.dataCSV)
    layout.addRow("Data CSV: ", self.dataCSVSelector)

    self.trainingScriptSelector = ctk.ctkPathLineEdit()
    self.trainingScriptSelector.showBrowseButton = True
    self.trainingScript = os.path.join(self.moduleDir, os.pardir, "Networks")
    self.trainingScriptSelector.setCurrentPath(self.trainingScript)
    layout.addRow("Training Script: ",self.trainingScriptSelector)

    self.trainingRunNameLineEdit = qt.QLineEdit()
    self.trainingRunNameLineEdit.setText("Name of training run")
    layout.addRow(self.trainingRunNameLineEdit)

    self.runTrainingButton = qt.QPushButton("Train")
    self.runTrainingButton.enabled = False
    layout.addRow(self.runTrainingButton)

    self.trainScriptSelected = False
    self.trainNameSelected = False
    self.dataCSVSelected  = False

    self.condaDirectoryPathSelector.connect('directorySelected(QString)', self.condaPathChanged)
    self.environmentNameLineEdit.connect('textChanged(QString)', self.onEnvironmentNameChanged)
    self.dataCSVSelector.connect('currentPathChanged(QString)',self.ondataCSVSelected)
    self.trainingScriptSelector.connect('currentPathChanged(QString)',self.onTrainingScriptSelected)
    self.trainingRunNameLineEdit.connect('textChanged(QString)',self.onTrainingRunNameChanged)
    self.runTrainingButton.connect('clicked(bool)',self.onTrainClicked)

  def ondataCSVSelected(self):
    print("Got here")
    if ".csv" in os.path.basename(self.dataCSVSelector.currentPath):
      self.logic.setDataCSV(self.dataCSVSelector.currentPath)
      self.dataCSVSelected = True
      print("Data csv selected")
      if self.trainScriptSelected and self.trainNameSelected:
        self.runTrainingButton.enabled = True

  def getCondaPath(self):
    condaPath = str(Path.home())
    homePath = str(Path.home())
    if "Anaconda3" in os.listdir(homePath):
        condaPath = os.path.join(homePath,"Anaconda3")
    return condaPath

  def condaPathChanged(self):
    self.condaDirectoryPath = self.condaDirectoryPathSelector.directory
    self.logic.setCondaDirectory(self.condaDirectoryPath)

  def onEnvironmentNameChanged(self):
    self.environmentName = self.environmentNameLineEdit.text
    self.logic.setCondaEnvironmentName(self.environmentName)

  def onTrainingScriptSelected(self):
    self.trainingScript = self.trainingScriptSelector.currentPath
    if not ".py" in os.path.basename(self.trainingScript) and not ".ipynb" in os.path.basename(self.trainingScript):
      logging.info("Training script needs to be a .py or .ipynb file")
    else:
      self.logic.setTrainingScriptLocation(self.trainingScript)
      self.trainScriptSelected = True

      if self.trainNameSelected and self.dataCSVSelected:
        self.runTrainingButton.enabled = True

  def onTrainingRunNameChanged(self):
    self.trainingRunName = self.trainingRunNameLineEdit.text
    if self.trainingRunName != "Name of training run":
      self.logic.setTrainingRunName(self.trainingRunName)
      self.trainNameSelected = True
      if self.trainScriptSelected and self.dataCSVSelected:
        self.runTrainingButton.enabled = True

  def onTrainClicked(self):
    self.logic.runTraining(self.moduleDir)

  def cleanup(self):
    pass


#
# TrainNeuralNetLogic
#

class TrainNeuralNetLogic(ScriptedLoadableModuleLogic):

  def runTraining(self,moduleDir):
    self.moduleDir = moduleDir
    saveLocation = os.path.join(os.path.dirname(self.trainingScriptPath),self.trainingRunName+'_Fold_0')
    if os.path.isdir(saveLocation):
      self.openWarningWidget(self.trainingRunName)
      self.warningWidget.show()
    else:
      cmd = [str(self.moduleDir + "\Scripts\openTrainCMDPrompt.bat"),
             str(self.moduleDir),
             str(self.condaPath),
             str(self.condaEnvName),
             str(self.dataCSV),
             str(os.path.join(os.path.dirname(self.trainingScriptPath), self.trainingRunName)),
             str(self.trainingScriptPath)]
      strCMD = cmd[0]
      for i in range(1,len(cmd)):
        strCMD = strCMD + ' ' + cmd[i]
      p = slicer.util.launchConsoleProcess(strCMD, useStartupEnvironment=True)
      slicer.util.logProcessOutput(p)
      logging.info("Saving training run to: " + str(os.path.join(os.path.dirname(self.trainingScriptPath), self.trainingRunName)))

  def openWarningWidget(self,trainingRunName):
    self.warningWidget = qt.QDialog()
    self.warningWidget.setWindowTitle("Permission")
    self.warningWidget.resize(211, 118)
    self.buttonBox = qt.QDialogButtonBox(self.warningWidget)
    self.buttonBox.setGeometry(qt.QRect(10, 70, 191, 32))
    self.buttonBox.setOrientation(qt.Qt.Horizontal)
    self.buttonBox.setStandardButtons(qt.QDialogButtonBox.Cancel | qt.QDialogButtonBox.Ok)
    self.buttonBox.setCenterButtons(True)
    self.warninglabel = qt.QLabel(self.warningWidget)
    self.warninglabel.setGeometry(qt.QRect(10, 10, 191, 51))
    self.warninglabel.setText("A training run named:\n\t" + self.trainingRunName + "\nalready exists.\nDo you want to overwrite?")

    self.buttonBox.accepted.connect(self.overwriteSelected)
    self.buttonBox.rejected.connect(self.cancelOverwrite)

  def overwriteSelected(self):
    self.warningWidget.hide()
    DirsToRemove = [dir for dir in os.listdir(os.path.dirname(self.trainingScriptPath)) if self.trainingRunName+'_Fold' in dir]
    baseDir = os.path.dirname(self.trainingScriptPath)
    for dir in DirsToRemove:
      for file in os.listdir(os.path.join(baseDir,dir)):
        os.remove(os.path.join(baseDir,dir,file))
      os.removedirs(os.path.join(baseDir,dir))
    cmd = [str(self.moduleDir + "\Scripts\openTrainCMDPrompt.bat"),
           str(self.moduleDir),
           str(self.condaPath),
           str(self.condaEnvName),
           str(self.dataCSV),
           str(os.path.join(os.path.dirname(self.trainingScriptPath), self.trainingRunName)),
           str(self.trainingScriptPath)]
    strCMD = cmd[0]
    for i in range(1, len(cmd)):
      strCMD = strCMD + ' ' + cmd[i]
    p = slicer.util.launchConsoleProcess(strCMD,useStartupEnvironment=True)
    slicer.util.logProcessOutput(p)
    logging.info("Saving training run to: " + str(os.path.join(os.path.dirname(self.trainingScriptPath),self.trainingRunName)))

  def cancelOverwrite(self):
    self.warningWidget.hide()

  def setDataCSV(self,dataCSV):
    self.dataCSV = dataCSV

  def setTrainingRunName(self,trainRunName):
    self.trainingRunName = trainRunName

  def setTrainingScriptLocation(self,trainingScript):
    self.trainingScriptPath = trainingScript
    if ".py" in os.path.basename(self.trainingScriptPath):
      self.setTrainingScriptType("Python script")
    else:
      self.setTrainingScriptType("Jupyter notebook")


  def setCondaDirectory(self,condaPath):
    self.condaPath = condaPath

  def setCondaEnvironmentName(self,envName):
    self.condaEnvName = envName

  def createTrainingScript(self,moduleDir):
    if self.scriptType == "Python":
      templateTrainScriptFilePath = os.path.join(moduleDir, "Scripts", "TemplatePythonTrainFile.txt")
    else:
      templateTrainScriptFilePath = os.path.join(moduleDir, "Scripts", "TemplateJupyterTrainFile.txt")
    newTrainScriptPath = os.path.join(self.modelDirectoryPath, self.networkType)
    templateFile = open(templateTrainScriptFilePath, 'r')
    templateFileText = templateFile.read()
    templateFile.close()
    newFileText = templateFileText.replace('MODELNAME', self.networkType)
    if self.scriptType == "Python":
      newModelFile = open(os.path.join(newTrainScriptPath, "Train_" +self.networkType + '.py'), 'w')
      newModelFileName = os.path.join(newTrainScriptPath, "Train_" +self.networkType + '.py')
    else:
      newModelFile = open(os.path.join(newTrainScriptPath, "Train_" +self.networkType + '.ipynb'), 'w')
      newModelFileName = os.path.join(newTrainScriptPath, "Train_" + self.networkType + '.ipynb')
    newModelFile.write(newFileText)
    newModelFile.close()
    logging.info("Successfully created training script: " + newModelFileName)



  def setTrainingScriptType(self,trainingScriptType):
    if trainingScriptType == "Jupyter notebook":
      self.scriptType = "Jupyter"
    else:
      self.scriptType = "Python"

  def setModelDirectoryPath(self,modelDirectoryPath):
    self.modelDirectoryPath = modelDirectoryPath

  def setNetworkType(self,networkType):
    self.networkType = networkType

  def createCSV(self):
    self.datasetPath = os.path.dirname(self.csvSavePath)
    Columns = ["Fold", "Set", "Folder", "FileName"]
    for i in range(0, len(self.Labels)):
      Columns.append(self.Labels[i])
    csvDataFrame = pandas.DataFrame(columns=Columns)
    totalRowCount = 0
    for fold in range(0, len(self.trainList)):
      setType = "Train"
      for j in range(len(self.trainList[fold])):
        folderName = os.path.join(self.datasetPath,self.trainList[fold][j])
        labelFileName = self.trainList[fold][j].replace("/","_") + "_Labels.csv"
        labelCSV = pandas.read_csv(os.path.join(folderName,labelFileName))
        for row in range(0,len(labelCSV.index)):
          csvDataFrame = csvDataFrame.append({"Fold": fold,
                                                "Set": setType,
                                                "Folder": folderName,
                                                "FileName": labelCSV["FileName"][row]}, ignore_index=True)
          for i in range(0,len(self.Labels)):
            csvDataFrame[self.Labels[i]][totalRowCount] = labelCSV[self.Labels[i]][row]
          totalRowCount += 1
      setType = "Validation"
      for j in range(len(self.valList[fold])):
        folderName = os.path.join(self.datasetPath, self.valList[fold][j])
        labelFileName = self.valList[fold][j].replace("/", "_") + "_Labels.csv"
        labelCSV = pandas.read_csv(os.path.join(folderName, labelFileName))
        for row in range(0, len(labelCSV.index)):
          csvDataFrame = csvDataFrame.append({"Fold": fold,
                                              "Set": setType,
                                              "Folder": folderName,
                                              "FileName": labelCSV["FileName"][row]}, ignore_index=True)
          for i in range(0, len(self.Labels)):
            csvDataFrame[self.Labels[i]][totalRowCount] = labelCSV[self.Labels[i]][row]
          totalRowCount += 1
      setType = "Test"
      for j in range(len(self.testList[fold])):
        folderName = os.path.join(self.datasetPath, self.testList[fold][j])
        labelFileName = self.testList[fold][j].replace("/", "_") + "_Labels.csv"
        labelCSV = pandas.read_csv(os.path.join(folderName, labelFileName))
        for row in range(0, len(labelCSV.index)):
          csvDataFrame = csvDataFrame.append({"Fold": fold,
                                              "Set": setType,
                                              "Folder": folderName,
                                              "FileName": labelCSV["FileName"][row]}, ignore_index=True)
          for i in range(0, len(self.Labels)):
            csvDataFrame[self.Labels[i]][totalRowCount] = labelCSV[self.Labels[i]][row]
          totalRowCount += 1
    csvDataFrame.to_csv(self.csvSavePath)
    logging.info("Successfully saved csv to: " + str(self.csvSavePath))
    return 0

  def createCSVFromGirder(self):
    self.collectionName = os.path.basename(os.path.dirname(self.csvSavePath))
    Columns = ["Fold","Set","Girder_URL","Collection_Name","Folder","FileName","GirderID"]
    for i in range(0,len(self.Labels)):
      Columns.append(self.Labels[i])
    csvDataFrame = pandas.DataFrame(columns = Columns)
    totalRowCount = 0
    for fold in range(0,len(self.trainList)):
      setType = "Train"
      for j in range(len(self.trainList[fold])):
        labelFileName = self.trainList[fold][j].replace("/", "_") + "_Labels.csv"
        try:
          labelCSV = pandas.read_csv(os.path.join(self.tempFileDir, labelFileName))
        except FileNotFoundError:
          csvFile = self.girderClient.listItem(self.trainGirderIDs[fold][j], name=labelFileName)
          for file in csvFile:
            csvFileID = file["_id"]
          self.girderClient.downloadItem(csvFileID, self.tempFileDir)
          localcsvFilePath = os.path.join(self.tempFileDir, labelFileName)
          labelCSV = pandas.read_csv(localcsvFilePath)
        folderName = self.trainList[fold][j]
        girderFile = self.girderClient.listItem(self.trainGirderIDs[fold][j])
        row = 0
        for file in girderFile:
          fileID = file["_id"]
          fileName = file["name"]
          if not ".csv" in fileName:
            csvDataFrame = csvDataFrame.append({"Fold": fold,
                                                "Set": setType,
                                                "Girder_URL": self.girderURL,
                                                "Collection_Name": self.collectionName,
                                                "Folder": folderName,
                                                "FileName": fileName,
                                                "GirderID": fileID}, ignore_index=True)

            for i in range(len(self.Labels)):
              csvDataFrame[self.Labels[i]][totalRowCount] = labelCSV[self.Labels[i]][row]
            row+=1
            totalRowCount += 1
      setType = "Validation"
      for j in range(len(self.valList[fold])):
        labelFileName = self.valList[fold][j].replace("/", "_") + "_Labels.csv"
        try:
          labelCSV = pandas.read_csv(os.path.join(self.tempFileDir, labelFileName))
        except FileNotFoundError:
          csvFile = self.girderClient.listItem(self.valGirderIDs[fold][j], name=labelFileName)
          for file in csvFile:
            csvFileID = file["_id"]
          self.girderClient.downloadItem(csvFileID, self.tempFileDir)
          localcsvFilePath = os.path.join(self.tempFileDir, labelFileName)
          labelCSV = pandas.read_csv(localcsvFilePath)
        folderName = self.valList[fold][j]
        row = 0
        girderFile = self.girderClient.listItem(self.valGirderIDs[fold][j],labelCSV["FileName"][row])
        for file in girderFile:
          fileID = file["_id"]
          fileName = file["name"]
          if not ".csv" in fileName:
            csvDataFrame = csvDataFrame.append({"Fold": fold,
                                                "Set": setType,
                                                "Girder_URL": self.girderURL,
                                                "Collection_Name": self.collectionName,
                                                "Folder": folderName,
                                                "FileName": fileName,
                                                "GirderID": fileID},ignore_index=True)
            for i in range(len(self.Labels)):
              csvDataFrame[self.Labels[i]][totalRowCount] = labelCSV[self.Labels[i]][row]
            totalRowCount +=1
            row += 1
      setType = "Test"
      for j in range(len(self.testList[fold])):
        labelFileName = self.testList[fold][j].replace("/", "_") + "_Labels.csv"
        try:
          labelCSV = pandas.read_csv(os.path.join(self.tempFileDir, labelFileName))
        except FileNotFoundError:
          csvFile = self.girderClient.listItem(self.testGirderIDs[fold][j], name=labelFileName)
          for file in csvFile:
            csvFileID = file["_id"]
          self.girderClient.downloadItem(csvFileID, self.tempFileDir)
          localcsvFilePath = os.path.join(self.tempFileDir, labelFileName)
          labelCSV = pandas.read_csv(localcsvFilePath)
        folderName = self.testList[fold][j]
        row=0
        girderFile = self.girderClient.listItem(self.testGirderIDs[fold][j], labelCSV["FileName"][row])
        for file in girderFile:
          fileID = file["_id"]
          fileName = file["name"]
          if not ".csv" in fileName:
            csvDataFrame = csvDataFrame.append({"Fold": fold,
                                                "Set": setType,
                                                "Girder_URL": self.girderURL,
                                                "Collection_Name": self.collectionName,
                                                "Folder": folderName,
                                                "FileName": fileName,
                                                "GirderID": fileID}, ignore_index=True)
            for i in range(len(self.Labels)):
              csvDataFrame[self.Labels[i]][totalRowCount] = labelCSV[self.Labels[i]][row]
            totalRowCount += 1
            row += 1
    try:
      csvDataFrame.to_csv(self.csvSavePath)
    except FileNotFoundError:
      os.mkdir(os.path.dirname(self.csvSavePath))
      csvDataFrame.to_csv(self.csvSavePath)
    logging.info("Successfully saved csv to: " + str(self.csvSavePath))

    for file in os.listdir(self.tempFileDir):
      os.remove(os.path.join(self.tempFileDir,file))
    os.removedirs(self.tempFileDir)
    logging.info("Successfully removed temporary files")
    return 0



  def setGirderClient(self, girderClient,tempFileDir):
    self.girderClient = girderClient
    self.tempFileDir = tempFileDir


  def setGirderAPIURL(self,URL):
    self.girderURL = URL


  def setCSVSavePath(self,filePath,fileName):
    if ".csv" in fileName:
      self.csvSavePath = os.path.join(filePath,fileName)
    else:
      self.csvSavePath = os.path.join(filePath,fileName+".csv")

  def setLabels(self,Labels):
    self.Labels = Labels

  def setSelectedVideoIDs(self,selectedVideoIDNames,selectedVideoGirderIDS=None,numberOfFolds=1,train=None,val=None,test=None):
    if selectedVideoGirderIDS != None:
      self.trainList,self.trainGirderIDs,self.valList,self.valGirderIDs,self.testList,self.testGirderIDs = self.selectByPercentage(selectedVideoIDNames,numberOfFolds,train,val,test,selectedVideoGirderIDS)
    else:
      self.trainList,self.valList,self.testList = self.selectByPercentage(selectedVideoIDNames,numberOfFolds,train,val,test)

  def setTrainValTestSet(self,trainSet,valSet,testSet):
    self.trainList = trainSet
    self.valList = valSet
    self.testList = testSet

  def setTrainValTestGirderIDs(self,trainSet,valSet,testSet):
    self.trainGirderIDs = trainSet
    self.valGirderIDs = valSet
    self.testGirderIDs = testSet

  def selectByPercentage(self,selectedVideoIDNames,numFolds,train,val,test,selectedVideoGirderIDs = None):
    if numFolds == 1:
      if selectedVideoGirderIDs == None:
        trainValVideos,testVideos,_,_ = train_test_split(selectedVideoIDNames,selectedVideoIDNames,test_size=(test/100.0))
        trainVideos,valVideos,_,_ = train_test_split(trainValVideos,trainValVideos,test_size=(val/100.0)/(1-(test/100.0)))
      else:
        trainValVideos,testVideos,trainValGirderIDs,testGirderIDs = train_test_split(selectedVideoIDNames,selectedVideoGirderIDs,test_size=(test/100.0))
        trainVideos, valVideos, trainGirderIDs, valGirderIDs = train_test_split(trainValVideos, trainValGirderIDs,test_size=(val / 100.0) / (1 - (test / 100.0)))
      if selectedVideoGirderIDs == None:
        return ([trainVideos], [valVideos], [testVideos])
      else:
        return ([trainVideos], [trainGirderIDs], [valVideos], [valGirderIDs], [testVideos], [testGirderIDs])
    else:
      kf = KFold(n_splits=numFolds)
      trainValIndexes=[]
      testIndexes = []
      for train_idx,test_idx in kf.split(selectedVideoIDNames):
        trainValIndexes.append(train_idx)
        testIndexes.append(test_idx)
      trainVideos = []
      valVideos = []
      testVideos = []
      if selectedVideoGirderIDs != None:
        trainGirderIDs = []
        valGirderIDs = []
        testGirderIDs = []
      for i in range(len(trainValIndexes)):
        fold = []
        if selectedVideoGirderIDs!=None:
          foldGirderIDs = []
        for j in range(len(trainValIndexes[i])):
          fold.append(selectedVideoIDNames[trainValIndexes[i][j]])
          if selectedVideoGirderIDs != None:
            foldGirderIDs.append(selectedVideoGirderIDs[trainValIndexes[i][j]])
        if selectedVideoGirderIDs != None:
          foldtrainVideos,foldvalVideos,foldtrainGirderIDs,foldvalGirderIDs = train_test_split(fold,foldGirderIDs,test_size=(val/100.0))
          trainVideos.append(foldtrainVideos)
          trainGirderIDs.append(foldtrainGirderIDs)
          valVideos.append(foldvalVideos)
          valGirderIDs.append(foldvalGirderIDs)
        else:
          foldtrainVideos,foldvalVideos,_,_ = train_test_split(fold,fold,test_size=(val/100.0))
          trainVideos.append(foldtrainVideos)
          valVideos.append(foldvalVideos)
      for i in range(len(testIndexes)):
        fold = []
        if selectedVideoGirderIDs != None:
          foldGirderIDs = []
        for j in range(len(testIndexes[i])):
          fold.append(selectedVideoIDNames[testIndexes[i][j]])
          if selectedVideoGirderIDs != None:
            foldGirderIDs.append(selectedVideoGirderIDs[testIndexes[i][j]])
        if selectedVideoGirderIDs != None:
          testVideos.append(fold)
          testGirderIDs.append(foldGirderIDs)
        else:
          testVideos.append(fold)
    if selectedVideoGirderIDs == None:
      return (trainVideos,valVideos,testVideos)
    else:
      return (trainVideos,trainGirderIDs,valVideos,valGirderIDs,testVideos,testGirderIDs)




#
# TrainNeuralNetTest
#

class TrainNeuralNetTest(ScriptedLoadableModuleTest):
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
    self.test_TrainNeuralNet1()

  def test_TrainNeuralNet1(self):
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

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 50

    # Test the module logic

    logic = TrainNeuralNetLogic()

    # Test algorithm with non-inverted threshold
    logic.run(inputVolume, outputVolume, threshold, True)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], threshold)

    # Test algorithm with inverted threshold
    logic.run(inputVolume, outputVolume, threshold, False)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], inputScalarRange[1])

    self.delayDisplay('Test passed')
