import os
import unittest
import numpy
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import subprocess
import pandas
import cv2
from pathlib import Path
#
# RunNeuralNet
#

class RunNeuralNet(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "RunNeuralNet"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Deep Learn Live"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Rebecca Hisey (Queen's University)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This module runs various neural networks in realtime in Slicer.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

#
# RunNeuralNetWidget
#

class RunNeuralNetWidget(ScriptedLoadableModuleWidget):

  def setup(self):

    ScriptedLoadableModuleWidget.setup(self)

    self.logic = RunNeuralNetLogic()
    self.moduleDir = os.path.dirname(slicer.modules.runneuralnet.path)

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    self.modelDirectoryFilePathSelector = ctk.ctkDirectoryButton()
    self.modelDirectoryFilePath = os.path.join(self.moduleDir,os.pardir,"Networks")
    self.modelDirectoryFilePathSelector.directory = self.modelDirectoryFilePath
    parametersFormLayout.addRow(self.modelDirectoryFilePathSelector)

    self.networkTypeSelector = qt.QComboBox()
    self.networkTypeSelector.addItems(["Select network type"])
    networks = os.listdir(os.path.join(self.moduleDir,os.pardir,'Networks'))
    networks = [x for x in networks if not '.' in x]
    self.networkTypeSelector.addItems(networks)
    self.networkType = "Select network output type"
    parametersFormLayout.addRow(self.networkTypeSelector)

    self.modelNameLineEdit = qt.QLineEdit()
    self.modelNameLineEdit.text = 'Model Name'
    parametersFormLayout.addRow(self.modelNameLineEdit)

    self.outputTypeSelector = qt.QComboBox()
    self.outputTypeSelector.addItems(["Select network output type","IMAGE","TRANSFORM","STRING"])
    self.outputType = "Select network output type"
    parametersFormLayout.addRow(self.outputTypeSelector)

    self.inputNodeSelector = slicer.qMRMLNodeComboBox()
    self.inputNodeSelector.selectNodeUponCreation = True
    self.inputNodeSelector.nodeTypes = (("vtkMRMLScalarVolumeNode","vtkMRMLVectorVolumeNode","vtkMRMLStreamingVolumeNode"))
    self.inputNodeSelector.addEnabled = True
    self.inputNodeSelector.removeEnabled = False
    self.inputNodeSelector.editEnabled = True
    self.inputNodeSelector.renameEnabled = True
    self.inputNodeSelector.noneEnabled = False
    self.inputNodeSelector.showHidden = False
    self.inputNodeSelector.showChildNodeTypes = False
    self.inputNodeSelector.setMRMLScene(slicer.mrmlScene)
    parametersFormLayout.addRow("Image Node: ", self.inputNodeSelector)

    self.outputNodeSelector = slicer.qMRMLNodeComboBox()
    self.outputNodeSelector.selectNodeUponCreation = True
    self.outputNodeSelector.nodeTypes = ((),"")
    self.outputNodeSelector.addEnabled = True
    self.outputNodeSelector.removeEnabled = False
    self.outputNodeSelector.editEnabled = True
    self.outputNodeSelector.renameEnabled = True
    self.outputNodeSelector.noneEnabled = False
    self.outputNodeSelector.showHidden = False
    self.outputNodeSelector.showChildNodeTypes = False
    self.outputNodeSelector.setMRMLScene(slicer.mrmlScene)
    parametersFormLayout.addRow("Label Node: ", self.outputNodeSelector)

    self.plusServerIncomingHostNameLineEdit = qt.QLineEdit("localhost")
    self.hostnameLabel = qt.QLabel()
    self.hostnameLabel.setText("Incoming Hostname: ")
    self.plusServerIncomingPortLineEdit = qt.QLineEdit()
    self.plusServerIncomingPortLineEdit.setText("18944")
    self.portLabel = qt.QLabel()
    self.portLabel.setText("Port: ")
    incominghbox = qt.QHBoxLayout()
    incominghbox.addWidget(self.hostnameLabel)
    incominghbox.addWidget(self.plusServerIncomingHostNameLineEdit)
    incominghbox.addWidget(self.portLabel)
    incominghbox.addWidget(self.plusServerIncomingPortLineEdit)
    parametersFormLayout.addRow(incominghbox)

    self.plusServerOutgoingHostNameLineEdit = qt.QLineEdit("localhost")
    self.outgoinghostnameLabel = qt.QLabel()
    self.outgoinghostnameLabel.setText("Outgoing Hostname: ")
    self.plusServerOutgoingPortLineEdit = qt.QLineEdit()
    self.plusServerOutgoingPortLineEdit.setText("18945")
    self.outgoingportLabel = qt.QLabel()
    self.outgoingportLabel.setText("Port: ")
    outgoinghbox = qt.QHBoxLayout()
    outgoinghbox.addWidget(self.outgoinghostnameLabel)
    outgoinghbox.addWidget(self.plusServerOutgoingHostNameLineEdit)
    outgoinghbox.addWidget(self.outgoingportLabel)
    outgoinghbox.addWidget(self.plusServerOutgoingPortLineEdit)
    parametersFormLayout.addRow(outgoinghbox)

    self.runNeuralNetworkButton = qt.QPushButton("Start Network")
    self.runNeuralNetworkButton.enabled = False
    self.networkRunning = False
    parametersFormLayout.addRow(self.runNeuralNetworkButton)

    condaSettingsCollapsibleButton = ctk.ctkCollapsibleButton()
    condaSettingsCollapsibleButton.text = "Conda settings"
    parametersFormLayout.addRow(condaSettingsCollapsibleButton)
    condaSettingsLayout = qt.QFormLayout(condaSettingsCollapsibleButton)
    condaSettingsCollapsibleButton.collapsed = True

    self.condaDirectoryPathSelector = ctk.ctkDirectoryButton()
    self.condaDirectoryPath = self.getCondaPath()
    self.condaDirectoryPathSelector.directory = self.condaDirectoryPath
    condaSettingsLayout.addRow(self.condaDirectoryPathSelector)

    self.environmentNameLineEdit = qt.QLineEdit("EnvironmentName")
    self.environmentName = "kerasGPUEnv"
    condaSettingsLayout.addRow(self.environmentNameLineEdit)

    createNewModelCollapsibleButton = ctk.ctkCollapsibleButton()
    createNewModelCollapsibleButton.text = "Create New Model"
    parametersFormLayout.addRow(createNewModelCollapsibleButton)
    newModelSettingsLayout = qt.QFormLayout(createNewModelCollapsibleButton)
    createNewModelCollapsibleButton.collapsed = True

    self.newModelNameLineEdit = qt.QLineEdit()
    self.newModelNameLineEdit.setText("Model Name")
    newModelSettingsLayout.addRow(self.newModelNameLineEdit)

    self.createNewModelButton = qt.QPushButton("Create")
    self.createNewModelButton.enabled = False
    newModelSettingsLayout.addRow(self.createNewModelButton)


    self.modelDirectoryFilePathSelector.connect('directorySelected()',self.onNetworkDirectorySelected)
    self.condaDirectoryPathSelector.connect('directorySelected()',self.onCondaDirectorySelected)
    self.environmentNameLineEdit.connect('textChanged(QString)',self.onCondaEnvironmentNameChanged)
    self.networkTypeSelector.connect('currentIndexChanged(int)',self.onNetworkTypeSelected)
    self.outputTypeSelector.connect('currentIndexChanged(int)',self.onOutputTypeSelected)
    self.inputNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.runNeuralNetworkButton.connect('clicked(bool)',self.onStartNetworkClicked)
    self.newModelNameLineEdit.connect('textChanged(QString)',self.onNewModelNameChanged)
    self.createNewModelButton.connect('clicked(bool)',self.onCreateModelClicked)

  def getCondaPath(self):
    condaPath = str(Path.home())
    homePath = str(Path.home())
    if "Anaconda3" in os.listdir(homePath):
        condaPath = os.path.join(homePath,"Anaconda3")
    return condaPath


  def cleanup(self):
    pass

  def onNetworkDirectorySelected(self):
    currentItems = self.networkTypeSelector.items
    for i in currentItems:
      if i != "Select network type":
        self.networkTypeSelector.removeItem(i)
    networks = os.listdir(os.path.join(self.moduleDir, os.pardir, 'Networks'))
    networks = [x for x in networks if not '.' in x]
    self.networkTypeSelector.addItems(networks)

  def onCondaDirectorySelected(self):
    self.condaDirectoryPath = self.condaDirectoryPathSelector.directory

  def onCondaEnvironmentNameChanged(self):
    self.environmentName = self.environmentNameLineEdit.text

  def onNetworkTypeSelected(self):
    self.networkType = self.networkTypeSelector.currentText

  def onOutputTypeSelected(self):
    self.outputType = self.outputTypeSelector.currentText
    if self.outputType =='IMAGE':
      self.outputNodeSelector.setMRMLScene(None)
      self.outputNodeSelector.nodeTypes = (("vtkMRMLScalarVolumeNode","vtkMRMLVectorVolumeNode"))
      self.outputNodeSelector.setMRMLScene(slicer.mrmlScene)
    elif self.outputType =='TRANSFORM':
      self.outputNodeSelector.setMRMLScene(None)
      self.outputNodeSelector.nodeTypes = ["vtkMRMLTransformNode"]
      self.outputNodeSelector.setMRMLScene(slicer.mrmlScene)
    elif self.outputType =='STRING':
      self.outputNodeSelector.setMRMLScene(None)
      self.outputNodeSelector.nodeTypes = ["vtkMRMLTextNode"]
      self.outputNodeSelector.setMRMLScene(slicer.mrmlScene)

  def onSelect(self):
    self.runNeuralNetworkButton.enabled = self.inputNodeSelector.currentNode() and self.outputNodeSelector.currentNode() and self.outputType!= "Select network output type"

  def onStartNetworkClicked(self):
    self.logic.setPathToCondaExecutable(self.condaDirectoryPath)
    self.logic.setCondaEnvironmentName(self.environmentName)
    self.logic.setInputNode(self.inputNodeSelector.currentNode())
    self.logic.setOutputNode(self.outputNodeSelector.currentNode())
    outgoingHostName = self.plusServerOutgoingHostNameLineEdit.text
    outgoingPort = self.plusServerOutgoingPortLineEdit.text
    incomingHostName = self.plusServerIncomingHostNameLineEdit.text
    incomingPort = self.plusServerIncomingPortLineEdit.text
    self.logic.setHostNameAndPort(incomingHostName, int(incomingPort), 'incoming')
    self.logic.setHostNameAndPort(outgoingHostName, int(outgoingPort), 'outgoing')
    self.logic.setNetworkName(self.modelNameLineEdit.text)
    self.logic.setNetworkPath(os.path.join(self.modelDirectoryFilePathSelector.directory,self.networkType))
    self.logic.setNetworkType(self.networkType)
    self.logic.setOutputType(self.outputType)
    if not self.networkRunning:
      self.runNeuralNetworkButton.setText("Stop Network")
      self.logic.startNeuralNetwork()
      self.networkRunning = True
    else:
      self.runNeuralNetworkButton.setText("Start Network")
      self.logic.stopNeuralNetwork()
      self.networkRunning = False

  def onNewModelNameChanged(self):
    self.newModelName = self.newModelNameLineEdit.text
    self.newModelName = self.newModelName.replace(" ","")
    self.createNewModelButton.enabled = True

  def onCreateModelClicked(self):
    self.logic.createNewModel(self.newModelName)
#
# RunNeuralNetLogic
#

class RunNeuralNetLogic(ScriptedLoadableModuleLogic):
  def __init__(self,
               condaPath = None,
               condaEnvName = "kerasGPUEnv",
               networkType = None,
               networkName = None,
               networkPath = None,
               inputNode = None,
               outputType = 'STRING',
               outputNode = None,
               outgoingHostName = 'localhost',
               outgoingPort = None,
               incomingHostName = 'localhost',
               incomingPort = None):
    ScriptedLoadableModuleLogic.__init__(self)
    self.condaPath = condaPath
    self.condaEnvName = condaEnvName
    self.networkType = networkType
    self.networkName = networkName
    self.networkPath = networkPath
    self.inputNode = inputNode
    self.outputNode = outputNode
    self.outputType = outputType
    self.outgoingHostName = outgoingHostName
    self.outgoingPort = outgoingPort
    self.incomingHostName = incomingHostName
    self.incomingPort = incomingPort
    self.moduleDir = os.path.dirname(slicer.modules.runneuralnet.path)

  def startNeuralNetwork(self):
    self.registerIncomingAndOutgoingNodes()
    self.incomingConnectorNode.Start()
    self.outgoingConnectorNode.Start()
    cmd = [str(self.moduleDir + "\Scripts\StartNeuralNet.lnk"),
           str(self.condaPath),
           str(self.condaEnvName),
           str(self.moduleDir + "\Scripts"),
           str(self.networkType),
           str(self.networkPath),
           str(self.networkName),
           str(self.outputType),
           str(self.outgoingHostName),
           str(self.outgoingPort),
           str(self.incomingHostName),
           str(self.incomingPort),
           str(self.outputNode.GetName())]
    self.p = subprocess.Popen(cmd, shell=True)
    logging.info("Starting neural network")

  def stopNeuralNetwork(self):
    self.incomingConnectorNode.Stop()
    self.outgoingConnectorNode.Stop()
    logging.info("Stopping neural network")

  def setPathToCondaExecutable(self,condaPath):
    self.condaPath = condaPath

  def setCondaEnvironmentName(self,name):
    self.condaEnvName = name

  def setNetworkType(self,networkType):
    self.networkType = networkType

  def setNetworkName(self,networkName):
    self.networkName = networkName

  def setNetworkPath(self,networkPath):
    self.networkPath = networkPath

  def setOutputType(self,outputType):
    self.outputType = outputType

  def setInputNode(self,node):
    if node.GetClassName() == "vtkMRMLStreamingVolumeNode":
      try:
        self.videoImage = slicer.util.getNode(node.GetName()+'_Image')
      except slicer.util.MRMLNodeNotFoundException:
        #Create a node to store the image data of the video so that IGTLink is
        # sending an IMAGE message, not a VIDEO message
        imageSpacing = [0.2, 0.2, 0.2]
        # Create volume node
        self.videoImage = slicer.vtkMRMLVectorVolumeNode()
        self.videoImage.SetName(node.GetName() + '_Image')
        self.videoImage.SetSpacing(imageSpacing)
        self.videoImage.SetAndObserveImageData(node.GetImageData())
        node.AddObserver(slicer.vtkMRMLStreamingVolumeNode.FrameModifiedEvent, self.referenceImageModified)
        # Add volume to scene
        slicer.mrmlScene.AddNode(self.videoImage)
      self.inputNode = self.videoImage
    else:
      self.inputNode = node

  def referenceImageModified(self,caller,eventId):
    self.videoImage.InvokeEvent(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent)

  def setOutputNode(self,node):
    self.outputNode = node

  def setHostNameAndPort(self,hostName,port,connectionType='incoming'):
    if connectionType == 'incoming':
      self.incomingHostName = hostName
      self.incomingPort = port
    elif connectionType == 'outgoing':
      self.outgoingHostName = hostName
      self.outgoingPort = port
    else:
      logging.error("Invalid connection type: " + connectionType)

  def setupIGTLinkConnectors(self,incomingHostname,incomingPort,outgoingPort):
    try:
      self.outgoingConnectorNode = slicer.util.getNode('OutgoingPlusConnector')
    except slicer.util.MRMLNodeNotFoundException:
      self.outgoingConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      self.outgoingConnectorNode.SetName('OutgoingPlusConnector')
      slicer.mrmlScene.AddNode(self.outgoingConnectorNode)
      self.outgoingConnectorNode.SetTypeServer(int(outgoingPort))
      logging.debug('Outgoing Connector Created')
    try:
      self.incomingConnectorNode = slicer.util.getNode('IncomingPlusConnector')
    except slicer.util.MRMLNodeNotFoundException:
      self.incomingConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      self.incomingConnectorNode.SetName('IncomingPlusConnector')
      slicer.mrmlScene.AddNode(self.incomingConnectorNode)
      self.incomingConnectorNode.SetTypeClient(incomingHostname,int(incomingPort))
      logging.debug('Incoming Connector Created')

  def registerIncomingAndOutgoingNodes(self):
    self.setupIGTLinkConnectors(self.incomingHostName,self.incomingPort,self.outgoingPort)
    self.incomingConnectorNode.RegisterIncomingMRMLNode(self.outputNode)
    self.outgoingConnectorNode.RegisterOutgoingMRMLNode(self.inputNode)

  def createNewModel(self,newModelName):
    templateModelFilePath = os.path.join(self.moduleDir,"Scripts","TemplateNetworkFile.txt")
    newModelPath = os.path.join(self.moduleDir,os.pardir,"Networks",newModelName)
    templateFile = open(templateModelFilePath,'r')
    templateFileText = templateFile.read()
    templateFile.close()
    os.mkdir(newModelPath)
    newFileText = templateFileText.replace('MODELNAME',newModelName)
    newModelFile = open(os.path.join(newModelPath,newModelName+'.py'),'w')
    newModelFile.write(newFileText)
    newModelFile.close()



#
# RunNeuralNetTest
#

class RunNeuralNetTest(ScriptedLoadableModuleTest):

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()
