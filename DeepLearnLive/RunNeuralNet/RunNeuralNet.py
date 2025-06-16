import os
import unittest
import numpy
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import subprocess
from pathlib import Path
import time
import socket

#
# RunNeuralNet
#

class RunNeuralNet(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Run Neural Net"  # TODO: make this more human readable by adding spaces
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
    networks = os.listdir(self.modelDirectoryFilePath)
    networks = [x for x in networks if not '.' in x]
    self.networkTypeSelector.addItems(networks)
    self.networkType = "Select network type"
    parametersFormLayout.addRow(self.networkTypeSelector)

    self.modelSelector = qt.QComboBox()
    self.modelSelector.addItems(["Select model"])
    self.modelName = "Select model"
    parametersFormLayout.addRow(self.modelSelector)

    #self.modelNameLineEdit = qt.QLineEdit()
    #self.modelNameLineEdit.text = 'Model Name'
    #parametersFormLayout.addRow(self.modelNameLineEdit)

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
    self.plusServerOutgoingPortLineEdit.setText("18946")
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
    self.logic.setPathToCondaExecutable(self.condaDirectoryPath)
    self.condaDirectoryPathSelector.directory = self.condaDirectoryPath
    condaSettingsLayout.addRow(self.condaDirectoryPathSelector)

    self.environmentNameLineEdit = qt.QLineEdit("EnvironmentName")
    self.environmentName = "kerasGPUEnv2"
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


    self.modelDirectoryFilePathSelector.connect('directorySelected(QString)',self.onNetworkDirectorySelected)
    self.condaDirectoryPathSelector.connect('directorySelected(QString)',self.onCondaDirectorySelected)
    self.environmentNameLineEdit.connect('textChanged(QString)',self.onCondaEnvironmentNameChanged)
    self.networkTypeSelector.connect('currentIndexChanged(int)',self.onNetworkTypeSelected)
    self.modelSelector.connect('currentIndexChanged(int)',self.onModelSelected)
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
    self.modelDirectoryFilePath = self.modelDirectoryFilePathSelector.directory
    currentItems = self.networkTypeSelector.count
    for i in range(currentItems,-1,-1):
      self.networkTypeSelector.removeItem(i)
    networks = os.listdir(self.modelDirectoryFilePath)
    networks = [x for x in networks if not '.' in x and x[0] != '_']
    networks = ["Select network type"] + networks
    self.networkTypeSelector.addItems(networks)

  def onModelSelected(self):
    self.modelName = self.modelSelector.currentText

  def onCondaDirectorySelected(self):
    self.condaDirectoryPath = self.condaDirectoryPathSelector.directory
    self.logic.setPathToCondaExecutable(self.condaDirectoryPath)

  def onCondaEnvironmentNameChanged(self):
    self.environmentName = self.environmentNameLineEdit.text
    self.logic.setCondaEnvironmentName(self.environmentName)

  def onNetworkTypeSelected(self):
    self.networkType = self.networkTypeSelector.currentText
    currentItems = self.modelSelector.count
    for i in range(currentItems, -1, -1):
      self.modelSelector.removeItem(i)
    self.modelSelector.addItem("Select model")
    if self.networkType != "Select network type":
      networks = os.listdir(os.path.join(self.modelDirectoryFilePath, self.networkType))
      networks = [x for x in networks if not '.' in x and "pycache" not in x]
      self.modelSelector.addItems(networks)


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
    #self.logic.setPathToCondaExecutable(self.condaDirectoryPath)
    #self.logic.setCondaEnvironmentName(self.environmentName)
    self.logic.setInputNode(self.inputNodeSelector.currentNode())
    self.logic.setOutputNode(self.outputNodeSelector.currentNode())
    outgoingHostName = self.plusServerOutgoingHostNameLineEdit.text
    outgoingPort = self.plusServerOutgoingPortLineEdit.text
    incomingHostName = self.plusServerIncomingHostNameLineEdit.text
    incomingPort = self.plusServerIncomingPortLineEdit.text
    self.logic.setHostNameAndPort(incomingHostName, int(incomingPort), 'incoming')
    self.logic.setHostNameAndPort(outgoingHostName, int(outgoingPort), 'outgoing')
    self.logic.setNetworkName(self.modelName)
    self.logic.setNetworkPath(os.path.join(self.modelDirectoryFilePathSelector.directory,self.networkType,self.modelName))
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
    try:
      self.moduleDir = os.path.dirname(slicer.modules.runneuralnet.path)
    except:
      self.moduleDir = None

  def startNeuralNetwork(self,minimized = False):
    if self.moduleDir == None:
      self.moduleDir = os.path.dirname(slicer.modules.runneuralnet.path)
    self.registerIncomingAndOutgoingNodes()
    self.incomingConnectorNode.Start()
    self.outgoingConnectorNode.Start()
    this_Host = self.getIPAddress()
    if self.incomingHostName == "localhost" or self.incomingHostName == "127.0.0.1" or self.incomingHostName == this_Host:
      cmd = [fr"{self.moduleDir}\Scripts\\openCMDPrompt.bat",
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
      print(self.condaEnvName)
      if minimized:
        strCMD = "START /MIN "
      else:
        strCMD = ''
      strCMD += cmd[0]
      for i in range(1,len(cmd)):
        strCMD = strCMD + ' ' + cmd[i]
      startupEnv = slicer.util.startupEnvironment()
      info = subprocess.STARTUPINFO()
      info.dwFlags = subprocess.CREATE_NEW_CONSOLE
      p = subprocess.Popen(cmd,creationflags=subprocess.CREATE_NEW_CONSOLE,env=startupEnv)
    startTime = time.time()
    while self.incomingConnectorNode.GetState() != 2 and self.outgoingConnectorNode.GetState() != 2 and time.time()-startTime<15:
      time.sleep(0.25)
    if self.incomingConnectorNode.GetState() !=2:
      logging.info("Failed to connect to neural network")
    else:
      logging.info("Connected to neural network")
  def handle_state(self,state):
    states = {qt.QProcess.NotRunning: 'Not running',
            qt.QProcess.Starting: 'Starting',
            qt.QProcess.Running: 'Running'}
    self.state_name = states[state]
    logging.info(self.state_name)
  def handle_stdout(self):
    data = self.p.readAllStandardOutput()
    #stdout = bytes(data).decode("utf8")
    stdout = data.data()
    logging.info(stdout)
  def handle_stderr(self):
    data = self.p.readAllStandardOutput()
    #stdout = bytes(data).decode("utf8")
    stderr = data.data()
    logging.info(stderr)

  def getIPAddress(self):
      hostname = socket.gethostbyname(socket.gethostname())
      return hostname

  def stopNeuralNetwork(self):
    '''if self.state_name != None:
      self.p.kill()'''
    self.outgoingConnectorNode.UnregisterOutgoingMRMLNode(self.inputNode)
    self.incomingConnectorNode.UnregisterIncomingMRMLNode(self.outputNode)
    self.outgoingConnectorNode.Stop()
    self.incomingConnectorNode.Stop()
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
        self.videoImage.SetAndObserveImageData(node.GetImageData())
        node.AddObserver(slicer.vtkMRMLStreamingVolumeNode.FrameModifiedEvent, self.referenceImageModified)
        self.streamingNode = node
      except slicer.util.MRMLNodeNotFoundException:
        #Create a node to store the image data of the video so that IGTLink is
        # sending an IMAGE message, not a VIDEO message
        imageSpacing = [0.2, 0.2, 0.2]
        # Create volume node
        self.videoImage = slicer.vtkMRMLVectorVolumeNode()
        self.videoImage.SetName(node.GetName() + '_Image')
        self.videoImage.SetSpacing(imageSpacing)
        self.videoImage.SetAndObserveImageData(node.GetImageData())
        #self.videoImage.SetImageDataConnection(node.GetImageDataConnection())
        node.AddObserver(slicer.vtkMRMLStreamingVolumeNode.FrameModifiedEvent, self.referenceImageModified)
        # Add volume to scene
        slicer.mrmlScene.AddNode(self.videoImage)
      self.inputNode = self.videoImage
      self.streamingNode= node
    else:
      self.inputNode = node

  def referenceImageModified(self,caller,eventId):
    self.videoImage.SetAndObserveImageData(self.streamingNode.GetImageData())
    self.videoImage.UpdateScene(slicer.mrmlScene)
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
      self.outgoingConnectorNode.SetTypeServer(int(outgoingPort))
    except slicer.util.MRMLNodeNotFoundException:
      self.outgoingConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      self.outgoingConnectorNode.SetName('OutgoingPlusConnector')
      slicer.mrmlScene.AddNode(self.outgoingConnectorNode)
      self.outgoingConnectorNode.SetTypeServer(int(outgoingPort))
      logging.debug('Outgoing Connector Created')
    try:
      self.incomingConnectorNode = slicer.util.getNode('IncomingPlusConnector')
      self.incomingConnectorNode.SetTypeClient(incomingHostname,int(incomingPort))
    except slicer.util.MRMLNodeNotFoundException:
      self.incomingConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      self.incomingConnectorNode.SetName('IncomingPlusConnector')
      slicer.mrmlScene.AddNode(self.incomingConnectorNode)
      self.incomingConnectorNode.SetTypeClient(incomingHostname,int(incomingPort))
      logging.debug('Incoming Connector Created')

  def registerIncomingAndOutgoingNodes(self):
    self.setupIGTLinkConnectors(self.incomingHostName,self.incomingPort,self.outgoingPort)
    #self.incomingConnectorNode.RegisterIncomingMRMLNode(self.outputNode,self.outputNode.GetName())
    self.outgoingConnectorNode.RegisterOutgoingMRMLNode(self.inputNode)

  def createNewModel(self,newModelName,newModelLocation = None):
    if self.moduleDir == None:
      self.moduleDir = os.path.dirname(slicer.modules.runneuralnet.path)
    templateModelFilePath = os.path.join(self.moduleDir,"Scripts","TemplateNetworkFile.txt")
    if newModelLocation == None:
      newModelPath = os.path.join(self.moduleDir,os.pardir,"Networks",newModelName)
    else:
      newModelPath = os.path.join(newModelLocation,newModelName)
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
