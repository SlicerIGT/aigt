import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import json

#
# LiveFeedbackModule
#

class LiveFeedbackModule(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "LiveFeedbackModule" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
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
# LiveFeedbackModuleWidget
#

class LiveFeedbackModuleWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    self.logic = LiveFeedbackModuleLogic()
    
    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.volumeNodeSelector = slicer.qMRMLNodeComboBox()
    self.volumeNodeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.volumeNodeSelector.selectNodeUponCreation = True
    self.volumeNodeSelector.addEnabled = False
    self.volumeNodeSelector.removeEnabled = False
    self.volumeNodeSelector.noneEnabled = False
    self.volumeNodeSelector.showHidden = False
    self.volumeNodeSelector.showChildNodeTypes = False
    self.volumeNodeSelector.setMRMLScene( slicer.mrmlScene )
    self.volumeNodeSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input volume: ", self.volumeNodeSelector)
    
    self.textNodeSelector = slicer.qMRMLNodeComboBox()
    self.textNodeSelector.nodeTypes = ["vtkMRMLTextNode"]
    self.textNodeSelector.selectNodeUponCreation = True
    self.textNodeSelector.addEnabled = False
    self.textNodeSelector.removeEnabled = False
    self.textNodeSelector.noneEnabled = False
    self.textNodeSelector.showHidden = False
    self.textNodeSelector.showChildNodeTypes = False
    self.textNodeSelector.setMRMLScene( slicer.mrmlScene )
    self.textNodeSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("CentralLine position: ", self.textNodeSelector)

    # connections
    self.volumeNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.textNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.logic.setTextNode(self.textNodeSelector.currentNode())
    self.logic.setVolumeNode(self.volumeNodeSelector.currentNode())

#
# LiveFeedbackModuleLogic
#

class LiveFeedbackModuleLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self.textNode = None
    self.textNodeObserverTag = None
    self.volumeNode = None
    self.fiducialNode = None

  
  def setConnectorNode(self, connectorNode):
    """
    """
    pass
    
  def setVolumeNode(self, volumeNode):
    """
    """
    self.volumeNode = volumeNode
    
  def setTextNode(self, textNode):
    """
    """
    if self.textNode is textNode:
      return
      
    if self.textNode is not None:
      self.textNode.RemoveObserver(self.textNodeObserverTag)
    
    self.textNode = textNode
    
    if self.textNode is not None:
      self.textNodeObserverTag = self.textNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onTextNodeModified)
    else:
      self.textNodeObserverTag = None
  
  def onTextNodeModified(self, object, event):
    """
    """
    if self.textNode is None or self.volumeNode is None:
      return
    text = self.textNode.GetText()
    predictedLocation = json.loads(text)

    extent = self.volumeNode.GetImageData().GetExtent()
    i = (1.0-predictedLocation[0]) * extent[1]-extent[0]
    j = predictedLocation[1] * extent[3]-extent[2]
    k = 0.0
    #print str(i) + ", " + str(j) + ", " + str(k)
    
    mat = vtk.vtkMatrix4x4()
    self.volumeNode.GetIJKToRASMatrix(mat)
    rasPoint = mat.MultiplyPoint([i,j,k,1])
    
    if self.fiducialNode is None:
      self.fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    if self.fiducialNode.GetNumberOfFiducials() < 1:
      self.fiducialNode.AddFiducial(0,0,0)
    self.fiducialNode.SetNthFiducialPosition(0, rasPoint[0], rasPoint[1], rasPoint[2])

class LiveFeedbackModuleTest(ScriptedLoadableModuleTest):
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
    self.test_LiveFeedbackModule1()

  def test_LiveFeedbackModule1(self):
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
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = LiveFeedbackModuleLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
