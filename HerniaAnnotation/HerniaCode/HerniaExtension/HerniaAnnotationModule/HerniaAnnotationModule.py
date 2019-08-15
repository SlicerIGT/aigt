import os
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import scipy.ndimage

from keras.models import load_model
import cv2


#
# HerniaAnnotationModule
#

class HerniaAnnotationModule(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Hernia Annotation"
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["Tamas Ungi (Perk Lab), Jacob Laframboise (Perk Lab)"]
    self.parent.helpText = """
This extensions annotates open surgical tools in a hernia phantom. 
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# HerniaAnnotationModuleWidget
#

class HerniaAnnotationModuleWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    self.logic = HerniaAnnotationModuleLogic()
    
    ScriptedLoadableModuleWidget.setup(self)
    
    self.detectionOn = False
    
    self.updateTimer = qt.QTimer()
    self.updateTimer.setInterval(100)
    self.updateTimer.setSingleShot(True)
    self.updateTimer.connect('timeout()', self.onUpdateTimer)

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
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLStreamingVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    self.toolOneSelector = slicer.qMRMLNodeComboBox()
    self.toolOneSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.toolOneSelector.selectNodeUponCreation = True
    self.toolOneSelector.addEnabled = False
    self.toolOneSelector.removeEnabled = False
    self.toolOneSelector.noneEnabled = False
    self.toolOneSelector.showHidden = False
    self.toolOneSelector.showChildNodeTypes = False
    self.toolOneSelector.setMRMLScene(slicer.mrmlScene)
    self.toolOneSelector.setToolTip("Pick the input to the algorithm.")
    parametersFormLayout.addRow("Input Tool One: ", self.toolOneSelector)

    self.toolTwoSelector = slicer.qMRMLNodeComboBox()
    self.toolTwoSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.toolTwoSelector.selectNodeUponCreation = True
    self.toolTwoSelector.addEnabled = False
    self.toolTwoSelector.removeEnabled = False
    self.toolTwoSelector.noneEnabled = False
    self.toolTwoSelector.showHidden = False
    self.toolTwoSelector.showChildNodeTypes = False
    self.toolTwoSelector.setMRMLScene(slicer.mrmlScene)
    self.toolTwoSelector.setToolTip("Pick the input to the algorithm.")
    parametersFormLayout.addRow("Input Tool Two: ", self.toolTwoSelector)


    self.modelPathEdit = ctk.ctkPathLineEdit()
    parametersFormLayout.addRow("Keras model: ", self.modelPathEdit)
    
    #
    # threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 0.05
    self.imageThresholdSliderWidget.minimum = 0
    self.imageThresholdSliderWidget.maximum = 1.0
    self.imageThresholdSliderWidget.value = 0.5
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for class probability.")
    parametersFormLayout.addRow("Prediction threshold", self.imageThresholdSliderWidget)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Start detection")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = True
    parametersFormLayout.addRow(self.applyButton)

    self.toolLabel = qt.QLabel("0")
    toolFont = self.toolLabel.font
    toolFont.setPointSize(32)
    self.toolLabel.setFont(toolFont)
    parametersFormLayout.addRow("Tool: ", self.toolLabel)



    self.classLabel = qt.QLabel("0")
    classFont = self.classLabel.font
    classFont.setPointSize(32)
    self.classLabel.setFont(classFont)
    parametersFormLayout.addRow("Tissue: ", self.classLabel)
    
    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    
    # Add vertical spacer
    self.layout.addStretch(1)


  def onUpdateTimer(self):
    if self.detectionOn:
      newText = self.logic.getLastClass()
      self.classLabel.setText(newText)
      newText2 = self.logic.getToolInUse()
      self.toolLabel.setText(newText2)
      self.updateTimer.start()
    else:
      self.classLabel.setText("")
    

  def cleanup(self):
    pass
  
  
  def setDetection(self, currentState):
    self.detectionOn = currentState
    if self.detectionOn is True:
      self.applyButton.setText("Stop detection")
    else:
      self.applyButton.setText("Start detection")
  
  
  def onApplyButton(self):
    imageThreshold = self.imageThresholdSliderWidget.value
    modelFilePath = self.modelPathEdit.currentPath
    
    # Try to load Keras model
    
    success = self.logic.loadKerasModel(modelFilePath)
    if not success:
      logging.error("Failed to load Keras model: {}".format(modelFilePath))
      self.setDetection(False)
      return
    
    inputVolumeNode = self.inputSelector.currentNode()
    if inputVolumeNode is None:
      logging.error("Please select a valid image node!")
      self.setDetection(False)
      return

    inputToolOneNode = self.toolOneSelector.currentNode()
    if inputToolOneNode is None:
      logging.error("Please select a valid transform one node!")
      self.setDetection(False)
      return

    inputToolTwoNode = self.toolTwoSelector.currentNode()
    if inputToolTwoNode is None:
      logging.error("Please select a valid transform two node!")
      self.setDetection(False)
      return

    if inputToolOneNode == inputToolTwoNode:
      logging.error("The transforms cannot be the same!")
      self.setDetection(False)
      return

    success = self.logic.run(inputVolumeNode, imageThreshold, inputToolOneNode, inputToolTwoNode)
    if not success:
      logging.error("Could not start classification!")
      self.setDetection(False)
      return
    
    if self.detectionOn is True:
      self.setDetection(False)
      return
    else:
      self.setDetection(True)
      self.updateTimer.start()
    

#
# HerniaAnnotationModuleLogic
#

class HerniaAnnotationModuleLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  
  def __init__(self):
    self.model = None
    self.observerTag = None
    self.lastObservedVolumeId = None
    self.lastClass = ""
    self.toolInUse = ""
    self.model_input_size = None
    self.classes = ['None', 'Extob', 'Fat', 'Sack', 'Skin', 'Spchd']
    self.predictionThreshold = 0.0
    self.toolOnePositions = []
    self.toolTwoPositions = []

  
  def getLastClass(self):
    return self.lastClass

  def getToolInUse(self):
    return self.toolInUse
  
  def loadKerasModel(self, modelFilePath):
    """
    Tries to load Keras model for classifiation
    :param modelFilePath: full path to saved model file
    :return: True on success, False on error
    """
    try:
      self.model = load_model(modelFilePath)
    except:
      self.model = None
      return False
    
    return True
  

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


  def run(self, inputVolumeNode, imageThreshold, transformOneNode, transformTwoNode):
    """
    Run the classification algorithm on each new image
    """

    import math
    self.count =0

    if self.model is None:
      logging.error('Cannot run classification without model!')
      return False
    
    self.predictionThreshold = imageThreshold
    self.transformOneNode = transformOneNode
    self.transformTwoNode = transformTwoNode


    image = inputVolumeNode.GetImageData()
    shape = list(image.GetDimensions())
    shape.reverse()
    components = image.GetNumberOfScalarComponents()
    if components > 1:
      shape.append(components)
      shape.remove(1)
    self.model_input_size = self.model.layers[0].input_shape[1]
    
    if self.observerTag is None:
      self.lastObservedVolumeId = inputVolumeNode.GetID()
      self.observerTag = inputVolumeNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onImageModified)
      logging.info('Processing started')
    else:
      lastVolumeNode = slicer.util.getNode(self.lastObservedVolumeId)
      if lastVolumeNode is not None:
        lastVolumeNode.RemoveObserver(self.observerTag)
        self.observerTag = None
        self.lastObservedVolumeId = None
      logging.info('Processing ended')
    
    return True
  

  def frobeniusNorm(self, matrix):
    sum = 0
    for i in range(4):
      for j in range(4):
        sum += matrix.GetElement(i, j) ** 2
    return math.sqrt(sum)


  def onImageModified(self, caller, event):
    logging.debug("Count == {}".format(self.count))
    print("Count == {}".format(self.count))
    image_node = slicer.util.getNode(self.lastObservedVolumeId)
    image = image_node.GetImageData()
    shape = list(image.GetDimensions())
    shape.reverse()
    components = image.GetNumberOfScalarComponents()
    if components > 1:
      shape.append(components)
      shape.remove(1)
    input_array = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape)

    cropped_size = 256
    scaled_size = 128

    # Resize image and scale between 0.0 and 1.0
    #logging.info("The orig array was: " + str(input_array.shape))
    #logging.info(input_array[127][127])
    resized_input_array = cv2.resize(input_array[70:70+cropped_size, 150:150+cropped_size], dsize=(scaled_size, scaled_size))
    resized_input_array = resized_input_array / 255.0
    resized_input_array = np.expand_dims(resized_input_array, 0)
    #np.save(r"C:\Users\PerkLab\Desktop\image111.npy", resized_input_array)
    #logging.info("The cropped and resized array is: " + str(resized_input_array.shape))
    #logging.info(resized_input_array[0][127][127])
    #logging.info(resized_input_array.dtype)
    
    # Run prediction and print result
    #print(self.model.summary())
    prediction = self.model.predict_proba(resized_input_array)
    #logging.info("Prediction is: " + str(prediction))
    #logging.info(prediction.shape)
    #logging.info(prediction.dtype)
    maxPredictionIndex = np.argmax(prediction[0])
    #logging.info("Max prediction Index is: " + str(maxPredictionIndex))
    maxPredictionConfidence = prediction[0][maxPredictionIndex]
    #logging.info("Max pred confidence is: " + str(maxPredictionConfidence))
    assert 1 >= maxPredictionConfidence >= 0

    if maxPredictionConfidence > self.predictionThreshold:
      self.lastClass = self.classes[maxPredictionIndex]
    else:
      self.lastClass = "Null"

    # Check which tool is being used.

    self.toolOneTransform = self.transformOneNode.GetTransformToParent()
    self.toolOneMatrix = self.toolOneTransform.GetMatrix()
    self.toolOnePosition = np.array(self.toolOneMatrix.MultiplyFloatPoint((0,0,0,1)))

    if self.count<5:
      self.toolInUse = "Null"
      self.toolOnePositions.append(self.toolOnePosition)
    else:
      self.toolOnePositions.append(self.toolOnePosition)
      del self.toolOnePositions[0]
      movementOne = np.array(self.toolOnePositions[-1]) - np.array(self.toolOnePositions[0])
      movementOneSum = np.sum(movementOne ** 2)


    self.toolTwoTransform = self.transformTwoNode.GetTransformToParent()
    self.toolTwoMatrix = self.toolTwoTransform.GetMatrix()
    self.toolTwoPosition = np.array(self.toolTwoMatrix.MultiplyFloatPoint((0, 0, 0, 1)))

    if self.count < 5:
      self.toolInUse = "Null"
      self.toolTwoPositions.append(self.toolTwoPosition)
    else:
      self.toolTwoPositions.append(self.toolTwoPosition)
      del self.toolTwoPositions[0]
      movementTwo = self.toolTwoPositions[-1] - self.toolTwoPositions[0]
      movementTwoSum = np.sum(movementTwo ** 2)

    if self.count>4 and movementOneSum > movementTwoSum:
      self.toolInUse = "1"
    else:
      self.toolInUse = "2"

    
    print("Prediction: {} at {:2.2%} probability".format(self.lastClass, maxPredictionConfidence))
    self.count += 1
    

class HerniaAnnotationModuleTest(ScriptedLoadableModuleTest):
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
    self.test_HerniaAnnotationModule1()

  def test_HerniaAnnotationModule1(self):
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
    self.assertIsNotNone( self.logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
