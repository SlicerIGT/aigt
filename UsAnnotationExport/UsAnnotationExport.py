import os
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

import csv
import numpy
from sys import float_info


#
# UsAnnotationExport
#

class UsAnnotationExport(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "UsAnnotationExport" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Machine learning"]
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

#
# UsAnnotationExportWidget
#

class UsAnnotationExportWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.logic = UsAnnotationExportLogic()
  

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

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
    # input browser node selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLSequenceBrowserNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input sequence" )
    parametersFormLayout.addRow("Input sequence browser: ", self.inputSelector)

    self.saveDirectoryLineEdit = ctk.ctkPathLineEdit()
    node = self.logic.getParameterNode()
    saveDirectory = node.GetParameter('SaveDirectory')
    self.saveDirectoryLineEdit.currentPath = saveDirectory
    self.saveDirectoryLineEdit.filters = ctk.ctkPathLineEdit.Dirs
    self.saveDirectoryLineEdit.options = ctk.ctkPathLineEdit.DontUseSheet
    self.saveDirectoryLineEdit.options = ctk.ctkPathLineEdit.ShowDirsOnly
    self.saveDirectoryLineEdit.showHistoryButton = True
    self.saveDirectoryLineEdit.setMinimumWidth(100)
    self.saveDirectoryLineEdit.setMaximumWidth(500)
    parametersFormLayout.addRow("Save directory", self.saveDirectoryLineEdit)

    self.filePrefixEdit = qt.QLineEdit()
    parametersFormLayout.addRow("File name prefix", self.filePrefixEdit)
    
    # Group with fiducials
    
    fiducialGroupBox = qt.QGroupBox(parametersCollapsibleButton)
    fiducialGroupLayout = qt.QFormLayout(fiducialGroupBox)
    fiducialGroupBox.setTitle("Fiducial annotations")
    fiducialGroupBox.setLayout(fiducialGroupLayout)

    self.fiducialsSelector = slicer.qMRMLNodeComboBox()
    self.fiducialsSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.fiducialsSelector.selectNodeUponCreation = True
    self.fiducialsSelector.addEnabled = True
    self.fiducialsSelector.removeEnabled = True
    self.fiducialsSelector.noneEnabled = False
    self.fiducialsSelector.showHidden = False
    self.fiducialsSelector.showChildNodeTypes = False
    self.fiducialsSelector.setMRMLScene(slicer.mrmlScene)
    self.fiducialsSelector.setToolTip("Pick the fiducial list that marks objects to detect")
    fiducialGroupLayout.addRow("Input fiducials: ", self.fiducialsSelector)

    self.proximityThresholdSliderWidget = ctk.ctkSliderWidget()
    self.proximityThresholdSliderWidget.singleStep = 0.1
    self.proximityThresholdSliderWidget.minimum = 0
    self.proximityThresholdSliderWidget.maximum = 100
    self.proximityThresholdSliderWidget.value = 2
    self.proximityThresholdSliderWidget.setToolTip("Set distance threshold for landmark detection")
    fiducialGroupLayout.addRow("Fiducial proximity threshold (mm)", self.proximityThresholdSliderWidget)

    self.exportFiducialsButton = qt.QPushButton("Export with fiducial annotations")
    self.exportFiducialsButton.toolTip = "Export all images from sequence to files"
    self.exportFiducialsButton.enabled = False
    fiducialGroupLayout.addRow(self.exportFiducialsButton)
    
    # Group with model
    
    modelGroupBox = qt.QGroupBox(parametersCollapsibleButton)
    modelGroupLayout = qt.QFormLayout(modelGroupBox)
    modelGroupBox.setTitle("Model annotations")
    modelGroupBox.setLayout(modelGroupLayout)

    self.modelSelector = slicer.qMRMLNodeComboBox()
    self.modelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.modelSelector.selectNodeUponCreation = True
    self.modelSelector.addEnabled = False
    self.modelSelector.removeEnabled = True
    self.modelSelector.noneEnabled = False
    self.modelSelector.showHidden = False
    self.modelSelector.showChildNodeTypes = False
    self.modelSelector.setMRMLScene(slicer.mrmlScene)
    self.modelSelector.setToolTip("Pick the model that marks objects to detect")
    modelGroupLayout.addRow("Input model: ", self.modelSelector)

    self.exportModelButton = qt.QPushButton("Export with model annotations")
    modelGroupLayout.addRow(self.exportModelButton)
    
    # Adding groups to main layout

    parametersFormLayout.addRow(fiducialGroupBox)
    parametersFormLayout.addRow(modelGroupBox)
    
    # connections
    self.exportFiducialsButton.connect('clicked(bool)', self.onExportButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.fiducialsSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    
    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()


  def cleanup(self):
    pass

  def onSelect(self):
    self.exportFiducialsButton.enabled = self.inputSelector.currentNode() and self.fiducialsSelector.currentNode()

  def onExportButton(self):
    proximityThreshold = self.proximityThresholdSliderWidget.value

    exportPath = self.saveDirectoryLineEdit.currentPath
    fileNamePrefix = self.filePrefixEdit.text
    self.exportFiducialsButton.setEnabled(False)
    self.logic.exportData(self.inputSelector.currentNode(), self.fiducialsSelector.currentNode(),
                          proximityThreshold, exportPath, fileNamePrefix)
    self.exportFiducialsButton.setEnabled(True)


#
# UsAnnotationExportLogic
#

class UsAnnotationExportLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

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


  def transformFiducialArrayByMatrix(self, fiducialArray, transformMatrix):
    """
    Create copy of array of points transformed by the specified transformation node
    :param fiducialArray: numpy.array((n,3))
    :param transformMatrix: vtk.vtkMatrix4x4
    :return: numpy.array((n,3))
    """

    if not isinstance(fiducialArray, numpy.ndarray):
      logging.error('fiducialArray is not a numpy.ndarray')
      return False

    if not isinstance(transformMatrix, vtk.vtkMatrix4x4):
      logging.error('transformNode is not a slicer.vtkMRMLLinearTransformNode')
      return False

    numRoadmapReference = fiducialArray.shape[0]
    fiducialArray_transformed = numpy.zeros((numRoadmapReference, 3))
    for i in range(numRoadmapReference):
      point_src = numpy.append(fiducialArray[i], 1)
      point_dst = numpy.array(transformMatrix.MultiplyFloatPoint(point_src))
      fiducialArray_transformed[i][:] = point_dst[:3]
    return fiducialArray_transformed


  def findClosestPoint(self, imageExtent, fiducials_Image):
    """
    Compute the closest point to and image (rectangle).
    :param imageExtent: array(4) = [minX, maxX, minY, maxY]
    :param fiducials_Image: numpy.array((n,3))
    :return: (closestIndex, closestDistanceMm)
    """
    closestIndex = 0
    closestDistanceMm = float_info.max
    numFiducials = fiducials_Image.shape[0]
    for i in range(numFiducials):
      # print fiducialLabels[i] + " : " + str(fiducials_Image[i])
      x = fiducials_Image[i][0]
      y = fiducials_Image[i][1]
      z = fiducials_Image[i][2]
      # Only work with fiducials within the width and height of the image
      if x < imageExtent[0] or x > imageExtent[1] or y < imageExtent[2] or y > imageExtent[3]:
        logging.debug("Skipping fiducial because it is out of X or Y range")
        continue
      if abs(z) < closestDistanceMm:
        closestDistanceMm = abs(z)
        closestIndex = i

    return (closestIndex, closestDistanceMm)


  def setupExportPaths(self, exportPath):
    self.exportPath = exportPath
    
    # Create folders if they don't exist already
    
    self.imagesPath = os.path.join(self.exportPath, 'images')
    self.annotationsPath = os.path.join(self.exportPath, 'annotations')

    if not os.path.exists(self.imagesPath):
      os.makedirs(self.imagesPath)
      logging.info("Creating folder: " + self.imagesPath)

    if not os.path.exists(self.annotationsPath):
      os.makedirs(self.annotationsPath)
      logging.info("Creating folder: " + self.annotationsPath)


  def exportData(self, inputBrowserNode, inputFiducials, proximityThresholdMm, exportPath, fileNamePrefix=""):
    logging.info('Processing started')

    if len(exportPath) < 3:
      logging.warning('Export path not specified')
      return

    self.setupExportPaths(exportPath)

    # Convert fiducials to array

    numFiducials = inputFiducials.GetNumberOfFiducials()
    fiducialLabels = [''] * numFiducials
    self.fiducialCoordinates_Reference = numpy.zeros([numFiducials, 3])
    for i in range(numFiducials):
      inputFiducials.GetNthFiducialPosition(i, self.fiducialCoordinates_Reference[i])
      fiducialLabels[i] = inputFiducials.GetNthFiducialLabel(i)

    # Assume that the selected browser node has a master image node, and it has a transform node

    self.browserNode = inputBrowserNode
    masterNode = self.browserNode.GetMasterSequenceNode()
    imageNode = self.browserNode.GetProxyNode(masterNode)

    imageTransformId = imageNode.GetTransformNodeID()
    imageToReferenceNode = slicer.util.getNode(imageTransformId)

    imageData = imageNode.GetImageData()
    imageExtent = imageData.GetExtent()
    self.browserNode.SelectFirstItem()
    numItems = self.browserNode.GetNumberOfItems()
    
    if len(fileNamePrefix) < 1:
      fileNamePrefix = self.browserNode.GetName()
    csvFileName = fileNamePrefix + ".csv"
    csvFilePath = os.path.join(self.annotationsPath, csvFileName)
    csvFile = open(csvFilePath, 'wb')
    csvWriter = csv.writer(csvFile, delimiter=',')
    csvWriter.writerow(['filename', 'width', 'height', 'class', 'x', 'y'])
    
    pngWriter = vtk.vtkPNGWriter()
    
    for i in range(numItems - 1):
      self.browserNode.SelectNextItem()
      imageData = imageNode.GetImageData()
      pngWriter.SetInputData(imageData)
      pngFileName = fileNamePrefix + "_%04d" % i + ".png"
      pngFilePathName = os.path.join(self.imagesPath, pngFileName)
      logging.info("Writing: " + pngFilePathName)
      pngWriter.SetFileName(pngFilePathName)
      pngWriter.Update()
      pngWriter.Write()
      
      # Compute ReferenceToImage transform of current time

      imageToReferenceMatrix = vtk.vtkMatrix4x4()
      imageToReferenceNode.GetMatrixTransformToWorld(imageToReferenceMatrix)
      referenceToImageMatrix = vtk.vtkMatrix4x4()
      referenceToImageMatrix.DeepCopy(imageToReferenceMatrix)
      referenceToImageMatrix.Invert()
      
      # Get PixelsToMm scaling
      
      imageToReferenceTransform = vtk.vtkTransform()
      imageToReferenceTransform.SetMatrix(imageToReferenceMatrix)
      pixelToMm = imageToReferenceTransform.GetScale()[0]

      # Create fiducial list in the Image coordinate system

      fiducials_Image = self.transformFiducialArrayByMatrix(self.fiducialCoordinates_Reference, referenceToImageMatrix)

      (closestIndex, closestDistancePixels) = self.findClosestPoint(imageExtent, fiducials_Image)
      
      p = fiducials_Image[closestIndex]
      p[1] = imageExtent[1] - p[1]  # Second dimension is reversed between IJK and PNG coordinates.
      
      x = int(round(p[0]))
      y = int(round(p[1]))

      if (closestDistancePixels * pixelToMm) < proximityThresholdMm:
        logging.info("Saving landmark for image " + pngFileName)
        csvWriter.writerow([pngFileName, str(imageExtent[1] + 1), str(imageExtent[3] + 1),
                            fiducialLabels[closestIndex], x, y])

      slicer.app.processEvents()

    csvFile.close()

    logging.info('Processing completed')
    return True


class UsAnnotationExportTest(ScriptedLoadableModuleTest):
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
    self.test_UsAnnotationExport1()

  def test_UsAnnotationExport1(self):
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
    logic = UsAnnotationExportLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
