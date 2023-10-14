import logging
import os
import numpy as np
import numpy.typing as npt
from typing import Annotated, Optional, Literal

import qt
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)


#
# LumpNavAISimulation
#

class LumpNavAISimulation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "LumpNavAISimulation"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [translate(self.__class__.__name__, "Ultrasound")]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Chris Yeung (Queen's Univ.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#LumpNavAISimulation">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # LumpNavAISimulation1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='LumpNavAISimulation',
        sampleName='LumpNavAISimulation1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'LumpNavAISimulation1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='LumpNavAISimulation1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='LumpNavAISimulation1'
    )

    # LumpNavAISimulation2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='LumpNavAISimulation',
        sampleName='LumpNavAISimulation2',
        thumbnailFileName=os.path.join(iconsPath, 'LumpNavAISimulation2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='LumpNavAISimulation2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='LumpNavAISimulation2'
    )


#
# LumpNavAISimulationParameterNode
#

@parameterNodeWrapper
class LumpNavAISimulationParameterNode:
    """
    The parameters needed by module.
    """
    # Nodes
    inputVolume: slicer.vtkMRMLScalarVolumeNode
    tumorModel: slicer.vtkMRMLModelNode
    trackingSeqBr: slicer.vtkMRMLSequenceBrowserNode
    needleToReference: slicer.vtkMRMLLinearTransformNode
    cauteryTipToCautery: slicer.vtkMRMLLinearTransformNode
    breachWarning: slicer.vtkMRMLBreachWarningNode
    currentResultsTable: slicer.vtkMRMLTableNode

    # Other parameters
    threshold: float = 127.0
    smooth: Annotated[float, WithinRange(0, 50)] = 15
    decimate: Annotated[float, WithinRange(0, 1.0)] = 0.25
    closeMarginThreshold: float = 1.0


#
# LumpNavAISimulationWidget
#

class LumpNavAISimulationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.lastSeqBr = None
        self.lastResultsTable = None

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/LumpNavAISimulation.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = LumpNavAISimulationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.createSurfaceButton.connect("clicked()", self.onCreateSurface)
        self.ui.restoreDefaultsButton.connect("clicked()", self.onRestoreDefaults)
        self.ui.setStartButton.connect("clicked()", self.onSetStart)
        self.ui.setStopButton.connect("clicked()", self.onSetStop)
        self.ui.runButton.connect("clicked()", self.onRun)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._setNeedleToReference)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanCreateSurface)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._setMinMaxTrim)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanRun)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._populateTable)

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[LumpNavAISimulationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._setNeedleToReference)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanCreateSurface)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._setMinMaxTrim)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanRun)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._populateTable)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

            # Observer for needle coordinate system transform
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._setNeedleToReference)
            self._setNeedleToReference()

            # Observer for create surface button
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanCreateSurface)
            self._checkCanCreateSurface()

            # Observer for trim slider
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._setMinMaxTrim)
            self._setMinMaxTrim()

            # Observer for run button
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanRun)
            self._checkCanRun()

            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._populateTable)
            self._populateTable()
    
    def _setNeedleToReference(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.needleToReference and self._parameterNode.tumorModel:
            self._parameterNode.tumorModel.SetAndObserveTransformNodeID(self._parameterNode.needleToReference.GetID())
    
    def _checkCanCreateSurface(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.tumorModel:
            self.ui.createSurfaceButton.toolTip = _("Create 3D surface model from input volume")
            self.ui.createSurfaceButton.enabled = True
        else:
            self.ui.createSurfaceButton.toolTip = _("Select input volume and output model nodes")
            self.ui.createSurfaceButton.enabled = False
    
    def onCreateSurface(self) -> None:
        blockSignals = self.ui.createSurfaceButton.blockSignals(True)
        self.logic.createSurfaceFromVolume()
        self.ui.createSurfaceButton.blockSignals(blockSignals)
    
    def onRestoreDefaults(self) -> None:
        if self._parameterNode:
            self.ui.thresholdSpinBox.value = self.logic.DEFAULT_THRESHOLD
            self.ui.smoothSliderWidget.value = self.logic.DEFAULT_SMOOTH
            self.ui.decimateSliderWidget.value = self.logic.DEFAULT_DECIMATE
            self.ui.closeMarginSpinBox.value = self.logic.DEFAULT_CLOSE_MARGIN
    
    def _setMinMaxTrim(self, caller=None, event=None) -> None:
        if (self._parameterNode 
            and self._parameterNode.trackingSeqBr 
            and self.lastSeqBr != self._parameterNode.trackingSeqBr):
            self.lastSeqBr = self._parameterNode.trackingSeqBr
            sequenceNode = self._parameterNode.trackingSeqBr.GetMasterSequenceNode()
            maxItems = self._parameterNode.trackingSeqBr.GetNumberOfItems()
            maxTime = float(sequenceNode.GetNthIndexValue(maxItems - 1))
            minTime = float(sequenceNode.GetNthIndexValue(0))
            self.ui.trimRangeWidget.minimum = minTime
            self.ui.trimRangeWidget.maximum = maxTime
            self.ui.trimRangeWidget.minimumValue = minTime
            self.ui.trimRangeWidget.maximumValue = maxTime
    
    def onSetStart(self) -> None:
        currentTime = self.logic.getCurrentTime()
        if currentTime > self.ui.trimRangeWidget.maximumValue:
            self.ui.trimRangeWidget.maximumValue = currentTime
        self.ui.trimRangeWidget.minimumValue = currentTime

    def onSetStop(self) -> None:
        currentTime = self.logic.getCurrentTime()
        if currentTime < self.ui.trimRangeWidget.minimumValue:
            self.ui.trimRangeWidget.minimumValue = currentTime
        self.ui.trimRangeWidget.maximumValue = currentTime

    def _checkCanRun(self, caller=None, event=None) -> None:
        if (self._parameterNode
            and self._parameterNode.tumorModel
            and self._parameterNode.trackingSeqBr
            and self._parameterNode.cauteryTipToCautery):
            self.ui.runButton.toolTip = _("Replay cautery tracking relative to tumor model and record breaches")
            self.ui.runButton.enabled = True
        else:
            self.ui.runButton.toolTip = _("Select tumor model, tracking sequence browser, "
                                          "and needle and cautery tip transforms")
            self.ui.runButton.enabled = False

    def onRun(self) -> None:
        start = self.ui.trimRangeWidget.minimumValue
        stop = self.ui.trimRangeWidget.maximumValue
        runStatus = self.logic.runSimulation(start, stop)
        # Display results if successful
        if runStatus == 0:
            self.ui.inputsCollapsibleButton.collapsed = True
            self.ui.resultsCollapsibleButton.collapsed = False
    
    def _populateTable(self, caller=None, event=None) -> None:
        if (self._parameterNode 
            and self._parameterNode.currentResultsTable
            and self.lastResultsTable != self._parameterNode.currentResultsTable):
            self.lastResultsTable = self._parameterNode.currentResultsTable
            self.ui.resultsTableComboBox.setCurrentNode(self._parameterNode.currentResultsTable)
            self.ui.resultsTableView.setMRMLTableNode(self._parameterNode.currentResultsTable)
            self.ui.resultsTableView.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)


#
# LumpNavAISimulationLogic
#

class LumpNavAISimulationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    DEFAULT_THRESHOLD = 100.0
    DEFAULT_SMOOTH = 15
    DEFAULT_DECIMATE = 0.25
    DEFAULT_CLOSE_MARGIN = 1.0

    RESULTS_TABLE_SUFFIX = "_results"
    TIME_COLUMN = 0
    MARGIN_STATUS_COLUMN = 1
    DISTANCE_COLUMN = 2
    LOCATION_COLUMN = 3

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return LumpNavAISimulationParameterNode(super().getParameterNode())
    
    def createSurfaceFromVolume(self) -> None:
        logging.info("Creating surface model from volume")

        # Get input/output nodes
        parameterNode = self.getParameterNode()
        inputVolume = parameterNode.inputVolume
        tumorModel = parameterNode.tumorModel

        # Move to needle coordinate system if it exists
        if parameterNode.needleToReference:
            tumorModel.SetAndObserveTransformNodeID(parameterNode.needleToReference.GetID())

        # Set up grayscale model maker CLI node
        parameters = {
            "InputVolume": inputVolume.GetID(),
            "OutputGeometry": tumorModel.GetID(),
            "Threshold": parameterNode.threshold,
            "Smooth": parameterNode.smooth,
            "Decimate": parameterNode.decimate,
            "SplitNormals": True,
            "PointNormals": True
        }
        modelMaker = slicer.modules.grayscalemodelmaker

        # Run the CLI
        cliNode = slicer.cli.runSync(modelMaker, None, parameters)

        # Process results
        if cliNode.GetStatus() & cliNode.ErrorsMask:
            # error
            errorText = cliNode.GetErrorText()
            slicer.mrmlScene.RemoveNode(cliNode)
            raise ValueError("CLI execution failed: " + errorText)
        # success
        slicer.mrmlScene.RemoveNode(cliNode)
        
        # Change color to green
        displayNode = tumorModel.GetDisplayNode()
        displayNode.SetColor(0, 1, 0)
        displayNode.SetOpacity(0.3)

        # Clean up model
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputData(tumorModel.GetPolyData())
        cleanFilter.Update()
        tumorModel.SetAndObservePolyData(cleanFilter.GetOutput())

        # Extract largest portion
        connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
        connectivityFilter.SetInputData(tumorModel.GetPolyData())
        connectivityFilter.SetExtractionModeToLargestRegion()
        connectivityFilter.Update()
        tumorModel.SetAndObservePolyData(connectivityFilter.GetOutput())

        # Convert to convex hull
        convexHull = vtk.vtkDelaunay3D()
        convexHull.SetInputData(tumorModel.GetPolyData())
        outerSurface = vtk.vtkGeometryFilter()
        outerSurface.SetInputConnection(convexHull.GetOutputPort())
        outerSurface.Update()
        tumorModel.SetAndObservePolyData(outerSurface.GetOutput())
    
    def getCurrentTime(self) -> float:
        """
        Get the current time of the tracking sequence browser.
        """
        parameterNode = self.getParameterNode()
        if parameterNode.trackingSeqBr:
            sequenceNode = parameterNode.trackingSeqBr.GetMasterSequenceNode()
            currentItem = parameterNode.trackingSeqBr.GetSelectedItemNumber()
            currentIndex = sequenceNode.GetNthIndexValue(currentItem)
            return float(currentIndex)
        else:
            return 0.0
    
    def createResultsTable(self, name) -> slicer.vtkMRMLTableNode:
        parameterNode = self.getParameterNode()
        parameterNode.currentResultsTable = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
        
        # Add columns to table
        parameterNode.currentResultsTable.AddColumn()
        parameterNode.currentResultsTable.RenameColumn(self.TIME_COLUMN, "Time")
        parameterNode.currentResultsTable.SetColumnProperty(self.TIME_COLUMN, "type", "float")
        parameterNode.currentResultsTable.SetColumnUnitLabel("Time", "s")

        parameterNode.currentResultsTable.AddColumn()
        parameterNode.currentResultsTable.RenameColumn(self.MARGIN_STATUS_COLUMN, "Margin Status")
        parameterNode.currentResultsTable.SetColumnProperty(self.MARGIN_STATUS_COLUMN, "type", "string")

        parameterNode.currentResultsTable.AddColumn()
        parameterNode.currentResultsTable.RenameColumn(self.DISTANCE_COLUMN, "Distance to Tumor")
        parameterNode.currentResultsTable.SetColumnProperty(self.DISTANCE_COLUMN, "type", "float")
        parameterNode.currentResultsTable.SetColumnUnitLabel("Distance to Tumor", "mm")

        parameterNode.currentResultsTable.AddColumn()
        parameterNode.currentResultsTable.RenameColumn(self.LOCATION_COLUMN, "Location")
        parameterNode.currentResultsTable.SetColumnProperty(self.LOCATION_COLUMN, "type", "string")

        # Set table node parameters
        parameterNode.currentResultsTable.SetUseColumnNameAsColumnHeader(True)
        parameterNode.currentResultsTable.SetLocked(True)

    def getRelativeCoordinates(self) -> Annotated[npt.NDArray, Literal[3]]:
        parameterNode = self.getParameterNode()

        # Get cautery tip to RAS transform
        cauteryTipToRasMatrix = vtk.vtkMatrix4x4()
        parameterNode.cauteryTipToCautery.GetMatrixTransformToWorld(cauteryTipToRasMatrix)
        cauteryTipToRas = cauteryTipToRasMatrix.MultiplyFloatPoint([0, 0, 0, 1])
        cauteryTipToRas = np.array(cauteryTipToRas)

        # Get tumor center
        tumorModel = parameterNode.tumorModel
        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(tumorModel.GetPolyData())
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()
        center = centerOfMassFilter.GetCenter()

        # Get tumor center to RAS transform
        needleToReference = tumorModel.GetParentTransformNode()
        needleToRasMatrix = vtk.vtkMatrix4x4()
        needleToReference.GetMatrixTransformToWorld(needleToRasMatrix)
        tumorCenterToRas = needleToRasMatrix.MultiplyFloatPoint([center[0], center[1], center[2], 1])
        tumorCenterToRas = np.array(tumorCenterToRas)

        # Subtract to get relative position
        cauteryTipToTumorCenter = cauteryTipToRas[:3] - tumorCenterToRas[:3]
        return cauteryTipToTumorCenter
    
    @staticmethod
    def getAnatomicalPositionFromRelativeCoords(relativeCoordinates) -> str:
        absMax = max(relativeCoordinates.min(), relativeCoordinates.max(), key=abs)
        if absMax > 0:
            position = np.argmax(relativeCoordinates)
            if position == 0:
                return "Right"
            elif position == 1:
                return "Anterior"
            else:
                return "Superior"
        else:
            position = np.argmin(relativeCoordinates)
            if position == 0:
                return "Left"
            elif position == 1:
                return "Posterior"
            else:
                return "Inferior"
    
    # TODO: add progress bar for simulation
    def runSimulation(self, start, stop) -> int:
        logging.info("Simulation started")

        parameterNode = self.getParameterNode()
        sequenceNode = parameterNode.trackingSeqBr.GetMasterSequenceNode()
        if not parameterNode.breachWarning:
            parameterNode.breachWarning = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLBreachWarningNode")

        # Set breach warning node parameters
        parameterNode.breachWarning.SetOriginalColor(*parameterNode.tumorModel.GetDisplayNode().GetColor())
        parameterNode.breachWarning.SetAndObserveWatchedModelNodeID(parameterNode.tumorModel.GetID())
        parameterNode.breachWarning.SetAndObserveToolTransformNodeId(parameterNode.cauteryTipToCautery.GetID())

        # Create results table
        if parameterNode.inputVolume:
            modelName = parameterNode.inputVolume.GetName()
        else:
            modelName = parameterNode.tumorModel.GetName()
        resultsTableName = modelName + self.RESULTS_TABLE_SUFFIX
        self.createResultsTable(resultsTableName)

        selectedItemNumber = parameterNode.trackingSeqBr.GetSelectedItemNumber()  # for restoring later
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            startItem = sequenceNode.GetItemNumberFromIndexValue(str(start), False)
            stopItem = sequenceNode.GetItemNumberFromIndexValue(str(stop), False)
            for item in range(startItem, stopItem + 1):
                parameterNode.trackingSeqBr.SetSelectedItemNumber(item)

                # TODO: only add row if cautery wasn't breached or close before
                # Check distance of cautery to tumor and record in table
                distanceToTumor = parameterNode.breachWarning.GetClosestDistanceToModelFromToolTip()
                if distanceToTumor < self.DEFAULT_CLOSE_MARGIN:
                    rowIdx = parameterNode.currentResultsTable.AddEmptyRow()
                    parameterNode.currentResultsTable.SetCellText(
                        rowIdx, self.TIME_COLUMN, sequenceNode.GetNthIndexValue(item)
                    )
                    parameterNode.currentResultsTable.SetCellText(rowIdx, self.DISTANCE_COLUMN, str(distanceToTumor))
                    location = self.getAnatomicalPositionFromRelativeCoords(self.getRelativeCoordinates())
                    parameterNode.currentResultsTable.SetCellText(rowIdx, self.LOCATION_COLUMN, location)
                    
                    if distanceToTumor <= 0:  # Tumor breach
                        parameterNode.currentResultsTable.SetCellText(rowIdx, self.MARGIN_STATUS_COLUMN, "Breach")
                    else:  # Close margin
                        parameterNode.currentResultsTable.SetCellText(rowIdx, self.MARGIN_STATUS_COLUMN, "Close margin")

            logging.info("Simulation completed")
            exitCode = 0

        except Exception as e:
            logging.error(str(e))
            exitCode = 1

        finally:
            qt.QApplication.restoreOverrideCursor()
            parameterNode.trackingSeqBr.SetSelectedItemNumber(selectedItemNumber)
            return exitCode


#
# LumpNavAISimulationTest
#

class LumpNavAISimulationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_LumpNavAISimulation1()

    def test_LumpNavAISimulation1(self):
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
        registerSampleData()
        inputVolume = SampleData.downloadSample('LumpNavAISimulation1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        # Test the module logic

        logic = LumpNavAISimulationLogic()

        self.delayDisplay('Test passed')
