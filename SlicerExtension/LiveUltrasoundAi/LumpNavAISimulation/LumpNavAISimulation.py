import logging
import os
import numpy as np
import numpy.typing as npt
import pandas as pd
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
    needleTipToNeedle: slicer.vtkMRMLLinearTransformNode
    cauteryToReference: slicer.vtkMRMLLinearTransformNode
    cauteryTipToCautery: slicer.vtkMRMLLinearTransformNode
    breachWarning: slicer.vtkMRMLBreachWarningNode
    currentResultsTable: slicer.vtkMRMLTableNode

    # Other parameters
    timestampBuffer: int = 5
    threshold: float = 127.0
    smooth: Annotated[float, WithinRange(0, 50)] = 15
    decimate: Annotated[float, WithinRange(0, 1.0)] = 0.25
    closeMarginThreshold: float = 1.0
    cleanThreshold: float = 30.0


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

        slicer.mymodW = self  # for debugging

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
        self.logic.updateProgressBarCallback = self.updateProgressBar

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Update output directory path from settings
        lastOutputFolder = slicer.util.settingsValue(self.logic.LAST_OUTPUT_FOLDER_SETTING, None)
        if lastOutputFolder:
            self.ui.outputDirectoryButton.directory = lastOutputFolder

        # Make some collapsible buttons exclusive
        self.ui.inputsCollapsibleButton.connect("contentsCollapsed(bool)", self.onInputsCollapsed)
        self.ui.resultsCollapsibleButton.connect("contentsCollapsed(bool)", self.onResultsCollapsed)

        # Buttons
        self.ui.createSurfaceButton.connect("clicked()", self.onCreateSurface)
        self.ui.restoreDefaultsButton.connect("clicked()", self.onRestoreDefaults)
        self.ui.trimRangeWidget.connect("minimumValueIsChanging(double)", self.onTrimRangeChanging)
        self.ui.trimRangeWidget.connect("maximumValueIsChanging(double)", self.onTrimRangeChanging)
        self.ui.setStartButton.connect("clicked()", self.onSetStart)
        self.ui.setStopButton.connect("clicked()", self.onSetStop)
        self.ui.setTransformsButton.connect("clicked()", self.onFreeze)
        self.ui.runButton.connect("clicked()", self.onRun)
        self.ui.plotButton.connect("clicked()", self.onPlotFromSequence)
        self.ui.resultsTableView.connect("selectionChanged()", self._checkCanPlotFromSelection)
        self.ui.plotSelectionButton.connect("clicked()", self.onPlotFromSelection)
        self.ui.outputDirectoryButton.connect("directoryChanged(const QString)", self.onFolderChanged)
        self.ui.exportButton.connect("clicked()", self.onExport)
        self.ui.stopButton.connect("clicked()", self.onStop)

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
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanFreeze)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanRun)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPlotFromSequence)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._populateTable)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPlotFromSelection)

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
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanFreeze)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanRun)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPlotFromSequence)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._populateTable)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPlotFromSelection)
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

            # Observer for setting reference transforms
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanFreeze)
            self._checkCanFreeze()

            # Observer for run button
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanRun)
            self._checkCanRun()

            # Observer for plot button
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPlotFromSequence)
            self._checkCanPlotFromSequence()

            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._populateTable)
            self._populateTable()

            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPlotFromSelection)
            self._checkCanPlotFromSelection()

    def onInputsCollapsed(self, collapsed) -> None:
        if not collapsed:
            self.ui.resultsCollapsibleButton.collapsed = True
    
    def onResultsCollapsed(self, collapsed) -> None:
        if not collapsed:
            self.ui.inputsCollapsibleButton.collapsed = True
    
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
            self.ui.cleanCheckBox.checked = self.logic.DEFAULT_CLEAN
            self.ui.cleanThresholdSpinBox.value = self.logic.DEFAULT_CLEAN_THRESHOLD
    
    def _setMinMaxTrim(self, caller=None, event=None) -> None:
        if (self._parameterNode 
            and self._parameterNode.trackingSeqBr 
            and self.lastSeqBr != self._parameterNode.trackingSeqBr):
            self.ui.trimSeqCollapsibleButton.collapsed = False
            self.lastSeqBr = self._parameterNode.trackingSeqBr
            sequenceNode = self._parameterNode.trackingSeqBr.GetMasterSequenceNode()
            maxItems = self._parameterNode.trackingSeqBr.GetNumberOfItems()
            maxTime = float(sequenceNode.GetNthIndexValue(maxItems - 1))
            minTime = float(sequenceNode.GetNthIndexValue(0))
            self.ui.trimRangeWidget.maximum = maxTime
            self.ui.trimRangeWidget.maximumValue = maxTime
            self.ui.trimRangeWidget.minimum = minTime
            self.ui.trimRangeWidget.minimumValue = minTime
    
    def onTrimRangeChanging(self, value) -> None:
        if self._parameterNode.trackingSeqBr:
            sequenceNode = self._parameterNode.trackingSeqBr.GetMasterSequenceNode()
            item = sequenceNode.GetItemNumberFromIndexValue(str(value), False)
            self._parameterNode.trackingSeqBr.SetSelectedItemNumber(item)
    
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

    def _checkCanFreeze(self, caller=None, event=None) -> None:
        if (self._parameterNode 
            and self._parameterNode.tumorModel 
            and self._parameterNode.trackingSeqBr 
            and self._parameterNode.needleToReference 
            and self._parameterNode.needleTipToNeedle 
            and self._parameterNode.cauteryToReference 
            and self._parameterNode.cauteryTipToCautery):
            self.ui.setTransformsButton.toolTip = _("Freeze reference transforms to current time")
            self.ui.setTransformsButton.enabled = True
        else:
            self.ui.setTransformsButton.toolTip = _("Select needle and cautery transforms")
            self.ui.setTransformsButton.enabled = False

    def onFreeze(self) -> None:
        self.logic.setReferenceTransformsToCurrentFrame()

    def _checkCanRun(self, caller=None, event=None) -> None:
        if (self._parameterNode
            and self._parameterNode.tumorModel
            and self._parameterNode.trackingSeqBr
            and self._parameterNode.cauteryTipToCautery):
            self.ui.runButton.toolTip = _("Replay cautery tracking relative to tumor model and record breaches")
            self.ui.runButton.enabled = True
        else:
            self.ui.runButton.toolTip = _("Select tumor model, tracking sequence browser, "
                                          "NeedleToReference, and cautery tip transforms")
            self.ui.runButton.enabled = False

    def onRun(self) -> None:
        blockRun = self.ui.runButton.blockSignals(True)
        blockPlot = self.ui.plotButton.blockSignals(True)
        blockPlotSelection = self.ui.plotSelectionButton.blockSignals(True)

        start = self.ui.trimRangeWidget.minimumValue
        stop = self.ui.trimRangeWidget.maximumValue
        runStatus = self.logic.runTrajectoryAnalysis(start, stop)
        # Display results if successful
        if runStatus == 0:
            self.ui.inputsCollapsibleButton.collapsed = True
            self.ui.resultsCollapsibleButton.collapsed = False
        
        self.ui.runButton.blockSignals(blockRun)
        self.ui.plotButton.blockSignals(blockPlot)
        self.ui.plotSelectionButton.blockSignals(blockPlotSelection)

    def _checkCanPlotFromSequence(self, caller=None, event=None) -> None:
        if (self._parameterNode 
            and self._parameterNode.tumorModel
            and self._parameterNode.trackingSeqBr 
            and self._parameterNode.needleTipToNeedle
            and self._parameterNode.cauteryTipToCautery):
            self.ui.plotButton.toolTip = _("Plot cautery trajectory")
            self.ui.plotButton.enabled = True
        else:
            self.ui.plotButton.toolTip = _("Select tumor model, tracking sequence browser, "
                                           "needle and cautery tip transforms")
            self.ui.plotButton.enabled = False
    
    def onPlotFromSequence(self) -> None:
        blockRun = self.ui.runButton.blockSignals(True)
        blockPlot = self.ui.plotButton.blockSignals(True)
        blockPlotSelection = self.ui.plotSelectionButton.blockSignals(True)

        start = self.ui.trimRangeWidget.minimumValue
        stop = self.ui.trimRangeWidget.maximumValue
        clean = self.ui.cleanCheckBox.checked
        status = self.logic.plotCauteryTrajectory(start, stop, clean)
        # Expand subject hierarchy if successful
        if status == 0:
            self.ui.trajectoryCollapsibleButton.collapsed = False
        
        self.ui.runButton.blockSignals(blockRun)
        self.ui.plotButton.blockSignals(blockPlot)
        self.ui.plotSelectionButton.blockSignals(blockPlotSelection)

    def _checkCanPlotFromSelection(self, caller=None, event=None) -> None:
        if (self._parameterNode 
            and self._parameterNode.tumorModel 
            and self._parameterNode.trackingSeqBr 
            and self._parameterNode.needleTipToNeedle 
            and self._parameterNode.cauteryTipToCautery 
            and self._parameterNode.currentResultsTable 
            and self.ui.resultsTableView.selectionModel().hasSelection):
            self.ui.plotSelectionButton.toolTip = _("Plot cautery trajectory from selected row")
            self.ui.plotSelectionButton.enabled = True
        else:
            self.ui.plotSelectionButton.toolTip = _("Select required nodes and timestamp to plot")
            self.ui.plotSelectionButton.enabled = False

    def onPlotFromSelection(self) -> None:
        blockRun = self.ui.runButton.blockSignals(True)
        blockPlot = self.ui.plotButton.blockSignals(True)
        blockPlotSelection = self.ui.plotSelectionButton.blockSignals(True)

        timestamp = float(self.ui.resultsTableView.selectionModel().selectedRows()[0].data())
        start = timestamp - self._parameterNode.timestampBuffer
        stop = timestamp + self._parameterNode.timestampBuffer
        clean = self.ui.cleanCheckBox.checked
        status = self.logic.plotCauteryTrajectory(start, stop, clean)
        # Expand subject hierarchy if successful
        if status == 0:
            self.ui.trajectoryCollapsibleButton.collapsed = False
        
        self.ui.runButton.blockSignals(blockRun)
        self.ui.plotButton.blockSignals(blockPlot)
        self.ui.plotSelectionButton.blockSignals(blockPlotSelection)
    
    def updateProgressBar(self, value) -> None:
        self.ui.progressBar.setValue(value)
    
    def _populateTable(self, caller=None, event=None) -> None:
        if (self._parameterNode 
            and self._parameterNode.currentResultsTable
            and self.lastResultsTable != self._parameterNode.currentResultsTable):
            self.lastResultsTable = self._parameterNode.currentResultsTable
            self.ui.resultsTableComboBox.setCurrentNode(self._parameterNode.currentResultsTable)
            self.ui.resultsTableView.setMRMLTableNode(self._parameterNode.currentResultsTable)
            self.ui.resultsTableView.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
            self.ui.filenameEdit.text = self._parameterNode.currentResultsTable.GetName()

    def onFolderChanged(self, outputFolder) -> None:
        settings = qt.QSettings()
        if not outputFolder:
            settings.setValue(self.logic.LAST_OUTPUT_FOLDER_SETTING, "")
        else:
            settings.setValue(self.logic.LAST_OUTPUT_FOLDER_SETTING, outputFolder)

    def onExport(self) -> None:
        filename = self.ui.filenameEdit.text
        resultsTable = self._parameterNode.currentResultsTable
        if resultsTable:
            try:
                outputDirectory = self.ui.outputDirectoryButton.directory
                fullpath = os.path.join(outputDirectory, filename + self.logic.RESULTS_CSV_SUFFIX)
                slicer.util.saveNode(resultsTable, fullpath)
                logging.info("Results table exported to " + fullpath)
                slicer.util.infoDisplay("Results table saved to " + fullpath)
            except Exception as e:
                slicer.util.errorDisplay("Failed to export results table: " + str(e))
    
    def onStop(self) -> None:
        if self.logic.processing:
            self.logic.stopProcess = True


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

    DEFAULT_THRESHOLD = 127.0
    DEFAULT_SMOOTH = 15
    DEFAULT_DECIMATE = 0.25
    DEFAULT_CLOSE_MARGIN = 1.0
    DEFAULT_CLEAN = True
    DEFAULT_CLEAN_THRESHOLD = 30.0

    RESULTS_TABLE_SUFFIX = "results"
    LAST_OUTPUT_FOLDER_SETTING =  "LumpNavAISimulation/LastOutputFolder"
    RESULTS_CSV_SUFFIX = ".csv"
    TIME_COLUMN = 0
    MARGIN_STATUS_COLUMN = 1
    DISTANCE_COLUMN = 2
    LOCATION_COLUMN = 3

    TRAJECTORY_MARKUPS_SUFFIX = "CauteryTipMarkups"
    TRAJECTORY_MODEL_SUFFIX = "CauteryTipModel"

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.updateProgressBarCallback = None
        self.processing = False
        self.stopProcess = False

        slicer.mymodL = self  # for debugging

    def getParameterNode(self):
        return LumpNavAISimulationParameterNode(super().getParameterNode())

    def updateProgress(self, value) -> None:
        if self.updateProgressBarCallback:
            self.updateProgressBarCallback(value)
    
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

        # Extract largest portion
        connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
        connectivityFilter.SetInputData(tumorModel.GetPolyData())
        connectivityFilter.SetExtractionModeToLargestRegion()

        # Clean up model
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(connectivityFilter.GetOutputPort())

        # Convert to convex hull
        convexHull = vtk.vtkDelaunay3D()
        convexHull.SetInputConnection(cleanFilter.GetOutputPort())
        outerSurface = vtk.vtkGeometryFilter()
        outerSurface.SetInputConnection(convexHull.GetOutputPort())
        outerSurface.Update()
        tumorModel.SetAndObservePolyData(outerSurface.GetOutput())

    def setReferenceTransformsToCurrentFrame(self) -> None:
        parameterNode = self.getParameterNode()
        parentTransformNode = parameterNode.needleToReference.GetParentTransformNode()

        # copy current NeedleToReference transform to new transform node
        currentTime = self.getCurrentTime()
        needleToRas = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"NeedleToRas_{currentTime}")
        # refNeedleToRef.SetAndObserveTransformNodeID(parentTransformNode.GetID())
        matrix = vtk.vtkMatrix4x4()
        parameterNode.needleToReference.GetMatrixTransformToWorld(matrix)
        needleToRas.SetMatrixTransformToParent(matrix)
        cauteryToNeedle = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "CauteryToNeedle")
        cauteryToNeedle.SetAndObserveTransformNodeID(needleToRas.GetID())

        # move tumor, needle, and cautery to new tranforms
        parameterNode.tumorModel.SetAndObserveTransformNodeID(needleToRas.GetID())
        parameterNode.needleTipToNeedle.SetAndObserveTransformNodeID(needleToRas.GetID())
        parameterNode.cauteryTipToCautery.SetAndObserveTransformNodeID(cauteryToNeedle.GetID())

        # update CauteryToNeedle transform using transform processor
        # transformProcessorLogic = slicer.modules.transformprocessor.logic()
        transformProcessorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformProcessorNode", "TransformProcessorNode")
        transformProcessorNode.SetAndObserveInputFromTransformNode(parameterNode.cauteryToReference)
        transformProcessorNode.SetAndObserveInputToTransformNode(parameterNode.needleToReference)
        transformProcessorNode.SetAndObserveOutputTransformNode(cauteryToNeedle)
        transformProcessorNode.SetProcessingMode(transformProcessorNode.PROCESSING_MODE_COMPUTE_FULL_TRANSFORM)
        transformProcessorNode.SetUpdateModeToAuto()

        # update parameter node
        parameterNode.needleToReference = needleToRas
    
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
        
    def getTumorCenterRas(self) -> Annotated[npt.NDArray, Literal[3]]:
        parameterNode = self.getParameterNode()
        tumorModel = parameterNode.tumorModel
        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(tumorModel.GetPolyData())
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()
        center = centerOfMassFilter.GetCenter()

        needleToReference = parameterNode.needleToReference
        if needleToReference:
            # Transform center to RAS coordinates
            needleToRasMatrix = vtk.vtkMatrix4x4()
            needleToReference.GetMatrixTransformToWorld(needleToRasMatrix)
            center = needleToRasMatrix.MultiplyFloatPoint([center[0], center[1], center[2], 1])
            center = np.array(center[:3])

        return center
    
    def getNeedleTipToWorld(self) -> Annotated[npt.NDArray, Literal[3]]:
        parameterNode = self.getParameterNode()
        needleTipToNeedle = parameterNode.needleTipToNeedle
        needleTipWorldMatrix = vtk.vtkMatrix4x4()
        needleTipToNeedle.GetMatrixTransformToWorld(needleTipWorldMatrix)
        needleTipToWorldArr = slicer.util.arrayFromVTKMatrix(needleTipWorldMatrix)
        return needleTipToWorldArr

    def getRelativeCoordinates(self) -> Annotated[npt.NDArray, Literal[3]]:
        parameterNode = self.getParameterNode()

        # Get cautery tip to RAS transform
        cauteryTipToRasMatrix = vtk.vtkMatrix4x4()
        parameterNode.cauteryTipToCautery.GetMatrixTransformToWorld(cauteryTipToRasMatrix)
        cauteryTipRas = cauteryTipToRasMatrix.MultiplyFloatPoint([0, 0, 0, 1])
        cauteryTipRas = np.array(cauteryTipRas)

        # Get tumor center
        center = self.getTumorCenterRas()

        # Subtract to get relative position
        cauteryTipToTumorCenter = cauteryTipRas[:3] - center[:3]
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
    
    def runTrajectoryAnalysis(self, start, stop) -> int:
        logging.info("Trajectory analysis started")
        self.processing = True

        parameterNode = self.getParameterNode()
        needleToReference = parameterNode.tumorModel.GetParentTransformNode()
        cauteryTipToCautery = parameterNode.cauteryTipToCautery
        sequenceNode = parameterNode.trackingSeqBr.GetMasterSequenceNode()
        if not parameterNode.breachWarning:
            parameterNode.breachWarning = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLBreachWarningNode")

        # Set breach warning node parameters
        parameterNode.breachWarning.SetOriginalColor(*parameterNode.tumorModel.GetDisplayNode().GetColor())
        parameterNode.breachWarning.SetAndObserveWatchedModelNodeID(parameterNode.tumorModel.GetID())
        parameterNode.breachWarning.SetAndObserveToolTransformNodeId(parameterNode.cauteryTipToCautery.GetID())

        startItem = sequenceNode.GetItemNumberFromIndexValue(str(start), False)
        stopItem = sequenceNode.GetItemNumberFromIndexValue(str(stop), False)
        numItems = stopItem - startItem

        # Create results table
        modelName = parameterNode.tumorModel.GetName()
        resultsTableName = f"{modelName}_{str(int(start))}-{str(int(stop))}_{self.RESULTS_TABLE_SUFFIX}"
        resultsTable = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", resultsTableName)

        # create results dataframe
        resultsDf = pd.DataFrame(columns=[
            "Time (s)", "Distance To Tumour (mm)", "CauteryTipNeedle", "NeedleTipToRas", "TumorCenterRas", "Location"]
        )

        selectedItemNumber = parameterNode.trackingSeqBr.GetSelectedItemNumber()  # for restoring later
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            for item in range(startItem, stopItem + 1):
                self.updateProgress((item - startItem) / numItems * 100)
                slicer.app.processEvents()
                if self.stopProcess:
                    raise RuntimeError("Trajectory analysis stopped by user")

                parameterNode.trackingSeqBr.SetSelectedItemNumber(item)

                # Check distance of cautery to tumor and record in table
                distanceToTumor = parameterNode.breachWarning.GetClosestDistanceToModelFromToolTip()
                cauteryTipNeedle = vtk.vtkMatrix4x4()
                cauteryTipToCautery.GetMatrixTransformToNode(needleToReference, cauteryTipNeedle)
                cauteryTipNeedle = cauteryTipNeedle.MultiplyFloatPoint([0, 0, 0, 1])
                resultsDf = pd.concat([resultsDf, pd.DataFrame([{
                    "Time (s)": sequenceNode.GetNthIndexValue(item),
                    "Distance To Tumour (mm)": distanceToTumor, 
                    "CauteryTipNeedle": cauteryTipNeedle[:3],
                    "NeedleTipToRas": self.getNeedleTipToWorld(),
                    "TumorCenterRas": self.getTumorCenterRas(),
                    "Location": self.getAnatomicalPositionFromRelativeCoords(self.getRelativeCoordinates())
                }])], ignore_index=True)

            # convert to table
            for col_name in resultsDf.columns:
                array = vtk.vtkVariantArray()
                array.SetName(str(col_name))
                for value in resultsDf[col_name]:
                    array.InsertNextValue(vtk.vtkVariant(str(value)))
                resultsTable.AddColumn(array)
            parameterNode.currentResultsTable = resultsTable

            logging.info("Trajectory analysis completed")
            exitCode = 0

        except Exception as e:
            logging.error(str(e))
            self.updateProgressBarCallback(0)
            self.stopProcess = False
            exitCode = 1

        finally:
            qt.QApplication.restoreOverrideCursor()
            slicer.app.processEvents()
            parameterNode.trackingSeqBr.SetSelectedItemNumber(selectedItemNumber)
            self.processing = False
            return exitCode
        
    def plotCauteryTrajectory(self, start, stop, clean) -> int:
        logging.info("Plotting cautery trajectory")
        self.processing = True

        parameterNode = self.getParameterNode()
        sequenceNode = parameterNode.trackingSeqBr.GetMasterSequenceNode()

        # Breach warning node is needed if clean is enabled
        if clean and not parameterNode.breachWarning:
            parameterNode.breachWarning = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLBreachWarningNode")
            parameterNode.breachWarning.SetOriginalColor(*parameterNode.tumorModel.GetDisplayNode().GetColor())
            parameterNode.breachWarning.SetAndObserveWatchedModelNodeID(parameterNode.tumorModel.GetID())
            parameterNode.breachWarning.SetAndObserveToolTransformNodeId(parameterNode.cauteryTipToCautery.GetID())

        # Create new markups node
        markupsNodeName = f"{str(int(start))}-{str(int(stop))}_{self.TRAJECTORY_MARKUPS_SUFFIX}"
        markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", markupsNodeName)
        markupsNode.SetAndObserveTransformNodeID(parameterNode.needleTipToNeedle.GetID())
        markupsNode.CreateDefaultDisplayNodes()
        markupsNode.GetMarkupsDisplayNode().SetTextScale(0)
        markupsNode.SetDisplayVisibility(False)

        startItem = sequenceNode.GetItemNumberFromIndexValue(str(start), False)
        stopItem = sequenceNode.GetItemNumberFromIndexValue(str(stop), False)
        numItems = stopItem - startItem

        selectedItemNumber = parameterNode.trackingSeqBr.GetSelectedItemNumber()  # for restoring later
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            for item in range(startItem, stopItem + 1):
                self.updateProgress((item - startItem) / numItems * 100)
                slicer.app.processEvents()
                if self.stopProcess:
                    raise RuntimeError("Trajectory analysis stopped by user")
                
                parameterNode.trackingSeqBr.SetSelectedItemNumber(item)

                # Check distance to tumor if needed
                if clean:
                    distanceToTumor = parameterNode.breachWarning.GetClosestDistanceToModelFromToolTip()
                    if distanceToTumor > parameterNode.cleanThreshold:
                        continue  # skip this point

                # Get cautery tip to RAS transform
                cauteryTipToRasMatrix = vtk.vtkMatrix4x4()
                parameterNode.cauteryTipToCautery.GetMatrixTransformToWorld(cauteryTipToRasMatrix)

                # Get needle tip to RAS transform
                needleTipToRasMatrix = vtk.vtkMatrix4x4()
                parameterNode.needleTipToNeedle.GetMatrixTransformToWorld(needleTipToRasMatrix)

                # Get cautery tip to needle tip
                rasToNeedleTip = vtk.vtkMatrix4x4()
                vtk.vtkMatrix4x4.Invert(needleTipToRasMatrix, rasToNeedleTip)
                cauteryTipToNeedleTip = vtk.vtkMatrix4x4()
                vtk.vtkMatrix4x4.Multiply4x4(rasToNeedleTip, cauteryTipToRasMatrix, cauteryTipToNeedleTip)

                # Add markup of cautery tip in needle tip coordinate system
                cauteryTip_NeedleTip = cauteryTipToNeedleTip.MultiplyFloatPoint([0, 0, 0, 1])
                slicer.modules.markups.logic().AddControlPoint(
                    cauteryTip_NeedleTip[0], cauteryTip_NeedleTip[1], cauteryTip_NeedleTip[2]
                )
            
            # Create cylinder model
            modelName = f"{str(int(start))}-{str(int(stop))}_{self.TRAJECTORY_MODEL_SUFFIX}"
            modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", modelName)
            modelNode.SetAndObserveTransformNodeID(parameterNode.needleTipToNeedle.GetID())
            modelNode.CreateDefaultDisplayNodes()
            modelNode.GetDisplayNode().SetColor(0.5, 0.5, 0.5)  # gray
            modelNode.SetDisplayVisibility(True)
            createModelsLogic = slicer.modules.createmodels.logic()
            createModelsLogic.CreateCylinder(1.0, 1.0, modelNode)

            # Convert markups to curve model
            markupsToModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsToModelNode")
            markupsToModelNode.SetAutoUpdateOutput(False)
            markupsToModelNode.SetModelType(markupsToModelNode.Curve)
            markupsToModelNode.SetAndObserveInputNodeID(markupsNode.GetID())
            markupsToModelNode.SetAndObserveOutputModelNodeID(modelNode.GetID())
            markupsToModelLogic = slicer.modules.markupstomodel.logic()
            markupsToModelLogic.UpdateOutputModel(markupsToModelNode)

            logging.info("Cautery trajectory plotted")
            exitCode = 0
        
        except Exception as e:
            logging.error(str(e))
            self.updateProgressBarCallback(0)
            self.stopProcess = False
            exitCode = 1
        
        finally:
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
            qt.QApplication.restoreOverrideCursor()
            slicer.app.processEvents()
            parameterNode.trackingSeqBr.SetSelectedItemNumber(selectedItemNumber)
            self.processing = False
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
