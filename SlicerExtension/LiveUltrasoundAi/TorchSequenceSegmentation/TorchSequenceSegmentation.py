import logging
import traceback
import os
import glob
import json
import qt
import vtk
import numpy as np

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

try:
    import yaml
except:
    slicer.util.pip_install("pyyaml")
    import yaml

try:
    import cv2
except:
    slicer.util.pip_install("opencv-python")
    import cv2

try:
    from scipy.ndimage import map_coordinates
    from scipy.interpolate import griddata
    from scipy.spatial import Delaunay
except:
    slicer.util.pip_install('scipy')
    from scipy.ndimage import map_coordinates
    from scipy.interpolate import griddata
    from scipy.spatial import Delaunay

INSTALL_PYTORCHUTILS = False
try:
    import torch
    import torchvision
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except (ImportError, OSError):
    try:
        import PyTorchUtils
        torch = PyTorchUtils.PyTorchUtilsLogic().torch
        import torchvision
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    except (ImportError, OSError):
        INSTALL_PYTORCHUTILS = True

try:
    import nrrd
except:
    slicer.util.pip_install('pynrrd')
    import nrrd


#
# TorchSequenceSegmentation
#

class TorchSequenceSegmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Torch Sequence Segmentation"
        self.parent.categories = ["Ultrasound"]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Chris Yeung (Queen's Univ.)"]
        self.parent.helpText = """
See more information in <a href="https://github.com/SlicerIGT/aigt/tree/master/SlicerExtension/LiveUltrasoundAi/TorchSequenceSegmentation">module documentation</a>.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)
        slicer.app.connect("startupCompleted()", installPytorchutils)


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

    # TorchSequenceSegmentation1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='TorchSequenceSegmentation',
        sampleName='TorchSequenceSegmentation1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'TorchSequenceSegmentation1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='TorchSequenceSegmentation1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='TorchSequenceSegmentation1'
    )

    # TorchSequenceSegmentation2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='TorchSequenceSegmentation',
        sampleName='TorchSequenceSegmentation2',
        thumbnailFileName=os.path.join(iconsPath, 'TorchSequenceSegmentation2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='TorchSequenceSegmentation2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='TorchSequenceSegmentation2'
    )


def installPytorchutils():
    if INSTALL_PYTORCHUTILS:
        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Information)
        msg.setText("PyTorch is needed for some modules. This can be installed automatically using the PyTorchUtils extension.")
        msg.setInformativeText("Do you want to install this extension now?")
        msg.setWindowTitle("Installing Required Packages")
        yesButton = msg.addButton("Install", qt.QMessageBox.AcceptRole)
        msg.addButton("Continue without Installing", qt.QMessageBox.RejectRole)
        msg.setModal(True)
        msg.exec_()
        if msg.clickedButton() == yesButton:
            manager = slicer.app.extensionsManagerModel()
            manager.connect("extensionInstalled(QString)", onExtensionInstalled)
            manager.downloadAndInstallExtensionByName("PyTorch")


def onExtensionInstalled(extensionName):
    msg = qt.QMessageBox()
    msg.setIcon(qt.QMessageBox.Information)
    msg.setText(f"{extensionName} has been installed. Slicer must be restarted to finish installation.")
    msg.setInformativeText("Do you want to restart Slicer now?")
    msg.setWindowTitle("Restart Slicer")
    yesButton = msg.addButton("Restart Now", qt.QMessageBox.AcceptRole)
    msg.addButton("Restart Later", qt.QMessageBox.RejectRole)
    msg.setModal(False)
    msg.exec_()
    if msg.clickedButton() == yesButton:
        slicer.app.restart()


#
# TorchSequenceSegmentationWidget
#

class TorchSequenceSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    LAYOUT_2D_3D = 501

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        slicer.mymod = self
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/TorchSequenceSegmentation.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = TorchSequenceSegmentationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        # Volume reconstruction parameters
        self.ui.modelComboBox.connect("currentTextChanged(const QString)", self.updateParameterNodeFromGUI)
        self.ui.volumeReconstructionSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.sequenceBrowserSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.allBrowsersCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.verticalFlipCheckbox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.modelInputSizeSpinbox.connect("valueChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.applyLogCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.thresholdSpinBox.connect("valueChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.segmentationBrowserSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.segmentationNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeChanged)
        self.ui.segmentComboBox.connect("currentIndexChanged(int)", self.onSegmentChanged)
        self.ui.skipFrameSpinBox.connect("valueChanged(int)", self.updateParameterNodeFromGUI)

        lastNormalizeSetting = slicer.util.settingsValue(self.logic.LAST_NORMALIZE_SETTING, True, converter=slicer.util.toBool)
        self.ui.normalizeCheckBox.checked = lastNormalizeSetting
        self.ui.normalizeCheckBox.connect("toggled(bool)", self.updateSettingsFromGUI)
        
        lastErosionX = slicer.util.settingsValue(self.logic.LAST_EROSION_X_SETTING, 0)
        lastErosionY = slicer.util.settingsValue(self.logic.LAST_EROSION_Y_SETTING, 0)
        self.ui.edgeErosionXSpinBox.value = float(lastErosionX)
        self.ui.edgeErosionYSpinBox.value = float(lastErosionY)
        self.ui.edgeErosionXSpinBox.connect("valueChanged(double)", self.onErodeEdgeX)
        self.ui.edgeErosionYSpinBox.connect("valueChanged(double)", self.onErodeEdgeY)
        
        # File paths
        # Set last model folder in UI
        lastModelFolder = slicer.util.settingsValue(self.logic.LAST_MODEL_FOLDER_SETTING, "")
        if lastModelFolder:
            self.ui.modelDirectoryButton.directory = lastModelFolder
            models = self.logic.getAllModelPaths()
            self.ui.modelComboBox.clear()
            for model in models:
                self.ui.modelComboBox.addItem(model.split(os.sep)[-1], model)
        self.ui.modelDirectoryButton.connect("directoryChanged(const QString)", self.updateSettingsFromGUI)
        
        # Set last scan conversion path in UI
        lastScanConversionPath = slicer.util.settingsValue(self.logic.LAST_SCAN_CONVERSION_PATH_SETTING, "")
        if lastScanConversionPath:
            self.ui.scanConversionPathLineEdit.currentPath = lastScanConversionPath
            self.logic.loadScanConversion(lastScanConversionPath)
        self.ui.scanConversionPathLineEdit.connect("currentPathChanged(const QString)", self.updateSettingsFromGUI)

        lastOutputFolder = slicer.util.settingsValue(self.logic.LAST_OUTPUT_FOLDER_SETTING, "")
        if lastOutputFolder:
            self.ui.outputDirectoryButton.directory = lastOutputFolder 
        self.ui.outputDirectoryButton.connect("directoryChanged(const QString)", self.updateSettingsFromGUI)

        # Buttons
        self.ui.useIndividualRadioButton.connect("toggled(bool)", self.onModelSelectionMethodChanged)
        self.ui.useAllRadioButton.connect("toggled(bool)", self.onModelSelectionMethodChanged)
        self.ui.inputResliceButton.connect("clicked()", self.onResliceVolume)
        self.ui.startButton.connect("toggled(bool)", self.onStartButton)
        self.ui.exportButton.connect("clicked()", self.onExportButton)
        self.ui.clearScanConversionButton.connect("clicked()", self.onClearScanConversion)
        self.ui.recordAsSegmentationButton.checked = False

        # Tracking widgets
        self.ui.localTrackingButton.checked = False
        self.ui.globalTrackingButton.checked = False
        self.ui.localTrackingButton.connect("toggled(bool)", self.onLocalTrackingButton)
        self.ui.globalTrackingButton.connect("toggled(bool)", self.onGlobalTrackingButton)
        self.ui.windowSizeSpinBox.connect("valueChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.windowTargetFrameComboBox.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.imagePixelNormSpinBox.connect("valueChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.globalROIComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.onGlobalROINodeChanged)
        self.ui.generateROIButton.connect("toggled(bool)", self.onGenerateROIButton)

        # Add custom 2D + 3D layout
        customLayout = """
            <layout type="horizontal" split="true">
                <item>
                <view class="vtkMRMLSliceNode" singletontag="Red">
                    <property name="orientation" action="default">Axial</property>
                    <property name="viewlabel" action="default">R</property>
                    <property name="viewcolor" action="default">#F34A33</property>
                </view>
                </item>
                <item>
                <view class="vtkMRMLViewNode" singletontag="1">
                    <property name="viewlabel" action="default">1</property>
                </view>
                </item>
            </layout>
        """
        customLayoutId = self.LAYOUT_2D_3D
        layoutManager = slicer.app.layoutManager()
        layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(customLayoutId, customLayout)

        viewToolBar = slicer.util.mainWindow().findChild('QToolBar', 'ViewToolBar')
        layoutMenu = viewToolBar.widgetForAction(viewToolBar.actions()[0]).menu()
        layoutSwitchActionParent = layoutMenu
        layoutSwitchAction = layoutSwitchActionParent.addAction("red + 3D side by side")  # add inside layout list
        layoutSwitchAction.setData(customLayoutId)
        layoutSwitchAction.setToolTip('3D and slice view')
        layoutSwitchAction.connect('triggered()', lambda layoutId=customLayoutId: slicer.app.layoutManager().setLayout(layoutId))

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

        # Switch to 2D + 3D layout and enable slice visibility in 3D view
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(self.LAYOUT_2D_3D)
        layoutManager.sliceWidget("Red").sliceController().setSliceVisible(True)

        # Enable sequence browser toolbar
        slicer.modules.sequences.setToolBarVisible(True)

        # Set models to use in parameter node
        useIndividualModel = self.ui.useIndividualRadioButton.checked
        if useIndividualModel:
            if self.ui.modelComboBox.count > 0:
                self.logic.setModelsToUse([self.ui.modelComboBox.itemData(self.ui.modelComboBox.currentIndex)])
            else:
                self.logic.setModelsToUse([])
        else:
            self.logic.setModelsToUse(self.logic.getAllModelPaths())

        # Create and select volume reconstruction node, if not done yet
        if not self.ui.volumeReconstructionSelector.currentNode():
            volumeReconstructionNode = self._parameterNode.GetNodeReference("VolumeReconstruction")
            if not volumeReconstructionNode:
                volumeReconstructionNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVolumeReconstructionNode", "VolumeReconstruction")
            self.ui.volumeReconstructionSelector.setCurrentNode(volumeReconstructionNode)

        # Collapse data probe widget
        slicer.util.findChild(slicer.util.mainWindow(), "DataProbeCollapsibleWidget").collapsed = True

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)
        
        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        
        # Initial GUI update
        self.updateGUIFromParameterNode()
    
    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        volumeReconstructionNode = self._parameterNode.GetNodeReference("VolumeReconstruction")
        wasBlocked = self.ui.volumeReconstructionSelector.blockSignals(True)
        self.ui.volumeReconstructionSelector.setCurrentNode(volumeReconstructionNode)
        self.ui.volumeReconstructionSelector.blockSignals(wasBlocked)

        sequenceBrowser = self._parameterNode.GetNodeReference("SequenceBrowser")
        wasBlocked = self.ui.sequenceBrowserSelector.blockSignals(True)
        self.ui.sequenceBrowserSelector.setCurrentNode(sequenceBrowser)
        self.ui.sequenceBrowserSelector.blockSignals(wasBlocked)

        useAllBrowsers = self._parameterNode.GetParameter("UseAllBrowsers").lower() == "true"
        wasBlocked = self.ui.allBrowsersCheckBox.blockSignals(True)
        self.ui.allBrowsersCheckBox.setChecked(useAllBrowsers)
        if useAllBrowsers:
            self.ui.sequenceBrowserSelector.setEnabled(False)
        else:
            self.ui.sequenceBrowserSelector.setEnabled(True)
        self.ui.allBrowsersCheckBox.blockSignals(wasBlocked)

        inputVolume = self._parameterNode.GetNodeReference("InputVolume")
        wasBlocked = self.ui.inputVolumeSelector.blockSignals(True)
        self.ui.inputVolumeSelector.setCurrentNode(inputVolume)
        self.ui.inputVolumeSelector.blockSignals(wasBlocked)

        segmentationBrowser = self._parameterNode.GetNodeReference("SegmentationBrowser")
        wasBlocked = self.ui.segmentationBrowserSelector.blockSignals(True)
        self.ui.segmentationBrowserSelector.setCurrentNode(segmentationBrowser)
        self.ui.segmentationBrowserSelector.blockSignals(wasBlocked)

        segmentationNode = self._parameterNode.GetNodeReference("Segmentation")
        wasBlocked = self.ui.segmentationNodeSelector.blockSignals(True)
        self.ui.segmentationNodeSelector.setCurrentNode(segmentationNode)
        self.ui.segmentationNodeSelector.blockSignals(wasBlocked)

        numSkipFrames = int(self._parameterNode.GetParameter("NumSkipFrames"))
        self.ui.skipFrameSpinBox.setValue(numSkipFrames)

        flipVertical = self._parameterNode.GetParameter("FlipVertical").lower() == "true"
        self.ui.verticalFlipCheckbox.setChecked(flipVertical)

        applyLog = self._parameterNode.GetParameter("ApplyLogTransform").lower() == "true"
        self.ui.applyLogCheckBox.setChecked(applyLog)

        modelInputSize = self._parameterNode.GetParameter("ModelInputSize")
        self.ui.modelInputSizeSpinbox.setValue(int(modelInputSize) if modelInputSize else 0)

        windowSize = self._parameterNode.GetParameter("WindowSize")
        self.ui.windowSizeSpinBox.setValue(int(windowSize) if windowSize else 0)

        threshold = self._parameterNode.GetParameter("Threshold")
        self.ui.thresholdSpinBox.setValue(int(threshold) if threshold else 0)

        # Tracking parameters
        trackingMethod = self._parameterNode.GetParameter("TrackingMethod")
        if trackingMethod == "Local":
            self.ui.localTrackingButton.setChecked(True)
        elif trackingMethod == "Global":
            self.ui.globalTrackingButton.setChecked(True)
        else:
            self.ui.localTrackingButton.setChecked(False)
            self.ui.globalTrackingButton.setChecked(False)

        windowTargetFrame = self._parameterNode.GetParameter("WindowTargetFrame")
        self.ui.windowTargetFrameComboBox.setCurrentIndex(int(windowTargetFrame) if windowTargetFrame else 0)

        imagePixelNorm = self._parameterNode.GetParameter("ImagePixelNorm")
        self.ui.imagePixelNormSpinBox.setValue(int(imagePixelNorm) if imagePixelNorm else 0)

        globalROI = self._parameterNode.GetNodeReference("ROI")
        wasBlocked = self.ui.globalROIComboBox.blockSignals(True)
        self.ui.globalROIComboBox.setCurrentNode(globalROI)
        self.ui.globalROIComboBox.blockSignals(wasBlocked)

        # Change output transform to parent of input volume
        if inputVolume:
            inputVolumeParent = inputVolume.GetParentTransformNode()
            if inputVolumeParent:
                self._parameterNode.SetNodeReferenceID("OutputTransform", inputVolumeParent.GetID())
            else:
                self._parameterNode.SetNodeReferenceID("OutputTransform", None)
            wasBlocked = self.ui.outputTransformSelector.blockSignals(True)
            self.ui.outputTransformSelector.setCurrentNode(inputVolumeParent)
            self.ui.outputTransformSelector.blockSignals(wasBlocked)
            
        # Enable/disable buttons
        if self.ui.reconstructButton.checked:
            self.ui.startButton.setEnabled(sequenceBrowser
                                            and inputVolume
                                            and volumeReconstructionNode
                                            and self.logic.getModelsToUse())
        else:
            self.ui.startButton.setEnabled(sequenceBrowser
                                            and inputVolume
                                            and self.logic.getModelsToUse())

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        # Update node references
        self._parameterNode.SetNodeReferenceID("VolumeReconstruction", self.ui.volumeReconstructionSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("SequenceBrowser", self.ui.sequenceBrowserSelector.currentNodeID)
        self._parameterNode.SetParameter("UseAllBrowsers", "true" if self.ui.allBrowsersCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputTransform", self.ui.outputTransformSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("SegmentationBrowser", self.ui.segmentationBrowserSelector.currentNodeID)

        # Update other parameters
        self._parameterNode.SetParameter("NumSkipFrames", str(self.ui.skipFrameSpinBox.value))
        self._parameterNode.SetParameter("FlipVertical", "true" if self.ui.verticalFlipCheckbox.checked else "false")
        self._parameterNode.SetParameter("ApplyLogTransform", "true" if self.ui.applyLogCheckBox.checked else "false")
        self._parameterNode.SetParameter("ModelInputSize", str(self.ui.modelInputSizeSpinbox.value))
        self._parameterNode.SetParameter("Threshold", str(self.ui.thresholdSpinBox.value))

        # Tracking parameters
        self._parameterNode.SetParameter("WindowSize", str(self.ui.windowSizeSpinBox.value))
        self._parameterNode.SetParameter("WindowTargetFrame", str(self.ui.windowTargetFrameComboBox.currentIndex))
        self._parameterNode.SetParameter("ImagePixelNorm", str(self.ui.imagePixelNormSpinBox.value))

        # Update individual model to use
        if self.ui.useIndividualRadioButton.checked:
            if self.ui.modelComboBox.count > 0:
                self.logic.setModelsToUse([self.ui.modelComboBox.itemData(self.ui.modelComboBox.currentIndex)])
            else:
                self.logic.setModelsToUse([])

        self._parameterNode.EndModify(wasModified)
    
    def updateSettingsFromGUI(self, caller=None, event=None):
        settings = qt.QSettings()

        # Update scan conversion path
        scanConversionPath = self.ui.scanConversionPathLineEdit.currentPath
        if scanConversionPath != slicer.util.settingsValue(self.logic.LAST_SCAN_CONVERSION_PATH_SETTING, ""):
            settings.setValue(self.logic.LAST_SCAN_CONVERSION_PATH_SETTING, scanConversionPath)
            self.logic.loadScanConversion(scanConversionPath)

        # Save to settings and update combo box
        modelFolder = self.ui.modelDirectoryButton.directory
        if modelFolder != slicer.util.settingsValue(self.logic.LAST_MODEL_FOLDER_SETTING, ""):
            settings.setValue(self.logic.LAST_MODEL_FOLDER_SETTING, modelFolder)
            models = self.logic.getAllModelPaths()
            self.logic.setModelsToUse(models)
            self.ui.modelComboBox.clear()
            for model in models:
                self.ui.modelComboBox.addItem(model.split(os.sep)[-1], model)

        # Update output folder path
        outputFolder = self.ui.outputDirectoryButton.directory
        if outputFolder != slicer.util.settingsValue(self.logic.LAST_OUTPUT_FOLDER_SETTING, ""):
            settings.setValue(self.logic.LAST_OUTPUT_FOLDER_SETTING, outputFolder)
        
        # Update normalize setting
        normalizeInput = self.ui.normalizeCheckBox.checked
        if normalizeInput != slicer.util.settingsValue(self.logic.LAST_NORMALIZE_SETTING, "", converter=slicer.util.toBool):
            settings.setValue(self.logic.LAST_NORMALIZE_SETTING, normalizeInput)
    
    def onClearScanConversion(self):
        self.ui.scanConversionPathLineEdit.currentPath = ""
        settings = qt.QSettings()
        settings.setValue(self.logic.LAST_SCAN_CONVERSION_PATH_SETTING, "")
        self.logic.loadScanConversion(None)

    def onErodeEdgeX(self, value):
        #todo: Save the erosion value to application settings
        settings = qt.QSettings()
        settings.setValue(self.logic.LAST_EROSION_X_SETTING, str(value))

    def onErodeEdgeY(self, value):
        settings = qt.QSettings()
        settings.setValue(self.logic.LAST_EROSION_Y_SETTING, str(value))

    def onGlobalROINodeChanged(self, caller=None, event=None):
        currentNode = self.ui.globalROIComboBox.currentNodeID
        self._parameterNode.SetNodeReferenceID("ROI", currentNode)
        if currentNode:
            self.ui.generateROIButton.checked = False
        else:
            self.ui.generateROIButton.checked = True

    def onGenerateROIButton(self, toggled):
        self._parameterNode.SetParameter("GenerateROI", "true" if toggled else "false")
        if toggled:
            self.ui.globalROIComboBox.setCurrentNode(None)

    def onSegmentationNodeChanged(self, caller=None, event=None):
        self._parameterNode.SetNodeReferenceID("Segmentation", self.ui.segmentationNodeSelector.currentNodeID)
        segmentationNode = self._parameterNode.GetNodeReference("Segmentation")
        if segmentationNode:
            segmentIds = vtk.vtkStringArray()
            segmentationNode.GetSegmentation().GetSegmentIDs(segmentIds)
            for i in range(segmentIds.GetNumberOfValues()):
                segmentName = segmentationNode.GetSegmentation().GetSegment(segmentIds.GetValue(i)).GetName()
                self.ui.segmentComboBox.addItem(segmentName)
    
    def onSegmentChanged(self, index):
        segmentationNode = self._parameterNode.GetNodeReference("Segmentation")
        if segmentationNode:
            ids = vtk.vtkStringArray()
            segmentationNode.GetSegmentation().GetSegmentIDs(ids)
            self._parameterNode.SetParameter("SegmentId", ids.GetValue(index))
    
    def onModelSelectionMethodChanged(self, caller=None, event=None):
        useIndividualModel = self.ui.useIndividualRadioButton.checked
        if useIndividualModel:
            self.ui.modelComboBox.setEnabled(True)
            if self.ui.modelComboBox.count > 0:
                self.logic.setModelsToUse([self.ui.modelComboBox.itemData(self.ui.modelComboBox.currentIndex)])
            else:
                self.logic.setModelsToUse([])
        else:
            self.ui.modelComboBox.setEnabled(False)
            self.logic.setModelsToUse(self.logic.getAllModelPaths())

    def onLocalTrackingButton(self, toggled):
        if toggled:
            self.ui.globalTrackingButton.setChecked(False)
            self._parameterNode.SetParameter("TrackingMethod", "Local")
        else:
            self._parameterNode.SetParameter("TrackingMethod", "None")

    def onGlobalTrackingButton(self, toggled):
        if toggled:
            self.ui.localTrackingButton.setChecked(False)
            self._parameterNode.SetParameter("TrackingMethod", "Global")
        else:
            self._parameterNode.SetParameter("TrackingMethod", "None")

    def onResliceVolume(self):
        inputVolume = self._parameterNode.GetNodeReference("InputVolume")
        if inputVolume:
            resliceDriverLogic = slicer.modules.volumereslicedriver.logic()
            
            # Get red slice node
            layoutManager = slicer.app.layoutManager()
            sliceWidget = layoutManager.sliceWidget("Red")
            sliceNode = sliceWidget.mrmlSliceNode()

            # Update slice using reslice driver
            resliceDriverLogic.SetDriverForSlice(inputVolume.GetID(), sliceNode)
            resliceDriverLogic.SetModeForSlice(resliceDriverLogic.MODE_TRANSVERSE, sliceNode)
            resliceDriverLogic.SetFlipForSlice(True, sliceNode)

            # Fit slice to background
            sliceWidget.sliceController().fitSliceToBackground()
    
    def setPredictionProgressBar(self, numSteps):
        self.ui.taskProgressBar.setMaximum(numSteps)
        self.logic.progressCallback = self.updatePredictionProgressBar
    
    def updatePredictionProgressBar(self, step):
        self.ui.taskProgressBar.setValue(step)
        slicer.app.processEvents()
    
    def setReconstructionProgressBar(self):
        self.ui.taskProgressBar.setMaximum(100)
        reconstructionNode = self._parameterNode.GetNodeReference("VolumeReconstruction")
        reconstructionNode.AddObserver(reconstructionNode.VolumeAddedToReconstruction, self.updateReconstructionProgressBar)

    def updateReconstructionProgressBar(self, caller=None, event=None):
        reconstructionNode = self._parameterNode.GetNodeReference("VolumeReconstruction")
        sequenceBrowser = self._parameterNode.GetNodeReference("SequenceBrowser")
        if reconstructionNode and sequenceBrowser:
            numFrames = sequenceBrowser.GetMasterSequenceNode().GetNumberOfDataNodes()
            progress = (100 * reconstructionNode.GetNumberOfVolumesAddedToReconstruction()) // numFrames
            self.ui.taskProgressBar.setValue(progress)
            slicer.app.processEvents()
            slicer.app.resumeRender()
            slicer.app.pauseRender()
    
    def resetTaskProgressBar(self):
        self.ui.taskProgressBar.setValue(0)
        self.logic.progressCallback = None
        reconstructionNode = self._parameterNode.GetNodeReference("VolumeReconstruction")
        reconstructionNode.RemoveObservers(reconstructionNode.VolumeAddedToReconstruction)
    
    def onStartButton(self, toggled):
        if toggled:
            # Update GUI
            self.ui.startButton.setText("Stop")
            self.ui.useIndividualRadioButton.setEnabled(False)
            self.ui.useAllRadioButton.setEnabled(False)
            self.ui.modelDirectoryButton.setEnabled(False)
            if self.ui.useIndividualRadioButton.checked:
                self.ui.modelComboBox.setEnabled(False)
            self.ui.sequenceBrowserSelector.setEnabled(False)
            self.ui.inputVolumeSelector.setEnabled(False)
            self.ui.volumeReconstructionSelector.setEnabled(False)
            self.ui.reconstructButton.setEnabled(False)
            self.ui.recordAsSegmentationButton.setEnabled(False)

            self.ui.verticalFlipCheckbox.setEnabled(False)
            self.ui.applyLogCheckBox.setEnabled(False)
            self.ui.modelInputSizeSpinbox.setEnabled(False)
            self.ui.outputTransformSelector.setEnabled(False)
            self.ui.scanConversionPathLineEdit.setEnabled(False)
            self.ui.clearScanConversionButton.setEnabled(False)
            
            if self.ui.edgeErosionXSpinBox.value > 0 or self.ui.edgeErosionYSpinBox.value > 0:
                self.logic.loadScanConversion(self.ui.scanConversionPathLineEdit.currentPath)
                self.logic.erodeCurvilinearMask(self.ui.edgeErosionXSpinBox.value, self.ui.edgeErosionYSpinBox.value)
            
            # Overall progress bar
            numModels = len(self.logic.getModelsToUse())
            progressMax = numModels * 2 if self.ui.reconstructButton.checked else numModels
            self.ui.overallProgressBar.setMaximum(progressMax)
            slicer.app.processEvents()

            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            for model in self.logic.getModelsToUse():
                modelName = model.split(os.sep)[-2]
                self.ui.overallStatusLabel.setText(f"Using {modelName}...")
                try:
                    # Generate predictions/reconstructions for each model
                    self.logic.loadModel(model)

                    # Generate predictions and add to sequence browser
                    self.ui.taskStatusLabel.setText("Generating predictions...")

                    # Create a list of sequence browsers nodes that need to be processed
                    if self.logic.getUseAllBrowsers():
                        sequenceBrowserNodes = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
                    else:
                        sequenceBrowserNodes = [self._parameterNode.GetNodeReference("SequenceBrowser")]

                    #todo: Iterate over sequence browser nodes
                    for sequenceBrowser in sequenceBrowserNodes:
                        self._parameterNode.SetNodeReferenceID("SequenceBrowser", sequenceBrowser.GetID())
                        numFrames = sequenceBrowser.GetMasterSequenceNode().GetNumberOfDataNodes() - 1
                        self.setPredictionProgressBar(numFrames)
                        self.logic.segmentSequence(
                            model, 
                            self.ui.recordAsSegmentationButton.checked
                        )
                        self.resetTaskProgressBar()

                        if self.ui.reconstructButton.checked:
                            self.ui.overallProgressBar.setValue(self.ui.overallProgressBar.value + 1)
                            self.ui.taskStatusLabel.setText("Reconstructing volume...")
                            self.setReconstructionProgressBar()
                            slicer.app.processEvents()
                            self.logic.runVolumeReconstruction(model)
                            self.resetTaskProgressBar()

                        self.ui.overallStatusLabel.setText(f"Done using {modelName}")
                except RuntimeError as rte:
                    logging.error(rte)
                    self.logic.isProcessing = False
                    self.logic.stopProcess = False
                    break
                except Exception as e:
                    logging.info(f"Skipping {modelName} due to error: {e}")
                    logging.info(traceback.format_exc())
                    self.logic.isProcessing = False
                    continue
                finally:
                    # Update overall progress bar
                    self.resetTaskProgressBar()
                    self.ui.overallProgressBar.setValue(self.ui.overallProgressBar.value + 1)
                    slicer.app.processEvents()

            # Restore UI
            qt.QApplication.restoreOverrideCursor()
            self.ui.startButton.checked = False
            self.ui.startButton.setText("Start")
            self.ui.overallProgressBar.setValue(0)
            self.ui.taskStatusLabel.setText("Ready")
            self.ui.overallStatusLabel.setText("Ready")
            self.ui.useIndividualRadioButton.setEnabled(True)
            self.ui.useAllRadioButton.setEnabled(True)
            self.ui.modelDirectoryButton.setEnabled(True)
            if self.ui.useIndividualRadioButton.checked:
                self.ui.modelComboBox.setEnabled(True)
            self.ui.sequenceBrowserSelector.setEnabled(True)
            self.ui.inputVolumeSelector.setEnabled(True)
            self.ui.volumeReconstructionSelector.setEnabled(True)
            self.ui.reconstructButton.setEnabled(True)
            self.ui.recordAsSegmentationButton.setEnabled(True)

            self.ui.verticalFlipCheckbox.setEnabled(True)
            self.ui.applyLogCheckBox.setEnabled(True)
            self.ui.modelInputSizeSpinbox.setEnabled(True)
            self.ui.outputTransformSelector.setEnabled(True)
            self.ui.scanConversionPathLineEdit.setEnabled(True)
            self.ui.clearScanConversionButton.setEnabled(True)
            slicer.app.processEvents()
        else:
            if self.logic.isProcessing:
                self.logic.stopProcess = True
    
    def onExportButton(self):
        predictionNodes = slicer.util.getNodes("*_Prediction")
        if len(predictionNodes) == 0:
            logging.error("No predictions to export!")
            return

        self.ui.patientIDLineEdit.setEnabled(False)
        self.ui.sequenceNameLineEdit.setEnabled(False)
        self.ui.outputDirectoryButton.setEnabled(False)
        self.ui.exportButton.setEnabled(False)
        slicer.app.processEvents()

        # Overall task progress bar
        self.ui.overallProgressBar.setMaximum(len(predictionNodes))

        outputFolder = self.ui.outputDirectoryButton.directory
        patientID = self.ui.patientIDLineEdit.text
        sequenceName = self.ui.sequenceNameLineEdit.text

        # Export ultrasound if needed
        if self.ui.exportUltrasoundCheckBox.checked:
            self.ui.overallProgressBar.setMaximum(len(predictionNodes) + 1)
            self.ui.overallStatusLabel.setText(f"Exporting ultrasound sequence...")

            self.logic.exportImageSequenceAsArray(outputFolder, patientID, sequenceName)

            self.ui.overallProgressBar.setValue(self.ui.overallProgressBar.value + 1)
            slicer.app.processEvents()

        # Export segmentations
        for name, proxyNode in predictionNodes.items():
            self.ui.overallStatusLabel.setText(f"Exporting {name}...")
            try:
                self.logic.exportPredictionSequenceAsArray(proxyNode, outputFolder, patientID, sequenceName)
            except Exception as e:
                logging.error(f"Error exporting {name}: {e}")
                continue
            finally:
                self.ui.overallProgressBar.setValue(self.ui.overallProgressBar.value + 1)
                slicer.app.processEvents()
        
        # Restore UI
        self.ui.overallProgressBar.setValue(0)
        self.ui.overallStatusLabel.setText("Ready")
        self.ui.patientIDLineEdit.setEnabled(True)
        self.ui.sequenceNameLineEdit.setEnabled(True)
        self.ui.outputDirectoryButton.setEnabled(True)
        self.ui.exportButton.setEnabled(True)
        slicer.app.processEvents()


#
# TorchSequenceSegmentationLogic
#

class TorchSequenceSegmentationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    LAST_NORMALIZE_SETTING = "TorchSequenceSegmentation/NormalizeInput"
    LAST_MODEL_FOLDER_SETTING = "TorchSequenceSegmentation/LastModelFolder"
    LAST_SCAN_CONVERSION_PATH_SETTING = "TorchSequenceSegmentation/LastScanConversionPath"
    LAST_OUTPUT_FOLDER_SETTING = "TorchSequenceSegmentation/LastOutputFolder"
    LAST_EROSION_X_SETTING = "TorchSequenceSegmentation/LastErosionX"
    LAST_EROSION_Y_SETTING = "TorchSequenceSegmentation/LastErosionY"

    TARGET_CHANNEL_IDX_FIRST = 0
    TARGET_CHANNEL_IDX_MIDDLE = 1
    TARGET_CHANNEL_IDX_LAST = 2

    ATTRIBUTE_PREFIX = "SingleSliceSegmentation_"
    ORIGINAL_IMAGE_INDEX = ATTRIBUTE_PREFIX + "OriginalImageIndex"

    LOGARITHMIC_TRANSFORMATION_DECIMALS = 4

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

        self.progressCallback = None
        self.isProcessing = False
        self.stopProcess = False
        self.model = None
        self.scanConversionDict = None
        self.cart_x = None
        self.cart_y = None
        self.grid_x = None
        self.grid_y = None
        self.vertices = None
        self.weights = None
        self.curvilinear_size = None
        self.curvilinear_mask = None
        self.volRecLogic = slicer.modules.volumereconstruction.logic()
    
    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("UseAllBrowsers"):
            parameterNode.SetParameter("UseAllBrowsers", "false")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")
        if not parameterNode.GetParameter("SegmentName"):
            parameterNode.SetParameter("SegmentName", "Segmentation")
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "127")
        if not parameterNode.GetParameter("NumSkipFrames"):
            parameterNode.SetParameter("NumSkipFrames", "0")
        if not parameterNode.GetParameter("TrackingMethod"):
            parameterNode.SetParameter("TrackingMethod", "None")
        if not parameterNode.GetParameter("GenerateROI"):
            parameterNode.SetParameter("GenerateROI", "true")
    
    def getAllModelPaths(self):
        modelFolder = slicer.util.settingsValue(self.LAST_MODEL_FOLDER_SETTING, "")
        if modelFolder:
            models = glob.glob(os.path.join(modelFolder, "**", "*.pt"), recursive=True)
            normModels = [os.path.normpath(model) for model in models]  # normalize paths
            return normModels
        else:
            return []
    
    def getModelsToUse(self):
        modelListJSON = self.getParameterNode().GetParameter("ModelsToUse")
        modelList = json.loads(modelListJSON)
        return modelList
    
    def setModelsToUse(self, modelPaths):
        modelListJSON = json.dumps(modelPaths)
        self.getParameterNode().SetParameter("ModelsToUse", modelListJSON)
    
    def loadModel(self, modelPath):
        """
        Load PyTorch model from file.
        """
        parameterNode = self.getParameterNode()
        if not modelPath:
            logging.warning("Model path is empty")
            self.model = None
        elif not os.path.isfile(modelPath):
            logging.error("Model file does not exist: " + modelPath)
            self.model = None
        else:
            extra_files = {"config.json": ""}
            self.model = torch.jit.load(modelPath, _extra_files=extra_files).to(DEVICE)
            self.model.eval()

            if extra_files["config.json"]:
                # Check for model input size metadata
                config = json.loads(extra_files["config.json"])
                inputSize = config["shape"][-1]
                parameterNode.SetParameter("ModelInputSize", str(inputSize))  # assume square
                parameterNode.SetParameter("WindowSize", str(config["shape"][1]))

                # check if model uses tracking data in input
                try:
                    useTrackingLayer = config["use_tracking_layer"]
                    if useTrackingLayer:
                        if config["tracking_method"] == "local":
                            parameterNode.SetParameter("TrackingMethod", "Local")
                            parameterNode.SetParameter("WindowTargetFrame", str(config["window_target_frame"]))
                            parameterNode.SetParameter("ImagePixelNorm", str(config["orig_img_size"]))
                        elif config["tracking_method"] == "global":
                            parameterNode.SetParameter("TrackingMethod", "Global")
                except KeyError:  # for backward compatibility
                    parameterNode.SetParameter("TrackingMethod", "None")
    
    def loadScanConversion(self, scanConversionPath):
        if not scanConversionPath:
            logging.warning("Scan conversion path is empty")
            self.scanConversionDict = None
            self.cart_x = None
            self.cart_y = None
        elif not os.path.isfile(scanConversionPath):
            logging.error("Scan conversion file does not exist: " + scanConversionPath)
            self.scanConversionDict = None
            self.cart_x = None
            self.cart_y = None
        else:
            with open(scanConversionPath, "r") as f:
                self.scanConversionDict = yaml.safe_load(f)
        
        if self.scanConversionDict:
            # Load scan conversion parameters
            self.curvilinear_size = self.scanConversionDict["curvilinear_image_size"]
            initial_radius = np.deg2rad(self.scanConversionDict["angle_min_degrees"])
            final_radius = np.deg2rad(self.scanConversionDict["angle_max_degrees"])
            radius_start_px = self.scanConversionDict["radius_start_pixels"]
            radius_end_px = self.scanConversionDict["radius_end_pixels"]
            num_samples_along_lines = self.scanConversionDict["num_samples_along_lines"]
            num_lines = self.scanConversionDict["num_lines"]
            center_coordinate_pixel = self.scanConversionDict["center_coordinate_pixel"]

            theta, r = np.meshgrid(np.linspace(initial_radius, final_radius, num_samples_along_lines),
                                   np.linspace(radius_start_px, radius_end_px, num_lines))

            # Precompute mapping parameters between scan converted and curvilinear images
            self.cart_x = r * np.cos(theta) + center_coordinate_pixel[0]
            self.cart_y = r * np.sin(theta) + center_coordinate_pixel[1]

            self.grid_x, self.grid_y = np.mgrid[0:self.curvilinear_size, 0:self.curvilinear_size]

            triangulation = Delaunay(np.vstack((self.cart_x.flatten(), self.cart_y.flatten())).T)

            simplices = triangulation.find_simplex(np.vstack((self.grid_x.flatten(), self.grid_y.flatten())).T)
            self.vertices = triangulation.simplices[simplices]

            X = triangulation.transform[simplices, :2]
            Y = np.vstack((self.grid_x.flatten(), self.grid_y.flatten())).T - triangulation.transform[simplices, 2]
            b = np.einsum('ijk,ik->ij', X, Y)
            self.weights = np.c_[b, 1 - b.sum(axis=1)]

            # Compute curvilinear mask, one pixel tighter to avoid artifacts
            angle1 = 90.0 + (self.scanConversionDict["angle_min_degrees"] + 1)
            angle2 = 90.0 + (self.scanConversionDict["angle_max_degrees"] - 1)

            self.curvilinear_mask = np.zeros((self.curvilinear_size, self.curvilinear_size), dtype=np.int8)
            self.curvilinear_mask = cv2.ellipse(self.curvilinear_mask,
                                                (center_coordinate_pixel[1], center_coordinate_pixel[0]),
                                                (radius_end_px - 1, radius_end_px - 1), 0.0, angle1, angle2, 1, -1)
            self.curvilinear_mask = cv2.circle(self.curvilinear_mask,
                                               (center_coordinate_pixel[1], center_coordinate_pixel[0]),
                                               radius_start_px + 1, 0, -1)
            self.curvilinear_mask = self.curvilinear_mask.astype(np.uint8)  # Convert mask_array to uint8
    
    def scanConvert(self, linearArray):
        z = linearArray.flatten()
        zi = np.einsum("ij,ij->i", np.take(z, self.vertices), self.weights)
        return zi.reshape(self.curvilinear_size, self.curvilinear_size)
    
    def erodeCurvilinearMask(self, edgeErosionX, edgeErosionY):
        # Erode mask by 10 percent of the image size to remove artifacts on the edges
        if self.curvilinear_mask is not None and (edgeErosionX > 0 or edgeErosionY > 0):
            # Repaint the borders of the mask to zero to allow erosion from all sides
            self.curvilinear_mask[0, :] = 0
            self.curvilinear_mask[:, 0] = 0
            self.curvilinear_mask[-1, :] = 0
            self.curvilinear_mask[:, -1] = 0
            # Erode the mask
            erosionSizeX = int(edgeErosionX * self.curvilinear_size)
            erosionSizeY = int(edgeErosionY * self.curvilinear_size)
            self.curvilinear_mask = cv2.erode(self.curvilinear_mask, np.ones((erosionSizeX, erosionSizeY), np.uint8), iterations=1)

    def getUniqueName(self, node, baseName):
        newName = baseName
        if node:
            className = node.GetClassName()
            nodes = slicer.util.getNodesByClass(className)
            names = sorted([node.GetName() for node in nodes if node.GetName().startswith(baseName)])
            if names:
                try:
                    lastNumber = int(names[-1][-1])
                    newName += f"_{str(lastNumber + 1)}"
                except ValueError:
                    newName += "_1"
        return newName

    def getUseAllBrowsers(self):
        parameterNode = self.getParameterNode()
        return parameterNode.GetParameter("UseAllBrowsers") == "true"

    def getPrediction(self, inputArray, inputTfmArray=None):
        if not self.model:
            return

        parameterNode = self.getParameterNode()
        
        # Flip image vertically if specified by user
        toFlip = parameterNode.GetParameter("FlipVertical").lower() == "true"
        if toFlip:
            inputArray = np.flip(inputArray, axis=0)

        # Normalize input if needed
        normalizeInput = slicer.util.settingsValue(self.LAST_NORMALIZE_SETTING, False, converter=slicer.util.toBool)
        if normalizeInput:
            if inputArray.max() <= 1.0:
                logging.info("Input image is already between 0 and 1, skipping normalization.")
            else:
                inputArray = inputArray.astype(float) / 255.0

        # Convert to tensor and add batch dimension
        inputTensor = torch.from_numpy(inputArray).unsqueeze(0).float().to(DEVICE)
        if inputTfmArray is not None:
            inputTfmTensor = torch.from_numpy(inputTfmArray).unsqueeze(0).float().to(DEVICE)

        # Run prediction
        with torch.inference_mode():
            if inputTfmArray is not None:
                output = self.model((inputTensor, inputTfmTensor))
            else:
                output = self.model(inputTensor)
        
        if isinstance(output, list):
            output = output[0]
        output = torch.nn.functional.softmax(output, dim=1)
        outputArray = output.detach().cpu().numpy()
        outputArray = outputArray[0, 1, :, :]
        
        # Flip output back if needed
        if toFlip:
            outputArray = np.flip(outputArray, axis=0)

        if parameterNode.GetParameter("ApplyLogTransform").lower() == "true":
            e = self.LOGARITHMIC_TRANSFORMATION_DECIMALS
            outputArray = np.log10(np.clip(outputArray, 10 ** (-e), 1.0) * (10 ** e)) / e

        outputArray *= 255
        outputArray = np.clip(outputArray, 0, 255)
        outputArray = outputArray.astype(np.uint8)[np.newaxis, ...]

        return outputArray
    
    def segmentSequence(self, modelName, recordAsSegmentation=False):
        self.isProcessing = True

        parameterNode = self.getParameterNode()
        sequencesLogic = slicer.modules.sequences.logic()
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")
        inputVolume = parameterNode.GetNodeReference("InputVolume")
        inputSequence = sequenceBrowser.GetSequenceNode(inputVolume)
        modelBasename = modelName.split(os.sep)[-2]
        segmentName = parameterNode.GetParameter("SegmentName")
        threshold = int(parameterNode.GetParameter("Threshold"))

        # tracking parameters
        trackingMethod = parameterNode.GetParameter("TrackingMethod")
        if trackingMethod != "None":
            inputTransform = inputVolume.GetParentTransformNode()
            if trackingMethod == "Local":
                windowSize = int(parameterNode.GetParameter("WindowSize"))
                windowTargetFrame = int(parameterNode.GetParameter("WindowTargetFrame"))
                if windowTargetFrame == self.TARGET_CHANNEL_IDX_MIDDLE:
                    windowTargetFrame = windowSize // 2
                elif windowTargetFrame == self.TARGET_CHANNEL_IDX_LAST:
                    windowTargetFrame = windowSize - 1
                # calculate scaling matrix
                imagePixelNorm = int(parameterNode.GetParameter("ImagePixelNorm"))
                imageToNorm = np.diag([*([1 / imagePixelNorm] * 3), 1])
            else:  # global tracking
                globalROINode = parameterNode.GetNodeReference("ROI")
                generateROI = parameterNode.GetParameter("GenerateROI").lower() == "true"
                if not globalROINode or generateROI:
                    globalROINode = self.addROINode(inputVolume)

                # calculate centering translation matrix
                center = globalROINode.GetCenterWorld()
                centeringMat = np.eye(4)
                centeringMat[:3, 3] = [-center[0], -center[1], -center[2]]

                # calculate scaling matrix
                size = np.zeros(3)
                globalROINode.GetSizeWorld(size)
                rangeZ = size[2]
                scalingFactor = 2 / rangeZ
                scalingMat = np.eye(4)
                scalingMat[0, 0] = scalingFactor
                scalingMat[1, 1] = scalingFactor
                scalingMat[2, 2] = scalingFactor

                # compute final normalization matrix
                imageToNorm = scalingMat @ centeringMat

        # Make new prediction volume to not overwrite existing one
        predictionVolume = parameterNode.GetNodeReference("PredictionVolume")
        # volumeName = self.getUniqueName(predictionVolume, f"{modelBasename}_Prediction")
        # predictionVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", volumeName)
        predictionVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Prediction")
        predictionVolume.CreateDefaultDisplayNodes()
        parameterNode.SetNodeReferenceID("PredictionVolume", predictionVolume.GetID())

        # Add a new sequence node to the sequence browser
        masterSequenceNode = sequenceBrowser.GetMasterSequenceNode()
        predictionSequenceNode = sequencesLogic.AddSynchronizedNode(None, predictionVolume, sequenceBrowser)
        sequenceBrowser.SetRecording(predictionSequenceNode, True)

        # Add segmentation node from prediction volume
        if recordAsSegmentation:
            segSeqBr = parameterNode.GetNodeReference("SegmentationBrowser")
            if not segSeqBr:
                # Create new segmentation sequence browser with Image_Image, ImageToReference, Segmentation
                segSeqBr = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", "SegmentationBrowser")
                parameterNode.SetNodeReferenceID("SegmentationBrowser", segSeqBr.GetID())
                inputSequenceSeg = sequencesLogic.AddSynchronizedNode(None, inputVolume, segSeqBr)
                segSeqBr.SetRecording(inputSequenceSeg, True)

                # Add transforms up to ImageToReference to sequence browser
                refTransformFound = False
                currentTransform = inputVolume
                while not refTransformFound:
                    parentNode = currentTransform.GetParentTransformNode()
                    transformSequenceNode = sequencesLogic.AddSynchronizedNode(None, parentNode, segSeqBr)
                    segSeqBr.SetRecording(transformSequenceNode, True)
                    if "ToRef" in parentNode.GetName():
                        refTransformFound = True
                    else:
                        currentTransform = parentNode
            
            ids = vtk.vtkStringArray()
            segmentationNode = parameterNode.GetNodeReference("Segmentation")
            if not segmentationNode:
                # Create new segmentation node
                segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation")
                segmentationNode.CreateDefaultDisplayNodes()
                segmentationNode.GetDisplayNode().SetVisibility(True)
                segmentId = segmentationNode.GetSegmentation().AddEmptySegment()
                ids.InsertNextValue(segmentId)
                # segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
                parameterNode.SetNodeReferenceID("Segmentation", segmentationNode.GetID())

                # Add segmentation node to sequence browser
                segSequenceNode = sequencesLogic.AddSynchronizedNode(None, segmentationNode, segSeqBr)
                segSeqBr.SetRecording(segSequenceNode, True)
            
            if segmentationNode.GetSegmentation().GetNumberOfSegments() == 0:
                segmentId = segmentationNode.GetSegmentation().AddEmptySegment()
            else:
                segmentId = parameterNode.GetParameter("SegmentId")
            ids.InsertNextValue(segmentId)

            numSkipFrames = int(parameterNode.GetParameter("NumSkipFrames"))

        # Place in output transform if it exists
        outputTransform = parameterNode.GetNodeReference("OutputTransform")
        if outputTransform:
            predictionVolume.SetAndObserveTransformNodeID(outputTransform.GetID())
            if recordAsSegmentation:
                segmentationNode.SetAndObserveTransformNodeID(outputTransform.GetID())

        # Overlay prediction volume in slice view
        predictionDisplayNode = predictionVolume.GetDisplayNode()
        predictionDisplayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGreen")
        slicer.util.setSliceViewerLayers(foreground=predictionVolume, foregroundOpacity=0.3)

        selectedItemNumber = sequenceBrowser.GetSelectedItemNumber()  # for restoring later
        # Iterate through each item in sequence browser and add generated segmentation
        for itemIndex in range(sequenceBrowser.GetNumberOfItems()):
            if self.stopProcess:
                raise RuntimeError("Processing stopped by user")
            
            # Get current frame
            image = inputSequence.GetNthDataNode(itemIndex)
            imageArray = slicer.util.arrayFromVolume(image)
            originalImageShape = imageArray.shape
            sequenceBrowser.SetSelectedItemNumber(itemIndex)

            # Use inverse scan conversion if specified by user, otherwise resize
            if self.scanConversionDict:
                imageArray = map_coordinates(imageArray[0, :, :], [self.cart_x, self.cart_y], order=1)
            else:
                inputSize = int(parameterNode.GetParameter("ModelInputSize"))
                imageArray = cv2.resize(imageArray[0, :, :], (inputSize, inputSize))  # default is bilinear
            
            # get tracking data if needed
            if trackingMethod != "None":
                tfmArray = slicer.util.arrayFromTransformMatrix(inputTransform, toWorld=True)
                if trackingMethod == "Global":  # global normalization
                    tfmArray = imageToNorm @ tfmArray

            # create numpy array from frame buffer
            if trackingMethod == "Local":
                if itemIndex == 0:  # initialize buffer
                    frameBufferList = [imageArray] * windowSize
                    transformBufferList = [tfmArray] * windowSize
                elif itemIndex < windowSize:
                    for i in range(itemIndex, windowSize):
                        frameBufferList[i] = imageArray
                        transformBufferList[i] = tfmArray
                else:  # update buffer
                    frameBufferList.pop(0)
                    frameBufferList.append(imageArray)
                    transformBufferList.pop(0)
                    transformBufferList.append(tfmArray)
                inputArray = np.stack(frameBufferList, axis=0)
                inputTfmArray = np.stack(transformBufferList, axis=0)
            else:
                inputArray = np.expand_dims(imageArray, axis=0)
                if trackingMethod == "Global":
                    inputTfmArray = np.expand_dims(tfmArray, axis=0)
            
            if trackingMethod != "None":
                if trackingMethod == "Local":  # normalize tracking data in window if needed
                    # apply transformation to each frame
                    refToImageMain = np.linalg.inv(inputTfmArray[windowTargetFrame])
                    for i in range(windowSize):
                        inputTfmArray[i] = imageToNorm @ refToImageMain @ inputTfmArray[i]
                    inputTfmArray = inputTfmArray.astype(np.float32)

                # get segmentation
                prediction = self.getPrediction(inputArray, inputTfmArray)
            else:
                # Generate segmentation
                prediction = self.getPrediction(inputArray)

            # Scan convert or resize
            if self.scanConversionDict:
                prediction = self.scanConvert(prediction)
                prediction *= self.curvilinear_mask
            else:
                prediction = cv2.resize(prediction, (originalImageShape[2], originalImageShape[1]))

            slicer.util.updateVolumeFromArray(predictionVolume, prediction)
            indexValue = masterSequenceNode.GetNthIndexValue(itemIndex)
            predictionSequenceNode.SetDataNodeAtValue(predictionVolume, indexValue)

            if recordAsSegmentation:
                # only record proxy nodes every NumSkipFrames frames
                if itemIndex % (numSkipFrames + 1) == 0:
                    # Create temporary label map
                    labelmapVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                    slicer.modules.volumes.logic().CreateLabelVolumeFromVolume(slicer.mrmlScene, labelmapVolume, predictionVolume)

                    # Fill label map by thresholding prediction
                    labelmapArray = slicer.util.arrayFromVolume(labelmapVolume)
                    labelmapArray[:, prediction < threshold] = 0
                    labelmapArray[:, prediction >= threshold] = 1
                    slicer.util.arrayFromVolumeModified(labelmapVolume)

                    # Import label map to segmentation
                    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolume, segmentationNode, ids)

                    # Add segmentation to sequence browser
                    segmentationNode.SetAttribute(self.ORIGINAL_IMAGE_INDEX, str(itemIndex))  # set index for TSA module
                    segSeqBr.SaveProxyNodesState()
                    segmentationNode.SetAttribute(self.ORIGINAL_IMAGE_INDEX, "None")
                    # segSequenceNode.SetDataNodeAtValue(segmentationNode, indexValue)
                    slicer.mrmlScene.RemoveNode(labelmapVolume)

            if self.progressCallback:
                self.progressCallback(itemIndex)
        sequenceBrowser.SetSelectedItemNumber(selectedItemNumber)

        self.isProcessing = False
    
    def addROINode(self, volumeNode):
        parameterNode = self.getParameterNode()
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")
        sequenceName = sequenceBrowser.GetName()

        # Create new ROI node
        roiNode = parameterNode.GetNodeReference("ROI")
        roiName = self.getUniqueName(roiNode, f"{sequenceName}_ROI")
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", roiName)
        parameterNode.SetNodeReferenceID("ROI", roiNode.GetID())
        roiNode.SetDisplayVisibility(False)
        
        self.volRecLogic.CalculateROIFromVolumeSequence(sequenceBrowser, volumeNode, roiNode)

        return roiNode

    def addReconstructionVolume(self, modelName, sequenceName):
        parameterNode = self.getParameterNode()

        # Replace underscores with dashes in model name and sequence name so the volumes can be used by the comparison module

        modelName = modelName.split(os.sep)[-2]
        modelName = modelName.replace("_", "-")
        sequenceName = sequenceName.replace("_", "-")

        # Default scan orientation is sagittal, but if "ax" is in the sequence name, then it is axial
        scanName = "Sagittal"
        if "ax" in sequenceName.lower():
            scanName = "Axial"

        volumeName = sequenceName + "_" + modelName + "_" + scanName

        reconstructionVolume = parameterNode.GetNodeReference("ReconstructionVolume")
        reconstructionName = self.getUniqueName(reconstructionVolume, volumeName)
        reconstructionVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", reconstructionName)
        reconstructionVolume.CreateDefaultDisplayNodes()
        parameterNode.SetNodeReferenceID("ReconstructionVolume", reconstructionVolume.GetID())
        
        return reconstructionVolume

    def runVolumeReconstruction(self, modelName):
        self.isProcessing = True

        volRenLogic = slicer.modules.volumerendering.logic()

        parameterNode = self.getParameterNode()
        reconstructionNode = parameterNode.GetNodeReference("VolumeReconstruction")
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")
        predictionVolume = parameterNode.GetNodeReference("PredictionVolume")

        # Set volume reconstruction parameters
        reconstructionNode.SetAndObserveInputSequenceBrowserNode(sequenceBrowser)
        reconstructionNode.SetAndObserveInputVolumeNode(predictionVolume)

        roiNode = parameterNode.GetNodeReference("ROI")
        if not roiNode:
            roiNode = self.addROINode(predictionVolume)
        reconstructionNode.SetAndObserveInputROINode(roiNode)

        # Set reconstruction output volume
        sequenceName = sequenceBrowser.GetName()
        reconstructionVolume = self.addReconstructionVolume(modelName, sequenceName)
        reconstructionNode.SetAndObserveOutputVolumeNode(reconstructionVolume)

        # Set volume rendering properties
        volRenDisplayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(reconstructionVolume)
        volRenDisplayNode.SetAndObserveROINodeID(roiNode.GetID())
        volPropertyNode = volRenDisplayNode.GetVolumePropertyNode()
        volPropertyNode.Copy(volRenLogic.GetPresetByName("MR-Default"))
        reconstructionVolume.SetDisplayVisibility(False)

        # Run volume reconstruction
        self.volRecLogic.ReconstructVolumeFromSequence(reconstructionNode)

        self.isProcessing = False
    
    def exportImageSequenceAsArray(self, outputFolder, patientID, sequenceName):
        # Get volume sequence data
        parameterNode = self.getParameterNode()
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")
        imageSequence = sequenceBrowser.GetMasterSequenceNode()
        sampleImage = imageSequence.GetNthDataNode(0)

        # Create array for storing each frame
        numFrames = sequenceBrowser.GetNumberOfItems()
        imageSize = slicer.util.arrayFromVolume(sampleImage).shape
        arraySize = (numFrames, imageSize[1], imageSize[2], 1)
        imageArray = self.convertSequenceToArray(imageSequence, arraySize)
        
        # Save ultrasound array
        filename = f"{patientID}_{sequenceName}.nrrd"
        fullPath = os.path.join(outputFolder, filename)
        nrrd.write(fullPath, imageArray, compression_level=1)
    
    def exportPredictionSequenceAsArray(self, proxyNode, outputFolder, patientID, sequenceName):
        # Get volume sequence data
        parameterNode = self.getParameterNode()
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")
        predictionSequence = sequenceBrowser.GetSequenceNode(proxyNode)

        # Create array for storing each frame
        numFrames = sequenceBrowser.GetNumberOfItems()
        imageSize = slicer.util.arrayFromVolume(proxyNode).shape
        arraySize = (numFrames, imageSize[1], imageSize[2], 1)
        predictionArray = self.convertSequenceToArray(predictionSequence, arraySize)
        
        # Save prediction array
        modelName = proxyNode.GetName().split("_Prediction")[0].replace("_", "-")
        filename = f"{patientID}_{modelName}_{sequenceName}.nrrd"
        fullPath = os.path.join(outputFolder, filename)
        nrrd.write(fullPath, predictionArray, compression_level=1)

    @staticmethod
    def convertSequenceToArray(sequenceNode, outputShape):
        # Create array for storing each frame
        outputArray = np.zeros(outputShape, dtype=np.uint8)

        # Iterate through each frame
        for itemIndex in range(outputShape[0]):
            # Add image to array
            currentImage = sequenceNode.GetNthDataNode(itemIndex)
            resizedImage = np.expand_dims(slicer.util.arrayFromVolume(currentImage), axis=3)
            if resizedImage.max() > 1:
                outputArray[itemIndex, ...] = resizedImage
            else:
                outputArray[itemIndex, ...] = resizedImage * 255

        return outputArray


#
# TorchSequenceSegmentationTest
#

class TorchSequenceSegmentationTest(ScriptedLoadableModuleTest):
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
        self.test_TorchSequenceSegmentation1()

    def test_TorchSequenceSegmentation1(self):
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
        inputVolume = SampleData.downloadSample('TorchSequenceSegmentation1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = TorchSequenceSegmentationLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')