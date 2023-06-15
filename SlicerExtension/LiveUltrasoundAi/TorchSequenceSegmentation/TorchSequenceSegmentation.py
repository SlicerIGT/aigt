import logging
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


#
# TorchSequenceSegmentation
#

class TorchSequenceSegmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Torch Sequence Segmentation"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Ultrasound"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Chris Yeung (Queen's Univ.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
See more information in <a href="https://github.com/SlicerIGT/aigt/tree/master/SlicerExtension/LiveUltrasoundAi/TorchSequenceSegmentation">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
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
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.verticalFlipCheckbox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.modelInputSizeSpinbox.connect("valueChanged(int)", self.updateParameterNodeFromGUI)

        # File paths
        # Set last model folder in UI
        lastModelFolder = slicer.util.settingsValue(self.logic.LAST_MODEL_FOLDER_SETTING, "")
        if lastModelFolder:
            self.ui.modelDirectoryButton.directory = lastModelFolder
            models = self.logic.getAllModelPaths()
            self.ui.modelComboBox.clear()
            for model in models:
                self.ui.modelComboBox.addItem(model.split(os.sep)[-2], model)
        self.ui.modelDirectoryButton.connect("directoryChanged(const QString)", self.updateSettingsFromGUI)
        
        # Set last scan conversion path in UI
        lastScanConversionPath = slicer.util.settingsValue(self.logic.LAST_SCAN_CONVERSION_PATH_SETTING, "")
        if lastScanConversionPath:
            self.ui.scanConversionPathLineEdit.currentPath = lastScanConversionPath
        self.ui.scanConversionPathLineEdit.connect("currentPathChanged(const QString)", self.updateSettingsFromGUI)

        # Buttons
        self.ui.useIndividualRadioButton.connect("toggled(bool)", self.onModelSelectionMethodChanged)
        self.ui.useAllRadioButton.connect("toggled(bool)", self.onModelSelectionMethodChanged)
        self.ui.inputResliceButton.connect("clicked()", self.onResliceVolume)
        self.ui.startButton.connect("clicked()", self.onStartButton)
        self.ui.clearScanConversionButton.connect("clicked()", self.onClearScanConversion)

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
            self.logic.setModelsToUse([self.ui.modelComboBox.itemData(self.ui.modelComboBox.currentIndex)])
        else:
            self.logic.setModelsToUse(self.logic.getAllModelPaths())

        # Create and select volume reconstruction node, if not done yet
        if not self.ui.volumeReconstructionSelector.currentNode():
            volumeReconstructionNode = self._parameterNode.GetNodeReference("VolumeReconstruction")
            if not volumeReconstructionNode:
                volumeReconstructionNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVolumeReconstructionNode", "VolumeReconstruction")
            self.ui.volumeReconstructionSelector.setCurrentNode(volumeReconstructionNode)

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

        inputVolume = self._parameterNode.GetNodeReference("InputVolume")
        wasBlocked = self.ui.inputVolumeSelector.blockSignals(True)
        self.ui.inputVolumeSelector.setCurrentNode(inputVolume)
        self.ui.inputVolumeSelector.blockSignals(wasBlocked)

        flipVertical = self._parameterNode.GetParameter("FlipVertical").lower() == "true"
        self.ui.verticalFlipCheckbox.setChecked(flipVertical)

        modelInputSize = self._parameterNode.GetParameter("ModelInputSize")
        self.ui.modelInputSizeSpinbox.setValue(int(modelInputSize) if modelInputSize else 0)

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
        if self.ui.reconstructCheckBox.checked:
            self.ui.startButton.setEnabled(sequenceBrowser
                                            and inputVolume
                                            and volumeReconstructionNode
                                            and self.logic.getModelsToUse()
                                            and not self.logic.isProcessing)
        else:
            self.ui.startButton.setEnabled(sequenceBrowser
                                            and inputVolume
                                            and self.logic.getModelsToUse()
                                            and not self.logic.isProcessing)

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
        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputTransform", self.ui.outputTransformSelector.currentNodeID)

        # Update other parameters
        self._parameterNode.SetParameter("FlipVertical", "true" if self.ui.verticalFlipCheckbox.checked else "false")
        self._parameterNode.SetParameter("ModelInputSize", str(self.ui.modelInputSizeSpinbox.value))

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
            self.ui.modelComboBox.clear()
            for model in models:
                self.ui.modelComboBox.addItem(model.split(os.sep)[-2], model)
    
    def onClearScanConversion(self):
        self.ui.scanConversionPathLineEdit.currentPath = ""
        settings = qt.QSettings()
        settings.setValue(self.logic.LAST_SCAN_CONVERSION_PATH_SETTING, "")
    
    def onModelSelectionMethodChanged(self, caller=None, event=None):
        useIndividualModel = self.ui.useIndividualRadioButton.checked
        if useIndividualModel:
            self.ui.modelComboBox.setEnabled(True)
            self.logic.setModelsToUse([self.ui.modelComboBox.itemData(self.ui.modelComboBox.currentIndex)])
        else:
            self.ui.modelComboBox.setEnabled(False)
            self.logic.setModelsToUse(self.logic.getAllModelPaths())

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
    
    def onStartButton(self):
        # Update GUI
        self.ui.startButton.setEnabled(False)
        self.ui.useIndividualRadioButton.setEnabled(False)
        self.ui.useAllRadioButton.setEnabled(False)
        self.ui.modelDirectoryButton.setEnabled(False)
        self.ui.modelComboBox.setEnabled(False)
        self.ui.sequenceBrowserSelector.setEnabled(False)
        self.ui.inputVolumeSelector.setEnabled(False)
        self.ui.volumeReconstructionSelector.setEnabled(False)
        self.ui.reconstructCheckBox.setEnabled(False)

        self.ui.verticalFlipCheckbox.setEnabled(False)
        self.ui.modelInputSizeSpinbox.setEnabled(False)
        self.ui.outputTransformSelector.setEnabled(False)
        self.ui.scanConversionPathLineEdit.setEnabled(False)

        # Overall progress bar
        numModels = len(self.logic.getModelsToUse())
        progressMax = numModels * 2 if self.ui.reconstructCheckBox.checked else numModels
        self.ui.overallProgressBar.setMaximum(progressMax)
        slicer.app.processEvents()

        for model in self.logic.getModelsToUse():
            modelName = model.split(os.sep)[-2]
            self.ui.overallStatusLabel.setText(f"Using {modelName}...")
            try:
                # Generate predictions/reconstructions for each model
                self.logic.loadModel(model)

                # Generate predictions and add to sequence browser
                self.ui.taskStatusLabel.setText("Generating predictions...")
                sequenceBrowser = self._parameterNode.GetNodeReference("SequenceBrowser")
                numFrames = sequenceBrowser.GetMasterSequenceNode().GetNumberOfDataNodes() - 1
                self.setPredictionProgressBar(numFrames)
                self.logic.segmentSequence(model)
                self.resetTaskProgressBar()

                if self.ui.reconstructCheckBox.checked:
                    self.ui.overallProgressBar.setValue(self.ui.overallProgressBar.value + 1)
                    # Reconstruct volume
                    self.ui.taskStatusLabel.setText("Reconstructing volume...")
                    self.setReconstructionProgressBar()
                    slicer.app.processEvents()
                    self.logic.runVolumeReconstruction(model)
                    self.resetTaskProgressBar()

                self.ui.overallStatusLabel.setText(f"Done using {modelName}")
            except Exception as e:
                logging.info(f"Skipping {modelName} due to error: {e}")
                continue
            finally:
                # Update overall progress bar
                self.resetTaskProgressBar()
                self.ui.overallProgressBar.setValue(self.ui.overallProgressBar.value + 1)
                slicer.app.processEvents()

        # Restore UI
        self.ui.startButton.setEnabled(True)
        self.ui.overallProgressBar.setValue(0)
        self.ui.taskStatusLabel.setText("Ready")
        self.ui.overallStatusLabel.setText("Ready")
        self.ui.useIndividualRadioButton.setEnabled(True)
        self.ui.useAllRadioButton.setEnabled(True)
        self.ui.modelDirectoryButton.setEnabled(True)
        self.ui.modelComboBox.setEnabled(True)
        self.ui.sequenceBrowserSelector.setEnabled(True)
        self.ui.inputVolumeSelector.setEnabled(True)
        self.ui.volumeReconstructionSelector.setEnabled(True)
        self.ui.reconstructCheckBox.setEnabled(True)

        self.ui.verticalFlipCheckbox.setEnabled(True)
        self.ui.modelInputSizeSpinbox.setEnabled(True)
        self.ui.outputTransformSelector.setEnabled(True)
        self.ui.scanConversionPathLineEdit.setEnabled(True)
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

    LAST_MODEL_FOLDER_SETTING = "TorchSequenceSegmentation/LastModelFolder"
    LAST_SCAN_CONVERSION_PATH_SETTING = "TorchSequenceSegmentation/LastScanConversionPath"

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

        self.progressCallback = None
        self.isProcessing = False
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
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")
    
    def getAllModelPaths(self):
        modelFolder = slicer.util.settingsValue(self.LAST_MODEL_FOLDER_SETTING, "")
        if modelFolder:
            models = glob.glob(os.path.join(modelFolder, "**", "*traced*.pt"), recursive=True)
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
        if not modelPath:
            logging.warning("Model path is empty")
            self.model = None
        elif not os.path.isfile(modelPath):
            logging.error("Model file does not exist: " + modelPath)
            self.model = None
        else:
            extra_files = {"config.json": ""}
            self.model = torch.jit.load(modelPath, _extra_files=extra_files).to(DEVICE)

            # Check for model input size metadata
            if extra_files["config.json"]:
                config = json.loads(extra_files["config.json"])
                inputSize = config["shape"][-1]
                self.getParameterNode().SetParameter("ModelInputSize", str(inputSize))
    
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
    
    def scanConvert(self, linearArray):
        z = linearArray.flatten()
        zi = np.einsum("ij,ij->i", np.take(z, self.vertices), self.weights)
        return zi.reshape(self.curvilinear_size, self.curvilinear_size)

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
    
    def addPredictionVolume(self, modelName):
        parameterNode = self.getParameterNode()

        # Make new prediction volume to not overwrite existing one
        predictionVolume = parameterNode.GetNodeReference("PredictionVolume")
        volumeName = self.getUniqueName(predictionVolume, f"{modelName.split(os.sep)[-2]}_Prediction")
        predictionVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", volumeName)
        predictionVolume.CreateDefaultDisplayNodes()
        parameterNode.SetNodeReferenceID("PredictionVolume", predictionVolume.GetID())
        
        # Place in output transform if it exists
        outputTransform = parameterNode.GetNodeReference("OutputTransform")
        if outputTransform:
            predictionVolume.SetAndObserveTransformNodeID(outputTransform.GetID())
        
        return predictionVolume
    
    def addPredictionSequenceNode(self, predictionVolume, modelName):
        parameterNode = self.getParameterNode()
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")

        # Add a new sequence node to the sequence browser
        masterSequenceNode = sequenceBrowser.GetMasterSequenceNode()
        sequenceName = self.getUniqueName(masterSequenceNode, f"{modelName.split(os.sep)[-2]}_PredictionSequence")
        predictionSequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", sequenceName)
        sequenceBrowser.AddSynchronizedSequenceNode(predictionSequenceNode)
        sequenceBrowser.AddProxyNode(predictionVolume, predictionSequenceNode, False)

        return predictionSequenceNode

    def getPrediction(self, image):
        if not self.model:
            return
        
        if not image.GetImageData():
            return
        
        imageArray = slicer.util.arrayFromVolume(image)

        # Flip image vertically if specified by user
        parameterNode = self.getParameterNode()
        toFlip = parameterNode.GetParameter("FlipVertical").lower() == "true"
        if toFlip:
            imageArray = np.flip(imageArray, axis=0)
        
        # Use inverse scan conversion if specified by user, otherwise resize
        if self.scanConversionDict:
            inputArray = map_coordinates(imageArray[0, :, :], [self.cart_x, self.cart_y], order=1)
        else:
            inputSize = int(parameterNode.GetParameter("ModelInputSize"))
            inputArray = cv2.resize(imageArray[0, :, :], (inputSize, inputSize))  # default is bilinear

        # Convert to tensor and add batch dimension
        inputTensor = torch.from_numpy(inputArray).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        # Run prediction
        with torch.inference_mode():
            output = self.model(inputTensor)
            output = torch.nn.functional.softmax(output, dim=1)

        # Scan convert or resize
        if self.scanConversionDict:
            outputArray = output.detach().cpu().numpy() * 255
            outputArray = self.scanConvert(outputArray[0, 1, :, :])
            outputArray *= self.curvilinear_mask
        else:
            outputArray = output.squeeze().detach().cpu().numpy() * 255
            outputArray = cv2.resize(outputArray[1], (imageArray.shape[2], imageArray.shape[1]))

        outputArray = outputArray.astype(np.uint8)[np.newaxis, ...]

        # Flip output back if needed
        if toFlip:
            outputArray = np.flip(outputArray, axis=0)

        return outputArray
    
    def segmentSequence(self, modelName):
        self.isProcessing = True

        parameterNode = self.getParameterNode()
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")
        inputVolume = parameterNode.GetNodeReference("InputVolume")
        inputSequence = sequenceBrowser.GetSequenceNode(inputVolume)

        # Create prediction sequence
        predictionVolume = self.addPredictionVolume(modelName)
        predictionSequenceNode = self.addPredictionSequenceNode(predictionVolume, modelName)

        # Overlay prediction volume in slice view
        predictionDisplayNode = predictionVolume.GetDisplayNode()
        predictionDisplayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGreen")
        slicer.util.setSliceViewerLayers(foreground=predictionVolume, foregroundOpacity=0.3)

        # Iterate through each item in sequence browser and add generated segmentation
        selectedItemNumber = sequenceBrowser.GetSelectedItemNumber()  # for restoring later
        for itemIndex in range(sequenceBrowser.GetNumberOfItems()):
            # Generate segmentation
            currentImage = inputSequence.GetNthDataNode(itemIndex)
            sequenceBrowser.SetSelectedItemNumber(itemIndex)
            prediction = self.getPrediction(currentImage)
            slicer.util.updateVolumeFromArray(predictionVolume, prediction)

            # Add segmentation to sequence browser
            indexValue = inputSequence.GetNthIndexValue(itemIndex)
            predictionSequenceNode.SetDataNodeAtValue(predictionVolume, indexValue)
            if self.progressCallback:
                self.progressCallback(itemIndex)
        sequenceBrowser.SetSelectedItemNumber(selectedItemNumber)

        self.isProcessing = False
    
    def addROINode(self, modelName):
        parameterNode = self.getParameterNode()
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")
        predictionVolume = parameterNode.GetNodeReference("PredictionVolume")

        # Create new ROI node
        roiNode = parameterNode.GetNodeReference("ROI")
        roiName = self.getUniqueName(roiNode, f"{modelName.split(os.sep)[-2]}_ROI")
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLAnnotationROINode", roiName)
        parameterNode.SetNodeReferenceID("ROI", roiNode.GetID())
        roiNode.SetDisplayVisibility(False)
        
        self.volRecLogic.CalculateROIFromVolumeSequence(sequenceBrowser, predictionVolume, roiNode)

        return roiNode

    def addReconstructionVolume(self, modelName):
        parameterNode = self.getParameterNode()

        reconstructionVolume = parameterNode.GetNodeReference("ReconstructionVolume")
        reconstructionName = self.getUniqueName(reconstructionVolume, f"{modelName.split(os.sep)[-2]}_ReconstructionVolume")
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

        roiNode = self.addROINode(modelName)
        reconstructionNode.SetAndObserveInputROINode(roiNode)

        # Set reconstruction output volume
        reconstructionVolume = self.addReconstructionVolume(modelName)
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