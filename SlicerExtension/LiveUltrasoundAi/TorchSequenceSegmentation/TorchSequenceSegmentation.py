import logging
import os
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
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#TorchSequenceSegmentation">module documentation</a>.
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
        self.ui.modelPathLineEdit.connect("currentPathChanged(const QString)", self.updateParameterNodeFromGUI)
        self.ui.volumeReconstructionSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.sequenceBrowserSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.verticalFlipCheckbox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.modelInputSizeSpinbox.connect("valueChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.scanConversionPathLineEdit.connect("currentPathChanged(const QString)", self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.inputResliceButton.connect("clicked(bool)", self.onInputResliceButton)
        self.ui.segmentButton.connect("clicked(bool)", self.onSegmentButton)
        self.ui.reconstructButton.connect("clicked(bool)", self.onReconstructButton)

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

        # Set last model path in UI
        lastModelPath = slicer.util.settingsValue(self.logic.LAST_MODEL_PATH_SETTING, "")
        if lastModelPath:
            self.ui.modelPathLineEdit.currentPath = lastModelPath
        
        # Set last scan conversion path in UI
        lastScanConversionPath = slicer.util.settingsValue(self.logic.LAST_SCAN_CONVERSION_PATH_SETTING, "")
        if lastScanConversionPath:
            self.ui.scanConversionPathLineEdit.currentPath = lastScanConversionPath

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
        self.ui.modelPathLineEdit.setCurrentPath(self._parameterNode.GetParameter("ModelPath"))

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

        self.ui.scanConversionPathLineEdit.setCurrentPath(self._parameterNode.GetParameter("ScanConversionPath"))
            
        # Enable/disable buttons
        self.ui.segmentButton.setEnabled(sequenceBrowser and inputVolume and not self.logic.isProcessing)
        self.ui.reconstructButton.setEnabled(volumeReconstructionNode and sequenceBrowser and not self.logic.isProcessing)

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

        # Update model path and load model
        modelPath = self.ui.modelPathLineEdit.currentPath
        if not modelPath:
            self._parameterNode.SetParameter("ModelPath", "")
        else:
            if modelPath != self._parameterNode.GetParameter("ModelPath"):
                self._parameterNode.SetParameter("ModelPath", modelPath)
                self.logic.loadModel(modelPath)

        # Update scan conversion path
        scanConversionPath = self.ui.scanConversionPathLineEdit.currentPath
        if not scanConversionPath:
            self._parameterNode.SetParameter("ScanConversionPath", "")
        else:
            if scanConversionPath != self._parameterNode.GetParameter("ScanConversionPath"):
                self._parameterNode.SetParameter("ScanConversionPath", scanConversionPath)
                self.logic.loadScanConversion(scanConversionPath)

        self._parameterNode.EndModify(wasModified)
    
    def onInputResliceButton(self):
        inputVolume = self._parameterNode.GetNodeReference("InputVolume")
        if inputVolume:
            self.resliceVolume(inputVolume)

    def onPredictionResliceButton(self):
        predictionVolume = self._parameterNode.GetNodeReference("PredictionVolume")
        if predictionVolume:
            self.resliceVolume(predictionVolume)

    def resliceVolume(self, volumeNode):
        resliceDriverLogic = slicer.modules.volumereslicedriver.logic()
        
        # Get red slice node
        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget("Red")
        sliceNode = sliceWidget.mrmlSliceNode()

        # Update slice using reslice driver
        resliceDriverLogic.SetDriverForSlice(volumeNode.GetID(), sliceNode)
        resliceDriverLogic.SetModeForSlice(resliceDriverLogic.MODE_TRANSVERSE, sliceNode)
        resliceDriverLogic.SetFlipForSlice(True, sliceNode)

        # Fit slice to background
        sliceWidget.sliceController().fitSliceToBackground()
    
    def updatePredictionProgressBar(self, step):
        """
        Update progress bar for prediction.
        """
        self.ui.statusProgressBar.setValue(step)
        slicer.app.processEvents()

    def onSegmentButton(self):
        """
        Generate segmentations for each frame and add to sequence browser.
        """
        # Update progress bar and GUI
        # segmentButtonBlocked = self.ui.segmentButton.blockSignals(True)
        # reconstructButtonBlocked = self.ui.reconstructButton.blockSignals(True)
        self.ui.segmentButton.setEnabled(False)
        self.ui.reconstructButton.setEnabled(False)
        self.ui.statusLabel.setText("Generating predictions...")
        self.ui.modelPathLineEdit.setEnabled(False)
        self.ui.sequenceBrowserSelector.setEnabled(False)
        self.ui.inputVolumeSelector.setEnabled(False)
        self.ui.verticalFlipCheckbox.setEnabled(False)
        self.ui.modelInputSizeSpinbox.setEnabled(False)
        self.ui.outputTransformSelector.setEnabled(False)
        sequenceBrowser = self._parameterNode.GetNodeReference("SequenceBrowser")
        self.ui.statusProgressBar.setMaximum(sequenceBrowser.GetNumberOfItems() - 1)
        # Progress bar callback
        self.logic.progressCallback = self.updatePredictionProgressBar
        slicer.app.processEvents()
        
        try:
            # Run predictions
            self.logic.segmentSequence()
            self.ui.statusLabel.setText("Ready")
        except Exception as e:
            # Restore GUI
            logging.error(e)
            self.ui.statusLabel.setText("Error")
        finally:
            # self.ui.segmentButton.blockSignals(segmentButtonBlocked)
            # self.ui.reconstructButton.blockSignals(reconstructButtonBlocked)
            self.ui.segmentButton.setEnabled(True)
            self.ui.reconstructButton.setEnabled(True)
            self.ui.modelPathLineEdit.setEnabled(True)
            self.ui.sequenceBrowserSelector.setEnabled(True)
            self.ui.inputVolumeSelector.setEnabled(True)
            self.ui.verticalFlipCheckbox.setEnabled(True)
            self.ui.modelInputSizeSpinbox.setEnabled(True)
            self.ui.outputTransformSelector.setEnabled(True)
            self.ui.statusProgressBar.setValue(0)
    
    def updateReconstructionProgressBar(self, caller=None, event=None):
        """
        Update progress bar for volume reconstruction.
        """
        reconstructionNode = self._parameterNode.GetNodeReference("VolumeReconstruction")
        sequenceBrowser = self._parameterNode.GetNodeReference("SequenceBrowser")
        if reconstructionNode and sequenceBrowser:
            numFrames = sequenceBrowser.GetMasterSequenceNode().GetNumberOfDataNodes()
            progress = (100 * reconstructionNode.GetNumberOfVolumesAddedToReconstruction()) // numFrames
            self.ui.statusProgressBar.setValue(progress)
            slicer.app.processEvents()
            slicer.app.resumeRender()
            slicer.app.pauseRender()

    def onReconstructButton(self):
        """
        Render volume reconstruction when user clicks "Render" button.
        """
        # Update progress bar and GUI
        # reconstructButtonBlocked = self.ui.reconstructButton.blockSignals(True)
        # segmentButtonBlocked = self.ui.segmentButton.blockSignals(True)
        self.ui.segmentButton.setEnabled(False)
        self.ui.reconstructButton.setEnabled(False)
        self.ui.statusLabel.setText("Reconstructing volume...")
        self.ui.volumeReconstructionSelector.setEnabled(False)
        self.ui.statusProgressBar.setMaximum(100)
        reconstructionNode = self._parameterNode.GetNodeReference("VolumeReconstruction")
        reconstructionNode.AddObserver(reconstructionNode.VolumeAddedToReconstruction, self.updateReconstructionProgressBar)
        slicer.app.processEvents()

        try:
            self.logic.runVolumeReconstruction()
            self.ui.statusLabel.setText("Ready")
        except Exception as e:
            logging.error(e)
            self.ui.statusLabel.setText("Error")
        finally:
            # self.ui.reconstructButton.blockSignals(reconstructButtonBlocked)
            # self.ui.segmentButton.blockSignals(segmentButtonBlocked)
            self.ui.segmentButton.setEnabled(True)
            self.ui.reconstructButton.setEnabled(True)
            self.ui.volumeReconstructionSelector.setEnabled(True)
            self.ui.statusProgressBar.setValue(0)
            reconstructionNode.RemoveObservers(reconstructionNode.VolumeAddedToReconstruction)


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

    LAST_MODEL_PATH_SETTING = "TorchSequenceSegmentation/LastModelPath"
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

        settings = qt.QSettings()
        settings.setValue(self.LAST_MODEL_PATH_SETTING, modelPath)
    
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
        
        settings = qt.QSettings()
        settings.setValue(self.LAST_SCAN_CONVERSION_PATH_SETTING, scanConversionPath)
    
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
    
    def addPredictionVolume(self):
        parameterNode = self.getParameterNode()

        # Make new prediction volume to not overwrite existing one
        predictionVolume = parameterNode.GetNodeReference("PredictionVolume")
        volumeName = self.getUniqueName(predictionVolume, "Prediction")
        predictionVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", volumeName)
        predictionVolume.CreateDefaultDisplayNodes()
        parameterNode.SetNodeReferenceID("PredictionVolume", predictionVolume.GetID())
        
        # Place in output transform if it exists
        outputTransform = parameterNode.GetNodeReference("OutputTransform")
        if outputTransform:
            predictionVolume.SetAndObserveTransformNodeID(outputTransform.GetID())
        
        return predictionVolume
    
    def addPredictionSequenceNode(self, predictionVolume):
        parameterNode = self.getParameterNode()
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")

        # Add a new sequence node to the sequence browser
        masterSequenceNode = sequenceBrowser.GetMasterSequenceNode()
        sequenceName = self.getUniqueName(masterSequenceNode, "PredictionSequence")
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
            imageArray = np.flip(imageArray, axis=0)  # axis 0 is channel dimension
        
        # Use inverse scan conversion if specified by user, otherwise resize
        if self.scanConversionDict:
            inputArray = map_coordinates(imageArray[0, :, :], [self.cart_x, self.cart_y], order=1)
        else:
            inputSize = int(parameterNode.GetParameter("ModelInputSize"))
            inputArray = cv2.resize(imageArray[0, :, :], (inputSize, inputSize), antialias=True)  # default is bilinear

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
            outputArray = cv2.resize(outputArray, (imageArray.shape[2], imageArray.shape[1]))

        outputArray = outputArray.astype(np.uint8)[np.newaxis, ...]

        # Flip output back if needed
        if toFlip:
            outputArray = np.flip(outputArray, axis=1)

        return outputArray
    
    def segmentSequence(self):
        self.isProcessing = True

        parameterNode = self.getParameterNode()
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")
        inputVolume = parameterNode.GetNodeReference("InputVolume")
        inputSequence = sequenceBrowser.GetSequenceNode(inputVolume)

        # Create prediction sequence
        predictionVolume = self.addPredictionVolume()
        predictionSequenceNode = self.addPredictionSequenceNode(predictionVolume)

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
    
    def addROINode(self):
        parameterNode = self.getParameterNode()
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")
        predictionVolume = parameterNode.GetNodeReference("PredictionVolume")

        # Create new ROI node
        roiNode = parameterNode.GetNodeReference("ROI")
        roiName = self.getUniqueName(roiNode, "ROI")
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLAnnotationROINode", roiName)
        parameterNode.SetNodeReferenceID("ROI", roiNode.GetID())
        roiNode.SetDisplayVisibility(False)
        
        self.volRecLogic.CalculateROIFromVolumeSequence(sequenceBrowser, predictionVolume, roiNode)

        return roiNode

    def addReconstructionVolume(self):
        parameterNode = self.getParameterNode()

        reconstructionVolume = parameterNode.GetNodeReference("ReconstructionVolume")
        reconstructionName = self.getUniqueName(reconstructionVolume, "ReconstructionVolume")
        reconstructionVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", reconstructionName)
        reconstructionVolume.CreateDefaultDisplayNodes()
        parameterNode.SetNodeReferenceID("ReconstructionVolume", reconstructionVolume.GetID())
        
        return reconstructionVolume

    def runVolumeReconstruction(self):
        self.isProcessing = True

        volRenLogic = slicer.modules.volumerendering.logic()

        parameterNode = self.getParameterNode()
        reconstructionNode = parameterNode.GetNodeReference("VolumeReconstruction")
        sequenceBrowser = parameterNode.GetNodeReference("SequenceBrowser")
        predictionVolume = parameterNode.GetNodeReference("PredictionVolume")

        # Set volume reconstruction parameters
        reconstructionNode.SetAndObserveInputSequenceBrowserNode(sequenceBrowser)
        reconstructionNode.SetAndObserveInputVolumeNode(predictionVolume)
        # reconstructionNode.SetInterpolationMode(reconstructionNode.LINEAR_INTERPOLATION)

        roiNode = self.addROINode()
        reconstructionNode.SetAndObserveInputROINode(roiNode)

        # Set reconstruction output volume
        reconstructionVolume = self.addReconstructionVolume()
        reconstructionNode.SetAndObserveOutputVolumeNode(reconstructionVolume)

        # Set volume rendering properties
        volRenDisplayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(reconstructionVolume)
        volRenDisplayNode.SetAndObserveROINodeID(roiNode.GetID())
        volPropertyNode = volRenDisplayNode.GetVolumePropertyNode()
        volPropertyNode.Copy(volRenLogic.GetPresetByName("MR-Default"))

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