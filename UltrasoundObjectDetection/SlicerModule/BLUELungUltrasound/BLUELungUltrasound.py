import logging
import os
import subprocess
import string
from PIL import Image
import numpy as np
from ctypes import windll
import cv2
import torch
from ultralytics import YOLO

import vtk
import qt

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


#
# BLUELungUltrasound
#

class BLUELungUltrasound(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "BLUE Lung Ultrasound"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["IGT"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Róbert Zsolt Szabó (Óbuda University)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#BLUELungUltrasound">module documentation</a>.
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

    # BLUELungUltrasound1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='BLUELungUltrasound',
        sampleName='BLUELungUltrasound1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'BLUELungUltrasound1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='BLUELungUltrasound1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='BLUELungUltrasound1'
    )

    # BLUELungUltrasound2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='BLUELungUltrasound',
        sampleName='BLUELungUltrasound2',
        thumbnailFileName=os.path.join(iconsPath, 'BLUELungUltrasound2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='BLUELungUltrasound2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='BLUELungUltrasound2'
    )


#
# BLUELungUltrasoundWidget
#

class BLUELungUltrasoundWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        slicer.BlueLungWidget = self  # then in python interactor, call "self = slicer.mymod" to use
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self._updatingGUIFromMRML = False
        self._updatingGui = False
        self.observedPlusServerLauncherNode = None

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/BLUELungUltrasound.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = BLUELungUltrasoundLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.plusConfigFileSelector.connect('currentPathChanged(const QString)', self.onPlusConfigFileChanged)
        self.ui.plusServerExeSelector.connect('currentPathChanged(const QString)', self.onPlusServerExePathChanged)        

        # Buttons
        self.ui.startPlusButton.connect('toggled(bool)', self.onStartPlusClicked)
        self.ui.setViewButton.connect('clicked(bool)', self.onSetViewButtonClicked)
        self.ui.startInferenceButton.connect('toggled(bool)', self.onStartInferenceButtonClicked)
        self.ui.setCustomUiButton.connect('toggled(bool)', self.onSetCustomUiButtonClicked)
        self.ui.placeMarkupLineButton.connect('clicked(bool)', self.onPlaceMarkupLineClicked)
        self.ui.generateMModeButton.connect('clicked(bool)', self.onGenerateMModeButtonClicked)
        self.ui.toggleTestObserverButton.connect('toggled(bool)', self.onToggleTestObserverButtonClicked)


        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        if self.logic.plus_server_process:
            self.logic.plus_server_process.kill()
        
        if self.logic.inference_server_process:
            self.logic.inference_server_process.kill()

        slicer.mrmlScene.RemoveNode(self.logic.InferenceIgtlConnectorNode)
        slicer.mrmlScene.RemoveNode(self.logic.RawInputIgtlConnectorNode)
        slicer.mrmlScene.RemoveNode(self.logic.InferenceOutputNode)

        self.removeObservers()


    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

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

        self.ui.plusConfigFileSelector.currentPath = self._parameterNode.GetParameter("PLUSConfigFile")
        self.ui.plusServerExeSelector.currentPath = self._parameterNode.GetParameter("PLUSExePath")
        
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
        self._parameterNode.EndModify(wasModified)
    

    def onPlusConfigFileChanged(self, configFilepath):
        logging.info(f"onPlusConfigFileChanged({configFilepath})")
        self._parameterNode.SetParameter("PLUSConfigFile", self.ui.plusConfigFileSelector.currentPath)

    def onPlusServerExePathChanged(self, plusExePath):
        logging.info(f"onPlusServerExePathChanged({plusExePath})")
        self.logic.settings.setValue('BLUELungUltrasound/PLUSExePath', self.ui.plusServerExeSelector.currentPath)
        self._parameterNode.SetParameter("PLUSExePath", self.ui.plusServerExeSelector.currentPath)

    def onStartPlusClicked(self, toggled):
        logging.info(f"onStartPlusClicked({toggled})")
        if toggled:
            self.ui.startPlusButton.text = "Stop PLUS Server"
            self.ui.plusConfigFileSelector.enabled = False
            self.ui.plusServerExeSelector.enabled = False
        else:
            self.ui.startPlusButton.text = "Start PLUS Server"
            self.ui.plusConfigFileSelector.enabled = True
            self.ui.plusServerExeSelector.enabled = True
        
        self.logic.setPlusServerClicked(toggled)

    def onSetViewButtonClicked(self):
        logging.info("onSetViewButtonClicked()")
        self.logic.setViewToIncomingData(self.logic.INPUT_NODE_NAME)

    def onStartInferenceButtonClicked(self, toggled):
        input_volume = slicer.util.getNode(self.logic.INPUT_NODE_NAME)
        logging.info(f'onStartInferenceButtonClicked({toggled})')
        if toggled:
            self.ui.startInferenceButton.text = "Stop Inference"
            self.addObserver(input_volume, slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent, self.logic.PredictStaticSignsOnFrame)
            print('inference START')
        else:
            self.ui.startInferenceButton.text = "Start Inference"
            self.removeObservers(self.logic.PredictStaticSignsOnFrame)
            print('inference STOP')

        #self.logic.ToggleInferenceMode(toggled)

    def onSetCustomUiButtonClicked(self, toggled):
        self.ui.setCustomUiButton.text = "Disable Custom UI" if toggled else "Enable Custom UI"
        self.logic.SetCustomStyle(toggled)

    def onGenerateMModeButtonClicked(self):
        self.logic.ProcessLungSlidingEvaluation(n_seconds=5)


    def onToggleTestObserverButtonClicked(self, toggled):
        input_volume = slicer.util.getNode(self.logic.INPUT_NODE_NAME)
        if toggled:
            self.logic.FRAMES = []
            #input_volume.AddObserver(slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent, self.logic.TestAddFrameToVolume)
            self.addObserver(input_volume, slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent, self.logic.TestAddFrameToVolume)
            print('observer added')
        else:
            #input_volume.RemoveObservers(slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent)
            self.removeObservers(self.logic.TestAddFrameToVolume)
            print('observer removed')


    def onPlaceMarkupLineClicked(self):
        layoutManager = slicer.app.layoutManager()
        redSliceLogic = layoutManager.sliceWidget("Red").sliceLogic()
        transducerCenter = [-95, 461, redSliceLogic.GetSliceOffset()]

        lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        lineNode.SetName("MMode_Line")
        lineNode.AddControlPoint(transducerCenter)

        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetActivePlaceNodeID(lineNode.GetID())
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        placeModePersistence = 1
        interactionNode.SetPlaceModePersistence(placeModePersistence)
        # mode 1 is Place, can also be accessed via slicer.vtkMRMLInteractionNode().Place
        interactionNode.SetCurrentInteractionMode(1)

#
# BLUELungUltrasoundLogic
#

class BLUELungUltrasoundLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    # OpenIGTLink PLUS connection
    CONFIG_FILE_DEFAULT = "default_plus_config.xml"  # Default config file if the user doesn't set another.
    PLUS_SERVER_EXECUTABLE = "PlusServer.exe" # having Plus Toolkit installed is a prerequisite
    INPUT_NODE_NAME = "Image_Reference"
    INFERENCE_NODE_NAME = "Inference"
    IGTL_RAW_INPUT_PORT = 18944 # TODO: read the port from the PLUS config file
    IGTL_INFERENCE_PORT = 18945

    # M-mode stuff (TEMPORARY):
    X_CENTER = 0
    Y_CENTER = 0
    FRAMES = []

    # Model parameters
    #MODEL_WEIGHTS_PATH = 'D:/GitRepos/aigt-LIVE/UltrasoundObjectDetection/YOLOv8/best.pt'
    #MODEL_WEIGHTS_PATH = 'lung_yolov8_pretrained.pt'
    CONFIDENCE_THRESHOLD = 0.55 # TODO: Add as UI parameter
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.settings = qt.QSettings()
        self.plus_server_process = None
        self.inference_server_process = None

        self.setupOpenIgtLink()

        self.InferenceOutputNode = slicer.vtkMRMLVectorVolumeNode()
        self.InferenceOutputNode.SetName(self.INFERENCE_NODE_NAME)
        slicer.mrmlScene.AddNode(self.InferenceOutputNode)
        #self.setupInferenceServer()

        self.model = YOLO(self.resourcePath(f'model/lung_yolov8_pretrained.pt'))        
        
        
    def setupOpenIgtLink(self):

        self.RawInputIgtlConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
        self.RawInputIgtlConnectorNode.SetName('Raw Input')
        self.RawInputIgtlConnectorNode.SetTypeClient('localhost', self.IGTL_RAW_INPUT_PORT)
        slicer.mrmlScene.AddNode(self.RawInputIgtlConnectorNode)
        self.RawInputIgtlConnectorNode.Start()

        self.InferenceIgtlConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
        self.InferenceIgtlConnectorNode.SetName('Inference')
        self.InferenceIgtlConnectorNode.SetTypeClient('localhost', self.IGTL_INFERENCE_PORT)
        slicer.mrmlScene.AddNode(self.InferenceIgtlConnectorNode)

    
    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """

        if not parameterNode.GetParameter("PLUSConfigFile"):
            parameterNode.SetParameter("PLUSConfigFile", self.resourcePath(self.CONFIG_FILE_DEFAULT))
        
        if not parameterNode.GetParameter("PLUSExePath"):
            if not self.settings.value('BLUELungUltrasound/PLUSExePath'):
                self.settings.setValue('BLUELungUltrasound/PLUSExePath', self.find_local_file(self.PLUS_SERVER_EXECUTABLE))
            
            parameterNode.SetParameter("PLUSExePath", self.settings.value('BLUELungUltrasound/PLUSExePath'))

    def resourcePath(self, filename):
        """
        Returns the full path to the given resource file.
        :param filename: str, resource file name
        :returns: str, full path to file specified
        """
        moduleDir = os.path.dirname(slicer.util.modulePath(self.moduleName))
        return os.path.join(moduleDir, 'Resources', filename)

    def setupInferenceServer(self):
        FNULL = open(os.devnull, 'w')
        python_executable = '"C:/Users/Guest admin/anaconda3/envs/pytorch/python.exe"'
        inference_server_script = 'C:/repos/aigt/UltrasoundObjectDetection/RealtimeInferenceOverOpenIGTLink.py'
        args = f'{python_executable} {inference_server_script}'
        print(args)
        self.inference_server_process = subprocess.Popen(args, env=os.environ)
        print('Inference server started')
    
    
    def setPlusServerClicked(self, toggled):
        if toggled:
            FNULL = open(os.devnull, 'w')
            config_file = self.getParameterNode().GetParameter("PLUSConfigFile")
            print(config_file)
            executable = self.getParameterNode().GetParameter("PLUSExePath")
            args = f'"{executable}" --config-file="{config_file}"'
            self.plus_server_process = subprocess.Popen(args, stdout=FNULL, stderr=FNULL, shell=False)
            print('PLUS server started')
        else:
            self.plus_server_process.kill()
            print('PLUS server stopped')


    def setViewToIncomingData(self, nodeName):
        try:
            slicer.util.setSliceViewerLayers(
                foreground=slicer.util.getNode(nodeName).GetID(),
                foregroundOpacity=0,
                fit=True)
        except:
            print("View reset unsuccessful - cannot find incoming data node. Try again in a few seconds")


    def find_local_file(self, filename):        
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drives.append(letter)
            bitmask >>= 1
        
        for drive in drives:
            for root, dirs, files in os.walk(f'{drive}:/'):
                if filename in files:
                    return os.path.join(root, filename)
                
        print("No PLUS installation found")
        return None
    
    def ToggleInferenceMode(self, toggled):
        if toggled:
            self.InferenceIgtlConnectorNode.Start()
            print("Inference running")
        else:
            self.InferenceIgtlConnectorNode.Stop()
            print("Inference stopped")

    def SetCustomStyle(self, visible):
        """
        Applies UI customization. Hide Slicer widgets and apply custom stylesheet.
        :param visible: True to apply custom style.
        :returns: None
        """
        #settings = qt.QSettings()
        #settings.setValue(self.SLICER_INTERFACE_VISIBLE, not visible)

        slicer.util.setToolbarsVisible(not visible)
        slicer.util.setMenuBarsVisible(not visible)
        slicer.util.setApplicationLogoVisible(not visible)
        slicer.util.setModuleHelpSectionVisible(not visible)
        slicer.util.setModulePanelTitleVisible(not visible)
        slicer.util.setDataProbeVisible(not visible)
        slicer.util.setStatusBarVisible(not visible)

        if visible:
            styleFile = self.resourcePath('UI/LumpNav.qss')
            f = qt.QFile(styleFile)
            f.open(qt.QFile.ReadOnly | qt.QFile.Text)
            ts = qt.QTextStream(f)
            stylesheet = ts.readAll()
            slicer.util.mainWindow().setStyleSheet(stylesheet)
        else:
            slicer.util.mainWindow().setStyleSheet("")

        #self.ui.customUiButton.checked = visible


    def TestAddFrameToVolume(self, volumeNode, event):
        #frame = np.expand_dims(slicer.util.arrayFromVolume(volumeNode)[0,:,:], axis=0).copy()
        self.FRAMES.append(slicer.util.arrayFromVolume(volumeNode).copy())
    
    
    def ProcessLungSlidingEvaluation(self, n_seconds=5):
        # 1: find center_point, r1, r2 to get the region of interest
        # 2: Place line markup
        
        # 3: gather frames coming over OpenIGTLink for n_seconds, stitch them together as 3D np array

        
        ultrasound_volume = np.concatenate([np.expand_dims(frame[0,:,:], axis=0) for frame in self.FRAMES], axis=0)
        test_us_im = Image.fromarray(ultrasound_volume[0,:,:])
        test_us_im.save(f'D:/test_us.png')
        
        # 4: generate M-mode image
        mmode_image = self.GenerateMModeImage(ultrasound_volume)
        print(f'n_frames: {len(self.FRAMES)}, mmode shape: {mmode_image.shape}')
        im = Image.fromarray(mmode_image)
        im.save("D:/test_mmode.png")
        # 5: run PTX inference / send M-mode image over OpenIGTLink for inference running script
        # 6: set view layout to side-by-side (layoutManager.setLayout(29))
        # 6: display M-mode image in yellow slice view

    def GenerateMModeImage(self, usVol, imageHeight=256):
        #usVol_flipped = usVol
        usVol_flipped = np.flip(usVol, axis=(1,2))
        center, r1, r2 = self.GetUltrasoundAreaControlPoints(usVol_flipped[0])
        inputPoint = np.flip(slicer.util.arrayFromMarkupsControlPoints(slicer.util.getNode("MMode_Line"))[1][:2])
        print(center, inputPoint)

        unitVector = np.subtract(inputPoint, center)/np.linalg.norm(np.subtract(inputPoint, center)) #Generate the unit vector of the line
        print(unitVector)
        P1, P2 = list(abs(unitVector*r1 + center)), list(abs(unitVector*r2 + center)) #The indices of the line intersections with the radius
        print(f'P1: {P1}, P2: {P2}')
        x, y = np.linspace(P1[0], P2[0], imageHeight).astype(np.uint32), np.linspace(P1[1], P2[1], imageHeight).astype(np.uint32) #A list of imageHeight indices between P1 and P2
        print(f'first point: {x[0]}, {y[0]}; last point: {x[len(x)-1]}, {y[len(y)-1]}')
        mFull = np.column_stack([[frame[xVal,yVal] for xVal, yVal in zip(x,y)] for frame in usVol]) #For each frame, for each [x,y], append the value
        cv2.imshow('mmode', mFull)
        return mFull

    
    def GetUltrasoundAreaControlPoints(self, ultrasound_frame):
        #center_point = [-95, 461]
        center_point = [461, -95]
        r_inner = 216
        r_outer = 591
        return center_point, r_inner, r_outer

    
    def preprocess_epiphan_image(self, image):
        image = np.rot90(np.transpose(image, (1,2,0)), 2)
        if image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=2)
        return np.ascontiguousarray(image)
    
    def PredictStaticSignsOnFrame(self, volumeNode, event):
        image = slicer.util.arrayFromVolume(volumeNode).copy()
        image = self.preprocess_epiphan_image(image)
        #cv2.imshow("input_img", image)

        prediction = self.model(image, conf=self.CONFIDENCE_THRESHOLD, device=self.DEVICE)[0].plot()
        print(prediction.shape)
        #cv2.imshow("pred", prediction)
        #self.PushNumpyDataToVolumeNode(prediction, self.InferenceOutputNode)
        prediction = np.flip(np.expand_dims(prediction, axis=0), axis=(1,2))
        print(prediction.shape)
        slicer.util.updateVolumeFromArray(self.InferenceOutputNode, prediction)



#
# BLUELungUltrasoundTest
#

class BLUELungUltrasoundTest(ScriptedLoadableModuleTest):
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
        self.test_BLUELungUltrasound1()

    def test_BLUELungUltrasound1(self):
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
        inputVolume = SampleData.downloadSample('BLUELungUltrasound1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = BLUELungUltrasoundLogic()

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
