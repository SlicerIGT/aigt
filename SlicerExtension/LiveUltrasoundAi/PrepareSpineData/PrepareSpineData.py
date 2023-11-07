import logging
import os

import vtk, qt, ctk, slicer

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import numpy as np


#
# PrepareSpineData
#

class PrepareSpineData(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Prepare spine data"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Ultrasound"]
        self.parent.dependencies = []
        self.parent.contributors = ["Nicole Kitner (Queen's University)"]
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#PrepareSpineData">module documentation</a>.
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

    # PrepareSpineData1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='PrepareSpineData',
        sampleName='PrepareSpineData1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'PrepareSpineData1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='PrepareSpineData1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='PrepareSpineData1'
    )

    # PrepareSpineData2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='PrepareSpineData',
        sampleName='PrepareSpineData2',
        thumbnailFileName=os.path.join(iconsPath, 'PrepareSpineData2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='PrepareSpineData2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='PrepareSpineData2'
    )


#
# PrepareSpineDataWidget
#

class PrepareSpineDataWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

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
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/PrepareSpineData.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = PrepareSpineDataLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneImportEnd)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.ctSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.ultrasoundSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.sequenceRange.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.ui.patientID.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.ui.ctToUsSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onCtToUsNodeSelected)

        # Buttons
        self.ui.generateCrop.connect('clicked(bool)', self.onGenerateCrop)
        self.ui.removeHidden.connect('clicked(bool)', self.onRemoveHidden)
        self.ui.removeUnusedMark.connect('clicked(bool)', self.onRemoveUnusedMark)
        self.ui.removeUnusedSeq.connect('clicked(bool)', self.onRemoveUnusedSeq)
        self.ui.removeUnusedVol.connect('clicked(bool)', self.onRemoveUnusedVol)
        self.ui.volReview.connect('clicked(bool)', self.onVolReview)
        self.ui.seqReview.connect('clicked(bool)', self.onSeqReview)
        # self.ui.genRegistration.connect('clicked(bool)', self.onGenRegistration)
        self.ui.showInteractor.connect('toggled(bool)', self.onShowInteractorToggled)

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

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)


    def onCtToUsNodeSelected(self, ctToUsNode):
        if ctToUsNode is not None:
            self._parameterNode.SetNodeReferenceID(self.logic.CT_TO_US_TRANSFORM, ctToUsNode.GetID())
        else:
            self._parameterNode.SetNodeReferenceID(self.logic.CT_TO_US_TRANSFORM, "")


    def onShowInteractorToggled(self, toggled):
        """
        Callback function for checkbox to toggle registration transform interactor visibility.
        """
        ctToUsNode = self._parameterNode.GetNodeReference(self.logic.CT_TO_US_TRANSFORM)
        if ctToUsNode is None:
            logging.warning("CT to US transform should be referenced, but not found")
            return

        ctToUsNode.GetDisplayNode().SetEditorVisibility(toggled)


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

    def onSceneImportEnd(self, caller, event):
        """
        Called after a scene was loaded in Slicer.
        """
        qt.QTimer.singleShot(0, self.finishImportingScene)  # Need some time for import to finish

    def finishImportingScene(self):
        self.updateGUIFromParameterNode()  # Update GUI using new parameter node

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

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
        if self._parameterNode is not None:
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

        sequenceNodeID = self.ui.inputSelector.currentNodeID
        sequenceNode = slicer.mrmlScene.GetNodeByID(sequenceNodeID)

        # Update node selectors and sliders
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        self.ui.ctSelector.setCurrentNode(self._parameterNode.GetNodeReference("CTVolume"))
        self.ui.ultrasoundSelector.setCurrentNode(self._parameterNode.GetNodeReference("UltrasoundVolume"))
        self.ui.ctToUsSelector.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.CT_TO_US_TRANSFORM))

        self.ui.sequenceRange.minimum = 0
        if sequenceNode is not None:
            # get sequence max
            numSequence = sequenceNode.GetNumberOfItems()
            seqMax = float(numSequence / 10)
            self.ui.sequenceRange.maximum = seqMax
            self.ui.sequenceRange.setValues(0, seqMax)
        else:
            self.ui.sequenceRange.maximum = 100
        self.ui.patientID.text = (self._parameterNode.GetParameter("SequenceRange"))

        ctToUsNode = self._parameterNode.GetNodeReference(self.logic.CT_TO_US_TRANSFORM)
        if ctToUsNode is not None:
            if ctToUsNode.GetDisplayNode() is None:
                ctToUsNode.CreateDefaultDisplayNodes()

            editorVisible = ctToUsNode.GetDisplayNode().GetEditorVisibility()
            self.ui.showInteractor.enabled = True
            if editorVisible:
                self.ui.showInteractor.checked = True
            else:
                self.ui.showInteractor.checked = False
        else:
            self.ui.showInteractor.checked = False
            self.ui.showInteractor.enabled = False

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

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("CTVolume", self.ui.ctSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("UltrasoundVolume", self.ui.ultrasoundSelector.currentNodeID)
        self._parameterNode.SetParameter("SequenceRange", self.ui.patientID.text)
        self._parameterNode.SetParameter("UltrasoundVolume", self.ui.ultrasoundSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onVolReview(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            if self.ui.ctSelector.currentNode() and self.ui.ultrasoundSelector.currentNode():
                ultrasound_name = self.ui.ultrasoundSelector.currentNode().GetName()
                ct_name = self.ui.ctSelector.currentNode().GetName()
                # Compute output
                self.logic.volReviewLogic(ultrasound_name, ct_name)

    def onSeqReview(self):
        # Make sure views are unlinked

        sliceCompositeNodes = slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode")
        defaultSliceCompositeNode = slicer.mrmlScene.GetDefaultNodeByClass("vtkMRMLSliceCompositeNode")
        if not defaultSliceCompositeNode:
            defaultSliceCompositeNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSliceCompositeNode")
            defaultSliceCompositeNode.UnRegister(None)  # CreateNodeByClass is factory method, need to unregister the result
            slicer.mrmlScene.AddDefaultNode(defaultSliceCompositeNode)
        sliceCompositeNodes.append(defaultSliceCompositeNode)
        for sliceCompositeNode in sliceCompositeNodes:
            sliceCompositeNode.SetLinkedControl(False)

        # Setting up volumes for views

        with slicer.util.tryWithErrorDisplay("Failed to set up views for seqeunce review", waitCursor=True):
            if self.ui.ctSelector.currentNode() and self.ui.ultrasoundSelector.currentNode():
                ultrasound_name = self.ui.ultrasoundSelector.currentNode().GetName()
                ct_name = self.ui.ctSelector.currentNode().GetName()
                self.logic.seqReviewLogic(ultrasound_name, ct_name)



    def onGenerateCrop(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            startTimeStamp = self.ui.sequenceRange.minimumValue
            endTimeStamp = self.ui.sequenceRange.maximumValue
            originalSequenceBrowser = self.ui.inputSelector.currentNode().GetName()
            croppedSequenceBrowserName = self.ui.nameSelector.currentText
            eraseOriginal = False
            listSynchronizedNodes = ["Image_Image", "ImageToTransd", "TransdToReference"]

            originalSequenceBrowser = slicer.mrmlScene.GetFirstNodeByName(originalSequenceBrowser)
            croppedSequenceBrowser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
            croppedSequenceBrowser.SetName(slicer.mrmlScene.GetUniqueNameByString(croppedSequenceBrowserName))
            listSequenceNodes = []
            for aSynchronizedNode in listSynchronizedNodes:
                sequenceNodeAdded = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode",
                                                                       "Sequence_" + aSynchronizedNode)
                listSequenceNodes.append(sequenceNodeAdded)

            sequenceBrowserLogic = slicer.modules.sequences.logic()
            for aSequenceNode, aSynchronizedNode in zip(listSequenceNodes, listSynchronizedNodes):
                croppedSequenceBrowser.AddSynchronizedSequenceNodeID(aSequenceNode.GetID())
                sequenceBrowserLogic.AddSynchronizedNode(aSequenceNode,
                                                         slicer.mrmlScene.GetFirstNodeByName(aSynchronizedNode),
                                                         croppedSequenceBrowser)
                croppedSequenceBrowser.SetRecording(aSequenceNode, True)

            startIndex = 0
            endIndex = 0
            for itemIndex in range(1, originalSequenceBrowser.GetNumberOfItems()):
                previousTimeSeconds = float(originalSequenceBrowser.GetFormattedIndexValue(itemIndex - 1))
                currentTimeSeconds = float(originalSequenceBrowser.GetFormattedIndexValue(itemIndex))
                if (startTimeStamp >= previousTimeSeconds) and (
                        startTimeStamp <= currentTimeSeconds):
                    startIndex = itemIndex
                if (endTimeStamp >= previousTimeSeconds) and (endTimeStamp <= currentTimeSeconds):
                    endIndex = itemIndex

            # Set up frame skipping when tracking info is missing (identity transforms)

            transdToReferenceMatrix = vtk.vtkMatrix4x4()
            skipInvalid = self.ui.skipInvalidTrackingCheckBox.checked
            transdToReference = slicer.mrmlScene.GetFirstNodeByName("TransdToReference")
            if transdToReference is None and skipInvalid == True:
                logging.warning("Could not find TransdToReference transform. Not skipping any frames.")
                skipInvalid = False

            # Copy desired sequences, within a period of time, to the new sequence browser
            progressbar = slicer.util.createProgressDialog(parent=slicer.util.mainWindow(), windowTitle='Cropping sequence...',
                                                           autoClose=True)
            progress = 0
            steps = endIndex - startIndex + 1
            progressbar.setCancelButton(None)

            slicer.app.pauseRender()  # Speeds up iteration by blocking rendering in 2D and 3D views

            for itemIndex in range(startIndex, endIndex + 1):
                originalSequenceBrowser.SetSelectedItemNumber(itemIndex)
                if skipInvalid:
                    transdToReference.GetMatrixTransformToParent(transdToReferenceMatrix)
                    p_Ref = transdToReferenceMatrix.MultiplyFloatPoint([0.0, 0.0, 0.0, 1.0])
                    if p_Ref[0] != 0.0 or p_Ref[1] != 0.0 or p_Ref[2] != 0.0:
                        croppedSequenceBrowser.SaveProxyNodesState()
                    else:
                        print(f"Skipping item: {itemIndex}")
                else:
                    croppedSequenceBrowser.SaveProxyNodesState()
                progressStep = 100 / steps
                # Update progress value
                progressbar.setValue(progress)
                progress += progressStep
                slicer.app.processEvents()

            slicer.app.resumeRender()  # Resume rendering in all views

            # Erase the original sequence browser and its linked sequence nodes, if required

            if self.ui.deleteOrig.isChecked():
                sequenceNodesToDelete = vtk.vtkCollection()
                originalSequenceBrowser.GetSynchronizedSequenceNodes(sequenceNodesToDelete, True)
                slicer.mrmlScene.RemoveNode(originalSequenceBrowser)
                for aSequence in sequenceNodesToDelete:
                    slicer.mrmlScene.RemoveNode(aSequence)

            # Check if there is already a segmentation and erase it
            segmentationName = "Segmentation"
            segmentationNode = slicer.mrmlScene.GetFirstNodeByName(segmentationName)
            if segmentationNode:
                slicer.mrmlScene.RemoveNode(segmentationNode)

            # Move Image_image out of any transform
            imageImageNode = slicer.mrmlScene.GetFirstNodeByName("Image_Image")
            imageImageNode.SetAndObserveTransformNodeID("")

            # Add segmentation node
            segmentationNodeNew = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", segmentationName)
            segmentationNodeNew.GetSegmentation().AddEmptySegment("", "Transverse process",
                                                                  (0.501961, 0.682353, 0.501961))
            segmentationNodeNew.GetSegmentation().AddEmptySegment("", "Spinous process", (0.945098, 0.839216, 0.568627))
            segmentationNodeNew.GetSegmentation().AddEmptySegment("", "Other bone", (0.694118, 0.478431, 0.396078))

            # Move Image_image and Segmentation back to ImageToTransd transform
            imageToTransdTransform = slicer.mrmlScene.GetFirstNodeByName("ImageToTransd")
            imageImageNode.SetAndObserveTransformNodeID(imageToTransdTransform.GetID())
            segmentationNodeNew.SetAndObserveTransformNodeID(imageToTransdTransform.GetID())

            # Remove SegmentationBrowser, if it exists
            segmentationBrowserName = "SegmentationBrowser"
            segmentationBrowser = slicer.mrmlScene.GetFirstNodeByName(segmentationBrowserName)
            if segmentationBrowser:
                sequenceNodesToDelete = vtk.vtkCollection()
                segmentationBrowser.GetSynchronizedSequenceNodes(sequenceNodesToDelete, True)
                slicer.mrmlScene.RemoveNode(segmentationBrowser)
                for aSequence in sequenceNodesToDelete:
                    slicer.mrmlScene.RemoveNode(aSequence)

            # Create new sequence browser and desired sequences
            listSynchronizedNodesSegmentation = ["Image_Image", "ImageToTransd", "TransdToReference", segmentationName]
            segmentationBrowser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode",
                                                                     segmentationBrowserName)
            listSequenceNodesSegmentation = []
            for aSynchronizedNode in listSynchronizedNodesSegmentation:
                sequenceNodeAdded = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode",
                                                                       "Sequence_" + aSynchronizedNode)
                listSequenceNodesSegmentation.append(sequenceNodeAdded)

            # Link desired sequences to new sequence browser and link each desired sequence to its corresponding
            # synchronized node
            sequenceBrowserLogic = slicer.modules.sequences.logic()
            for aSequenceNode, aSynchronizedNode in zip(listSequenceNodesSegmentation,
                                                        listSynchronizedNodesSegmentation):
                segmentationBrowser.AddSynchronizedSequenceNodeID(aSequenceNode.GetID())
                sequenceBrowserLogic.AddSynchronizedNode(aSequenceNode,
                                                         slicer.mrmlScene.GetFirstNodeByName(aSynchronizedNode),
                                                         segmentationBrowser)
                segmentationBrowser.SetRecording(aSequenceNode, True)

            progressbar.close()

    def onRemoveHidden(self):
        keepModelNodes = [""] # Not used for now.
        # Delete all volume nodes, except for those that are in the keep list
        print("")
        print("*** Removing modelNodes")
        modelNodes = slicer.util.getNodesByClass("vtkMRMLModelNode")
        numRemoved = 0
        for model in modelNodes:
            modelName = model.GetName()
            if modelName in keepModelNodes or model.GetHideFromEditors():
                print("Keep   {}".format(modelName))
            else:
                print("Delete {}".format(modelName))
                slicer.mrmlScene.RemoveNode(model)
                numRemoved += 1
        self.ui.feedbackLabel.setText(f"{numRemoved} models removed")

    def onRemoveUnusedMark(self):
        # Input parameters
        keepBrowserName = self.ui.ultrasoundSelector.currentNode().GetName()
        CTName = self.ui.ctSelector.currentNode().GetName()
        sequenceName = self.ui.inputSelector.currentNode().GetName()
        patientMarkups = "patientUsLandmarks_Ras"
        keepVolumeNames = [keepBrowserName, CTName, sequenceName, patientMarkups]

        # Delete all markups
        print("")
        print("*** Removing Markup fiducials")
        allMarkups = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
        numRemoved = 0
        for markup in allMarkups:
            if markup.GetName() in keepVolumeNames:
                print("Keep    {}".format(markup.GetName()))
            else:
                print("Delete  {}".format(markup.GetName()))
                slicer.mrmlScene.RemoveNode(markup)
                numRemoved += 1
            self.ui.feedbackLabel.setText(f"{numRemoved} markups removed")

    def onRemoveUnusedSeq(self):
        sequenceName = self.ui.inputSelector.currentNode().GetName()

        # Delete all sequence browsers, except for one
        keepBrowser = slicer.mrmlScene.GetFirstNodeByName(sequenceName)
        sequencesToKeep = vtk.vtkCollection()
        keepBrowser.GetSynchronizedSequenceNodes(sequencesToKeep, True)
        listSynchronizedNodes = []

        print("")
        print("*** Removing sequences")
        allSequences = slicer.util.getNodesByClass("vtkMRMLSequenceNode")
        numRemoved_seq = 0
        for sequence in allSequences:
            if sequencesToKeep.IsItemPresent(sequence):
                proxyNode = keepBrowser.GetProxyNode(sequence)
                listSynchronizedNodes.append(proxyNode.GetName())
                print("Keep    {}".format(sequence.GetName()))
            else:
                print("Delete  {}".format(sequence.GetName()))
                slicer.mrmlScene.RemoveNode(sequence)
                numRemoved_seq += 1

        # Delete all sequences except the ones belonging to the browser to keep
        print("")
        print("*** Removing sequence browsers")
        allSequenceBrowsers = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
        numRemoved_sb = 0
        for browser in allSequenceBrowsers:
            if browser == keepBrowser:
                print("Keep    {}".format(browser.GetName()))
            else:
                print("Delete  {}".format(browser.GetName()))
                slicer.mrmlScene.RemoveNode(browser)
                numRemoved_sb += 1
        self.ui.feedbackLabel.setText(f"{numRemoved_seq} sequences and {numRemoved_sb} sequence browsers removed")

    def onRemoveUnusedVol(self):
        # Input parameters
        keepBrowserName = self.ui.ultrasoundSelector.currentNode().GetName()
        CTName = self.ui.ctSelector.currentNode().GetName()
        sequenceName = self.ui.inputSelector.currentNode().GetName()
        patientMarkups = "patientUsLandmarks_Ras"
        inputImageName = "Image_Image"
        keepVolumeNames = [keepBrowserName, CTName, sequenceName, patientMarkups]

        # Delete all volume nodes, except for those that are in the keep list
        print("")
        print("*** Removing volumes")
        volumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        numRemoved = 0
        for volume in volumeNodes:
            volumeName = volume.GetName()
            if volumeName == inputImageName or volumeName in keepVolumeNames:
                print("Keep   {}".format(volumeName))
            else:
                print("Delete {}".format(volumeName))
                slicer.mrmlScene.RemoveNode(volume)
                numRemoved += 1
        self.ui.feedbackLabel.setText(f"{numRemoved} volumes removed")


#
# PrepareSpineDataLogic
#

class PrepareSpineDataLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    CT_TO_US_TRANSFORM = "CtToUsTransformNode"


    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True,
                                 update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime - startTime:.2f} seconds')

    def volReviewLogic(self, ultrasoundVolumeName, ctVolumeName):

        ultrasoundVolume = slicer.mrmlScene.GetFirstNodeByName(ultrasoundVolumeName)
        ctVolume = slicer.mrmlScene.GetFirstNodeByName(ctVolumeName)
        imageNode = slicer.mrmlScene.GetFirstNodeByName("Image_Image")

        layoutManager = slicer.app.layoutManager()

        redNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeRed')
        redNode.SetSliceResolutionMode(slicer.vtkMRMLSliceNode.SliceResolutionMatchVolumes)
        greenNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeGreen')
        yellowNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeYellow')

        cyan = slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeCyan")
        ultrasoundVolume.GetDisplayNode().SetAndObserveColorNodeID(cyan.GetID())
        yellow = slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeYellow")
        ctVolume.GetDisplayNode().SetAndObserveColorNodeID(yellow.GetID())

        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        resliceLogic = slicer.modules.volumereslicedriver.logic()
        sliceNodeList = [redNode, greenNode, yellowNode]
        sliceNames = ["Red", "Green", "Yellow"]

        for i, sliceNode in enumerate(sliceNodeList):
            resliceLogic.SetDriverForSlice("", sliceNode)  # No driver
            resliceLogic.SetModeForSlice(i, sliceNode)  # Axial, coronal, sagittal
            resliceLogic.SetFlipForSlice(False, sliceNode)
            sliceLogic = layoutManager.sliceWidget(sliceNames[i]).sliceLogic()
            sliceLogic.GetSliceCompositeNode().SetBackgroundVolumeID(ultrasoundVolume.GetID())
            sliceLogic.GetSliceCompositeNode().SetForegroundVolumeID(ctVolume.GetID())
            sliceLogic.GetSliceCompositeNode().SetForegroundOpacity(0.5)
            sliceLogic.FitSliceToAll()
            if i == 0:
                sliceNode.SetOrientationToAxial()
            elif i == 1:
                sliceNode.SetOrientationToCoronal()
            else:
                sliceNode.SetOrientationToSagittal()

        layoutManager.sliceWidget("Red").sliceController().setSliceVisible(False)

    def seqReviewLogic(self, ultrasoundVolumeName, ctVolumeName):

        ultrasoundVolume = slicer.mrmlScene.GetFirstNodeByName(ultrasoundVolumeName)
        ctVolume = slicer.mrmlScene.GetFirstNodeByName(ctVolumeName)
        imageNode = slicer.mrmlScene.GetFirstNodeByName("Image_Image")

        layoutManager = slicer.app.layoutManager()

        redNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeRed')
        redNode.SetSliceResolutionMode(slicer.vtkMRMLSliceNode.SliceResolutionMatchVolumes)
        greenNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeGreen')
        yellowNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeYellow')

        cyan = slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeCyan")
        ultrasoundVolume.GetDisplayNode().SetAndObserveColorNodeID(cyan.GetID())
        yellow = slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeYellow")
        ctVolume.GetDisplayNode().SetAndObserveColorNodeID(yellow.GetID())

        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        resliceLogic = slicer.modules.volumereslicedriver.logic()
        sliceNodeList = [redNode, greenNode, yellowNode]
        sliceNames = ["Red", "Green", "Yellow"]

        for i, sliceNode in enumerate(sliceNodeList):
            resliceLogic.SetDriverForSlice(imageNode.GetID(), sliceNode)  # No driver
            resliceLogic.SetModeForSlice(6, sliceNode)  # Axial, coronal, sagittal
            resliceLogic.SetFlipForSlice(True, sliceNode)
            sliceLogic = layoutManager.sliceWidget(sliceNames[i]).sliceLogic()
            sliceLogic.GetSliceCompositeNode().SetBackgroundVolumeID(imageNode.GetID())
            sliceLogic.GetSliceCompositeNode().SetForegroundVolumeID(ctVolume.GetID())
            sliceLogic.GetSliceCompositeNode().SetForegroundOpacity(0.5)
            sliceLogic.FitSliceToAll()
            if i == 0:
                sliceLogic.GetSliceCompositeNode().SetForegroundOpacity(0.3)
            elif i == 1:
                sliceLogic.GetSliceCompositeNode().SetForegroundOpacity(0.0)
            else:
                sliceLogic.GetSliceCompositeNode().SetForegroundOpacity(0.5)

        layoutManager.sliceWidget("Red").sliceController().setSliceVisible(True)

#
# PrepareSpineDataTest
#

class PrepareSpineDataTest(ScriptedLoadableModuleTest):
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
        self.test_PrepareSpineData1()

    def test_PrepareSpineData1(self):
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
        inputVolume = SampleData.downloadSample('PrepareSpineData1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = PrepareSpineDataLogic()

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
