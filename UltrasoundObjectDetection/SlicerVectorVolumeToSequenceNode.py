'''
Description:
    Helper script to turn a Slicer VectorVolumeNode into a SequenceNode. Necessary for testing real time inference with recorded data, in case
    data was captured as a VectorVolumeNode, where the 3rd dimension represents time. The script iterates through the temporal dimension
    of the VectorVolumeNode, and stitches the 2D frames together into a new SequenceNode

Usage:
    Open the VectorVolume, and copy-paste the below script in python interactor in Slicer. A new sequence will be created from the
    first VectorVolumeNode open in Slicer (relevant if you have more than one open).
'''

import numpy as np

videoVolume = getNodesByClass("vtkMRMLVectorVolumeNode")[0]
videoArray = slicer.util.arrayFromVolume(videoVolume)
index = 0.00
sequenceNode = slicer.vtkMRMLSequenceNode()
slicer.mrmlScene.AddNode(sequenceNode)
sequenceNode.SetName("Image_Reference")
for i in range(videoArray.shape[0]):
    index += 0.1
    indexFloat = round(float(index), 2)
    newVideoNode = slicer.vtkMRMLScalarVolumeNode()

    imageData = vtk.vtkImageData()
    imageMat = np.flip(np.rot90(videoArray[i], 3),axis=0)

    destinationArray = vtk.util.numpy_support.numpy_to_vtk(imageMat.transpose(2, 1, 0).ravel(), deep=True)
    imageData.SetDimensions(imageMat.shape)
    imageData.GetPointData().SetScalars(destinationArray)

    newVideoNode.SetAndObserveImageData(imageData)
    sequenceNode.SetDataNodeAtValue(newVideoNode, str(index))

sequenceBrowser = slicer.vtkMRMLSequenceBrowserNode()
sequenceBrowser.AddSynchronizedSequenceNode(sequenceNode)
slicer.mrmlScene.AddNode(sequenceBrowser)



