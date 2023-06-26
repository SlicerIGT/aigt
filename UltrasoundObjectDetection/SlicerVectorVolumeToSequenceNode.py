import numpy as np

#nodeName = "SCN_22_meroleges_RUA_334"
#videoVolume = slicer.util.getNode(nodeName)
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
    newVideoNode.SetName(videoVolume.GetName() + "-Sequence_" + str(i).zfill(4))

    imageSpacing = [0.2, 0.2, 0.2]
    imageData = vtk.vtkImageData()

    imageMat = np.rot90(videoArray[i], 3)
    imageMat = np.flip(imageMat, axis=0)
    destinationArray = vtk.util.numpy_support.numpy_to_vtk(imageMat.transpose(2, 1, 0).ravel(), deep=True)
    imageData.SetDimensions(imageMat.shape)
    imageData.GetPointData().SetScalars(destinationArray)

    newVideoNode.SetAndObserveImageData(imageData)
    sequenceNode.SetDataNodeAtValue(newVideoNode, str(index))