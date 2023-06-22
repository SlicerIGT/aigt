nodeName = "SCN_17_meroleges_LLA_433.dcm.dcm"
videoVolume = slicer.util.getNode(nodeName)
videoArray = slicer.util.arrayFromVolume(videoVolume)
index = 0.00
sequenceNode = slicer.vtkMRMLSequenceNode()
slicer.mrmlScene.AddNode(sequenceNode)
for i in range(videoArray.shape[0]):
    index += 0.1
    indexFloat = round(float(index), 2)
    newVideoNode = slicer.vtkMRMLVectorVolumeNode()
    newVideoNode.SetName(nodeName + "-Sequence_" + str(i).zfill(4))

    imageSpacing = [0.2, 0.2, 0.2]
    imageData = vtk.vtkImageData()

    imageMat = np.rot90(videoArray[i], 3)
    destinationArray = vtk.util.numpy_support.numpy_to_vtk(imageMat.transpose(2, 1, 0).ravel(), deep=True)
    imageData.SetDimensions(imageMat.shape)
    imageData.GetPointData().SetScalars(destinationArray)

    newVideoNode.SetAndObserveImageData(imageData)
    sequenceNode.SetDataNodeAtValue(newVideoNode, str(index))