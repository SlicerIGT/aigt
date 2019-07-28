import numpy as np

files = [r"j:\Temp\ultraosund_0.npy", r"j:\Temp\segmentation_0.npy", r"j:\Temp\prediction_0.npy"]
volumeNames = ["Ultrasound", "Segmentation", "Prediction"]
labelmapFlags = [False, False, False]
dataTypes = ["Double", "Double", "Float"]

nImages = len(files)
numpyArray = [None] * nImages
volume = [None] * nImages
volumeDisplayNode = [None] * nImages
importer = [None] * nImages

print("{} images will be loaded".format(nImages))

i = 0
for fileName, volumeName, labelmap, dataType in zip(files, volumeNames, labelmapFlags, dataTypes):
  print("")
  print("Importing {} into {}, type {}".format(fileName, volumeName, dataType))
  numpyArray[i] = np.load(fileName)
  importer[i] = vtk.vtkImageImport()
  importer[i].CopyImportVoidPointer(numpyArray[i], numpyArray[i].nbytes)
  setDataType = 'importer[i].SetDataScalarTypeTo' + dataType + '()'
  eval(setDataType)
  importer[i].SetNumberOfScalarComponents(1)
  importer[i].SetWholeExtent(0,numpyArray[i].shape[0]-1,0,numpyArray[i].shape[1]-1,0,numpyArray[i].shape[2]-1)
  importer[i].SetDataExtentToWholeExtent()
  importer[i].Update()
  if labelmap:
    volume[i] = slicer.vtkMRMLLabelMapVolumeNode()
  else:
    volume[i] = slicer.vtkMRMLScalarVolumeNode()
  volume[i].SetName(volumeName)
  volume[i].SetAndObserveImageData(importer[i].GetOutput())
  slicer.mrmlScene.AddNode(volume[i])
  volumeDisplayNode[i] = 0
  if labelmap == False:
    volumeDisplayNode[i] = slicer.vtkMRMLScalarVolumeDisplayNode()
  else:
    volumeDisplayNode[i] = slicer.vtkMRMLLabelMapVolumeDisplayNode()
  
  slicer.mrmlScene.AddNode(volumeDisplayNode[i])
  if labelmap == False:
    greyColorTable = slicer.util.getNode('Grey')
    volumeDisplayNode[i].SetAndObserveColorNodeID(greyColorTable.GetID())
  else:
    genericColorTableNode = slicer.util.getNode('GenericColors')
    volumeDisplayNode[i].SetAndObserveColorNodeID(genericColorTableNode.GetID())
  
  volume[i].SetAndObserveDisplayNodeID(volumeDisplayNode[i].GetID())
  i = i + 1