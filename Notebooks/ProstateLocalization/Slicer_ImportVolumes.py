import numpy as np


def import_numpy_array(fileName, volumeName, data_type):
  print("Importing {} into {} of voxel type {}".format(fileName, volumeName, dataType))
  numpyArray = np.load(fileName)
  volumeNode = slicer.mrmlScene.GetFirstNode(volumeName, 'vtkMRMLScalarVolumeNode')
  if volumeNode is None:
    volumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    volumeNode.SetName(volumeName)
    volumeNode.CreateDefaultDisplayNodes()
  
  slicer.util.updateVolumeFromArray(volumeNode, numpyArray)
  slicer.util.setSliceViewerLayers(background=volumeNode)


files = [r"j:\Temp\ultrasound_0.npy", r"j:\Temp\segmentation_0.npy", r"j:\Temp\prediction_0.npy"]
volumeNames = ["Ultrasound", "Segmentation", "Prediction"]
dataTypes = ["Double", "Double", "Float"]
for fileName, volumeName, dataType in zip(files, volumeNames, dataTypes):
  import_numpy_array(fileName, volumeName, dataType)


import_numpy_array(r"j:\Temp\ultrasound_0.npy", "Ultrasound", "Double")
import_numpy_array(r"j:\Temp\segmentation_0.npy", "Segmentation", "Double")
import_numpy_array(r"j:\Temp\prediction_0.npy", "Prediction", "Float")


import_numpy_array(r"j:\UsAnnotationExport\Notebooks\ProstateLocalization\bx_00.npy", "bx_00", "Double")
import_numpy_array(r"j:\UsAnnotationExport\Notebooks\ProstateLocalization\bx_01.npy", "bx_01", "Double")
import_numpy_array(r"j:\UsAnnotationExport\Notebooks\ProstateLocalization\bx_02.npy", "bx_02", "Double")
import_numpy_array(r"j:\UsAnnotationExport\Notebooks\ProstateLocalization\bx_03.npy", "bx_03", "Double")


import_numpy_array(r"j:\UsAnnotationExport\Notebooks\ProstateLocalization\sample_ultrasound_00.npy", "sample_ultrasound_00", "Double")
import_numpy_array(r"j:\UsAnnotationExport\Notebooks\ProstateLocalization\sample_ultrasound_01.npy", "sample_ultrasound_01", "Double")
import_numpy_array(r"j:\UsAnnotationExport\Notebooks\ProstateLocalization\sample_ultrasound_02.npy", "sample_ultrasound_02", "Double")


import_numpy_array(r"j:\UsAnnotationExport\Notebooks\ProstateLocalization\sample_segmentation_00.npy", "sample_segmentation_00", "Double")
import_numpy_array(r"j:\UsAnnotationExport\Notebooks\ProstateLocalization\sample_segmentation_01.npy", "sample_segmentation_01", "Double")
import_numpy_array(r"j:\UsAnnotationExport\Notebooks\ProstateLocalization\sample_segmentation_02.npy", "sample_segmentation_02", "Double")
