# Inputs

inputFolder = r"i:\Data\0_SegmentedScenes"
outputFolder = r"i:\Data\1_ExportedArrays"

import os  # ensure os is imported

# Build a list of all .mrb files in the inputFolder and subdirectories
file_paths = []
for root, _, files in os.walk(inputFolder):
    file_paths.extend(os.path.join(root, f) for f in files if f.endswith('.mrb'))
total_files = len(file_paths)
processed_count = 0

# Make sure the output folder exists

if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

# Get the time series annotation module components

tsaWidget = slicer.modules.timeseriesannotation.widgetRepresentation().self()
tsaLogic = tsaWidget.logic
tsaNode = tsaLogic.getParameterNode()

# Find the nodes necessary for setting up the TSA module

def findInputsForTsaNode(tsaNode):
    """
    Function to find the necessary nodes for the TSA module.
    :return: None if every input was found, string with error message otherwise.
    """
    inputSequenceBrowser = None
    inputVolume = None
    segmentationSequenceBrowser = None
    segmentation = None

    for i in range(slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSequenceBrowserNode')):
        sequenceBrowser = slicer.mrmlScene.GetNthNodeByClass(i, 'vtkMRMLSequenceBrowserNode')
        nodeName = sequenceBrowser.GetName()
        # If 'Sagittal' or 'Axial' is in the name, it is the input browser
        if 'Sagittal' in nodeName or 'Axial' in nodeName:
            inputSequenceBrowser = sequenceBrowser
        elif 'Segmentation' in nodeName:
            segmentationSequenceBrowser = sequenceBrowser
        else:
            print(f'Skipping unknown sequence browser: {nodeName}')

    if inputSequenceBrowser is None:
        return('No input sequence browser found!')
    if segmentationSequenceBrowser is None:
        return('No segmentation sequence browser found!')
    
    inputVolume = slicer.util.getFirstNodeByClassByName('vtkMRMLVolumeNode', 'Image_Image')
    if inputVolume is None:
        return('No input volume found!')
    
    segmentation = slicer.util.getFirstNodeByClassByName('vtkMRMLSegmentationNode', 'Segmentation')
    if segmentation is None:
        return('No segmentation found!')
    
    tsaNode.inputBrowser = inputSequenceBrowser
    tsaNode.inputVolume = inputVolume
    tsaNode.segmentationBrowser = segmentationSequenceBrowser
    tsaNode.segmentation = segmentation
    
    return None


# Iterate over the list of file paths
for filePath in file_paths:
    print(f'Processing file: {filePath}')
    
    try:
        slicer.util.loadScene(filePath)
    except Exception as e:
        print(f'Trying to ignore error loading file: {filePath}')
    
    # Switch to red view-only layout for faster processing
    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
    
    slicer.app.processEvents()
    error = findInputsForTsaNode(tsaNode)
    if error:
        print(f'Error: {error}')
        processed_count += 1
        print(f'Progress: {processed_count} of {total_files} files processed.')
        continue
    
    # Try to find the substring "_Ax_" in the file name to identify Axial scan. Otherwise it is Sagittal.
    
    if "_Ax_" in filePath:
        scanId = 'Ax'
    else:
        scanId = 'Sa'
    
    patientId = os.path.basename(filePath).split('_')[0]
    if len(patientId) < 1:
        print(f"Warning: Patient ID not found in file name: {filePath}")
        patientId = 'Unknown'
    
    prefix = f'{patientId}_{scanId}'

    tsaLogic.exportArrays(outputFolder, prefix)
    slicer.mrmlScene.Clear(0)
    slicer.app.processEvents()
    processed_count += 1
    print(f'Finished processing file: {filePath}')
    print(f'Progress: {processed_count} of {total_files} files processed.')
