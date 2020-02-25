input_browser_name = "LandmarkingScan"
output_browser_name = "ShortScan"
first_index = 10
last_index = 19

input_browser_node = slicer.util.getFirstNodeByName(input_browser_name, className='vtkMRMLSequenceBrowserNode')
if input_browser_node is None:
    logging.error("Could not find input browser node: {}".format(input_browser_node))

output_browser_node = slicer.util.getFirstNodeByName(output_browser_name, className='vtkMRMLSequenceBrowserNode')
if output_browser_node is None:
    output_browser_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceBrowserNode', output_browser_name)

# Add sequences to the output for every input sequence with the same proxy node

proxy_collection = vtk.vtkCollection()
input_browser_node.GetAllProxyNodes(proxy_collection)
browser_logic = slicer.modules.sequencebrowser.logic()

for i in range(proxy_collection.GetNumberOfItems()):
    proxy_node = proxy_collection.GetItemAsObject(i)
    output_sequence = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode')
    browser_logic.AddSynchronizedNode(output_sequence, proxy_node, output_browser_node)
    output_browser_node.SetRecording(output_sequence, True)

num_input_items = input_browser_node.GetNumberOfItems()

if last_index < first_index:
    logging.error("last index should not be less than first index")
if last_index > num_input_items - 1:
    logging.error("last index is too larger for the length of the input sequence")

input_browser_node.SelectFirstItem()
for i in range(first_index - 1):
    input_browser_node.SelectNextItem()

for i in range(last_index - first_index + 1):
    output_browser_node.SaveProxyNodesState()
    input_browser_node.SelectNextItem()

