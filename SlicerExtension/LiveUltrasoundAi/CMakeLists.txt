cmake_minimum_required(VERSION 3.5)

project(LiveUltrasoundAi)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/LiveUltrasoundAi")
set(EXTENSION_CATEGORY "IGT")
set(EXTENSION_CONTRIBUTORS "Tamas Ungi (Queen's University)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/LiveUltrasoundAi.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/LiveUltrasoundAi/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SingleSliceSegmentation)
add_subdirectory(SingleSliceSegmentation)
add_subdirectory(SegmentationUNet)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
