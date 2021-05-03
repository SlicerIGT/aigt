cmake_minimum_required(VERSION 3.13.4)

project(DeepLearnLive)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/DeepLearnLive")
set(EXTENSION_CATEGORY "DeepLearnLive")
set(EXTENSION_CONTRIBUTORS "John Doe (AnyWare Corp.)")
set(EXTENSION_DESCRIPTION "This extension contains utilities for running an image-based neural network in real-time")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/DeepLearnLive.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/DeepLearnLive/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ModelWizard)
add_subdirectory(TrainNeuralNet)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
