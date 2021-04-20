cmake_minimum_required(VERSION 3.5)

project(SlicerAIGT)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/SlicerIGT/aigt")
set(EXTENSION_CATEGORY "IGT")
set(EXTENSION_CONTRIBUTORS "Tamas Ungi (Queen's University), Rebecca Hisey (Queen's University), Kyle Sunderland (Queen's University)")
set(EXTENSION_DESCRIPTION "This extension contains modules to assist in deep learning for guided medical interventions. Included modules can collect data, train networks, and deploy models for real-time image classification." )
set(EXTENSION_ICONURL "https://github.com/SlicerIGT/aigt/raw/master/SlicerAIGTLogo.png" )
set(EXTENSION_SCREENSHOTURLS "https://github.com/SlicerIGT/aigt/raw/master/Screenshots/UltrasoundSegmentation.png" )

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DeepLearnLive)
add_subdirectory(SequenceLabeller)
add_subdirectory(SlicerExtension/UsAnnotationExport/SingleSliceSegmentation)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
