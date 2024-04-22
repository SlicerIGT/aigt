cmake_minimum_required(VERSION 3.5)

project(SlicerAIGT)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/SlicerIGT/aigt")
set(EXTENSION_CATEGORY "IGT")
set(EXTENSION_CONTRIBUTORS "Tamas Ungi (PerkLab, Queen's University), Rebbecca Hisey (PerkLab, Queen's University), Andras Lasso (PerkLab, Queen's University), Kyle Sunderland (PerkLab, Queen's University)")
set(EXTENSION_DESCRIPTION "This extension contains modules to assist in deep learning for guided medical interventions. Included modules can collect data, train networks, and deploy models for real-time image classification." )
set(EXTENSION_ICONURL "https://github.com/SlicerIGT/aigt/raw/master/SlicerAIGTLogo.png" )
set(EXTENSION_SCREENSHOTURLS "https://github.com/SlicerIGT/aigt/raw/master/Screenshots/UltrasoundSegmentation.png" )
set(EXTENSION_DEPENDS "SlicerOpenIGTLink SlicerIGT")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DeepLearnLive)
add_subdirectory(SequenceLabeller)
add_subdirectory(SlicerExtension/LiveUltrasoundAi/SingleSliceSegmentation)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
