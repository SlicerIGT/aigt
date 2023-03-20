cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(PrepareSpineData Spine Data)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/PrepareSpineData")
set(EXTENSION_CATEGORY "Ultrasound")
set(EXTENSION_CONTRIBUTORS "Nicole Kitner (Queen's University), ")
set(EXTENSION_DESCRIPTION "This module helps register spine CT scans to ultrasounds and cleans up data.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/PrepareSpineData.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/PrepareSpineData/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(PrepareSpineData)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
