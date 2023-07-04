cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(BLUELungUltrasound)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/BLUELungUltrasound")
set(EXTENSION_CATEGORY "Ultrasound")
set(EXTENSION_CONTRIBUTORS "Róbert Zsolt Szabó (Óbuda University)")
set(EXTENSION_DESCRIPTION "The purpose of this extension is to provide a real time decision support system for the BLUE ultrasound protocol.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/BLUELungUltrasound.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/BLUELungUltrasound/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(BLUELungUltrasound)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
