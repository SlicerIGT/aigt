cmake_minimum_required(VERSION 3.13.4)

project(HerniaAnnotationExtension)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Tamas Ungi (Perk Lab), Jacob Laframboise (Perk Lab)")
set(EXTENSION_DESCRIPTION "This extensions annotates open surgical tools in a hernia phantom. ")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/CarsExtension.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/CarsExtension/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(HerniaAnnotationModule)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
