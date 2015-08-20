cmake_minimum_required(VERSION 2.8.9)

project(TrackingErrorInspector)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/TrackingErrorInspector")
set(EXTENSION_CATEGORY "IGT")
set(EXTENSION_CONTRIBUTORS "Vinyas Harish, Aidan Baksh, Andras Lasso (PerkLab, Queen's University)")
set(EXTENSION_DESCRIPTION "Slicer modules for evaluating pose tracking error using a ground truth pose tracker")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/TrackingErrorInspector.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/TrackingErrorInspector/Screenshots/1.png")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DualModalityCalibration)
add_subdirectory(TrackingErrorMapping)
add_subdirectory(CompareDisplacementFields)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
