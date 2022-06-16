cmake_minimum_required(VERSION 2.8.9)

project(TrackingErrorInspector)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/SlicerIGT/SlicerTrackingErrorInspector")
set(EXTENSION_CATEGORY "IGT")
set(EXTENSION_CONTRIBUTORS "Vinyas Harish, Aidan Baksh, Andras Lasso (PerkLab, Queen's University)")
set(EXTENSION_DESCRIPTION "Slicer modules for evaluating pose tracking error using a ground truth pose tracker")
set(EXTENSION_ICONURL "https://github.com/SlicerIGT/SlicerTrackingErrorInspector/raw/master/TrackingErrorInspector.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/SlicerIGT/SlicerTrackingErrorInspector/raw/master/TrackingSetup.png https://github.com/SlicerIGT/SlicerTrackingErrorInspector/raw/master/CleanAndDistortedFields.jpg")

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
