import os, math
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import numpy as np
import logging

#
# TrackingErrorMapping
#

class TrackingErrorMapping(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    parent.title = "Tracking Error Mapping"
    parent.categories = ["Tracking Error Inspector"]
    parent.dependencies = []
    parent.contributors = ["Andras Lasso, Vinyas Harish, Aidan Baksh (PerkLab, Queen's)"]
    parent.helpText = "This is a simple example of using two trackers for mapping of the position tracking error. One of the trackers is used as ground truth and the other one is compared to that."
    parent.acknowledgementText = "This work was was funded by Cancer Care Ontario, the Ontario Consortium for Adaptive Interventions in Radiation Oncology (OCAIRO) \
    the Queen's University Internships in Computing (QUIC), and the Summer Work Experience Program (SWEP) at Queen's Unversity."

#
# TrackingErrorMappingWidget
#

class TrackingErrorMappingWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.logic = TrackingErrorMappingLogic()

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...


    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    # ground truth transform selector
    self.groundTruthTransformSelectorLabel = qt.QLabel()
    self.groundTruthTransformSelectorLabel.setText( "Ground truth transform: " )
    self.groundTruthTransformSelector = slicer.qMRMLNodeComboBox()
    self.groundTruthTransformSelector.nodeTypes = ( ["vtkMRMLTransformNode"] )
    self.groundTruthTransformSelector.noneEnabled = False
    self.groundTruthTransformSelector.addEnabled = True
    self.groundTruthTransformSelector.removeEnabled = True
    self.groundTruthTransformSelector.setMRMLScene( slicer.mrmlScene )
    self.groundTruthTransformSelector.setToolTip( "Pick the input ground truth transform (e.g., optical tracker)" )
    parametersFormLayout.addRow(self.groundTruthTransformSelectorLabel, self.groundTruthTransformSelector)

    # mapped transform selector
    self.mappedTransformSelectorLabel = qt.QLabel("Mapped transform: ")
    self.mappedTransformSelector = slicer.qMRMLNodeComboBox()
    self.mappedTransformSelector.nodeTypes = ( ["vtkMRMLTransformNode"] )
    self.mappedTransformSelector.noneEnabled = False
    self.mappedTransformSelector.addEnabled = True
    self.mappedTransformSelector.removeEnabled = True
    self.mappedTransformSelector.setMRMLScene( slicer.mrmlScene )
    self.mappedTransformSelector.setToolTip( "Pick the input transform to be mapped compared to the ground truth (e.g., electromagnetic tracker)" )
    parametersFormLayout.addRow(self.mappedTransformSelectorLabel, self.mappedTransformSelector)

    # output volume selector
    self.outputVisitedPointsModelSelectorLabel = qt.QLabel("Output visited points model: ")
    self.outputVisitedPointsModelSelector = slicer.qMRMLNodeComboBox()
    self.outputVisitedPointsModelSelector.nodeTypes = ( ["vtkMRMLModelNode"] )
    self.outputVisitedPointsModelSelector.noneEnabled = True
    self.outputVisitedPointsModelSelector.addEnabled = True
    self.outputVisitedPointsModelSelector.removeEnabled = True
    self.outputVisitedPointsModelSelector.baseName = "VisitedPoints"
    self.outputVisitedPointsModelSelector.setMRMLScene( slicer.mrmlScene )
    self.outputVisitedPointsModelSelector.setToolTip( "A glyph is added to the model at each measurement point. Optional." )
    parametersFormLayout.addRow(self.outputVisitedPointsModelSelectorLabel, self.outputVisitedPointsModelSelector)

    # Position error vector
    self.positionErrorTransformSelectorLabel = qt.QLabel("Position error transform: ")
    self.positionErrorTransformSelector = slicer.qMRMLNodeComboBox()
    self.positionErrorTransformSelector.nodeTypes = ( ["vtkMRMLTransformNode"] )
    self.positionErrorTransformSelector.noneEnabled = True
    self.positionErrorTransformSelector.addEnabled = True
    self.positionErrorTransformSelector.removeEnabled = True
    self.positionErrorTransformSelector.baseName = "PositionErrorTransform"
    self.positionErrorTransformSelector.setMRMLScene( slicer.mrmlScene )
    self.positionErrorTransformSelector.setToolTip( "The transform node will store and interpolate the measurement points to generate a vector field of the position error of the mapped transform compared to the ground truth transform. Optional." )
    parametersFormLayout.addRow(self.positionErrorTransformSelectorLabel, self.positionErrorTransformSelector)

    # Orientation error magnitude
    self.orientationErrorTransformSelectorLabel = qt.QLabel("Orientation error transform:")
    self.orientationErrorTransformSelector = slicer.qMRMLNodeComboBox()
    self.orientationErrorTransformSelector.nodeTypes = ( ["vtkMRMLTransformNode"] )
    self.orientationErrorTransformSelector.noneEnabled = True
    self.orientationErrorTransformSelector.addEnabled = True
    self.orientationErrorTransformSelector.removeEnabled = True
    self.orientationErrorTransformSelector.baseName = "OrientationErrorTransform"
    self.orientationErrorTransformSelector.setMRMLScene( slicer.mrmlScene )
    self.orientationErrorTransformSelector.setToolTip( "The transform node will store and interpolate the measurement points to generate a vector field of the orientation error of the mapped transform compared to the ground truth transform. Only the x component is used. Optional." )
    parametersFormLayout.addRow(self.orientationErrorTransformSelectorLabel, self.orientationErrorTransformSelector)

    # Select defaults (to make debugging easier)
    emPointerToEmTrackerNode = slicer.util.getNode('EmPointerToEmTracker')
    if emPointerToEmTrackerNode:
        self.mappedTransformSelector.setCurrentNode(emPointerToEmTrackerNode)
    emPointerGtruthToOpPointerNode = slicer.util.getNode('EmPointerGtruthToOpPointer')
    if emPointerGtruthToOpPointerNode:
        self.groundTruthTransformSelector.setCurrentNode(emPointerGtruthToOpPointerNode)
    visitedPointsModelNode = slicer.util.getNode('VisitedPoints')
    if visitedPointsModelNode:
        self.outputVisitedPointsModelSelector.setCurrentNode(visitedPointsModelNode)
    positionErrorTransformNode = slicer.util.getNode('PositionErrorTransform')
    if positionErrorTransformNode:
        self.positionErrorTransformSelector.setCurrentNode(positionErrorTransformNode)
    orientationErrorTransformNode = slicer.util.getNode('OrientationErrorTransform')
    if orientationErrorTransformNode:
        self.orientationErrorTransformSelector.setCurrentNode(orientationErrorTransformNode)

    #
    # Check box to enable creating output transforms automatically.
    # The function is useful for testing and initial creation of the transforms but not recommended when the
    # transforms are already in the scene.
    #
    self.enableTransformMappingCheckBox = qt.QCheckBox()
    self.enableTransformMappingCheckBox.checked = 0
    self.enableTransformMappingCheckBox.setToolTip("If checked, then the mapped transform difference compared to the ground truth is written into the volume.")
    parametersFormLayout.addRow("Enable mapping", self.enableTransformMappingCheckBox)
    self.enableTransformMappingCheckBox.connect('stateChanged(int)', self.setEnableTransformMapping)

    #
    # View current error area
    #

    errorCollapsibleButton = ctk.ctkCollapsibleButton()
    errorCollapsibleButton.text = "Current Error"
    self.layout.addWidget(errorCollapsibleButton)
    errorFormLayout = qt.QFormLayout(errorCollapsibleButton)

    self.errorPositionLabel = qt.QLabel("Position error (mm): ")
    self.errorPositionValueLabel = qt.QLabel()

    self.errorOrientationLabel = qt.QLabel("Orientation error (deg): ")
    self.errorOrientationValueLabel = qt.QLabel()

    self.errorXLabel = qt.QLabel("Error in X-value (mm): ")
    self.errorXValueLabel = qt.QLabel()

    self.errorYLabel = qt.QLabel("Error in Y-value (mm): ")
    self.errorYValueLabel = qt.QLabel()

    self.errorZLabel = qt.QLabel("Error in Z-value (mm): ")
    self.errorZValueLabel = qt.QLabel()

    errorFormLayout.addRow(self.errorPositionLabel, self.errorPositionValueLabel)
    errorFormLayout.addRow(self.errorOrientationLabel, self.errorOrientationValueLabel)
    errorFormLayout.addRow(self.errorXLabel, self.errorXValueLabel)
    errorFormLayout.addRow(self.errorYLabel, self.errorYValueLabel)
    errorFormLayout.addRow(self.errorZLabel, self.errorZValueLabel)

    #
    # Export Area
    #
    exportCollapsibleButton = ctk.ctkCollapsibleButton()
    exportCollapsibleButton.text = "Export"
    self.layout.addWidget(exportCollapsibleButton)
    exportFormLayout = qt.QFormLayout(exportCollapsibleButton)

    # ROI selector
    self.exportRoiSelectorLabel = qt.QLabel()
    self.exportRoiSelectorLabel.setText( "Region of interest: " )
    self.exportRoiSelector = slicer.qMRMLNodeComboBox()
    self.exportRoiSelector.nodeTypes = ( "vtkMRMLAnnotationROINode", "" )
    self.exportRoiSelector.noneEnabled = False
    self.exportRoiSelector.addEnabled = False
    self.exportRoiSelector.removeEnabled = True
    self.exportRoiSelector.setMRMLScene( slicer.mrmlScene )
    self.exportRoiSelector.setToolTip( "Pick the input region of interest for export" )
    exportFormLayout.addRow(self.exportRoiSelectorLabel, self.exportRoiSelector)

    # Export button
    self.exportButton = qt.QPushButton("Export")
    self.exportButton.toolTip = "Export the transform in the selected region of interest to a vector volume"
    self.exportButton.enabled = True
    exportFormLayout.addRow(self.exportButton)
    self.exportButton.connect('clicked(bool)', self.onExport)

    # Add vertical spacer
    self.layout.addStretch(1)

  def cleanup(self):
    pass

  def setEnableTransformMapping(self, enable):
    if enable:
      self.logic.positionErrorMappingWidget = self # it is not nice to pass GUI widget to the logic, need to use a callback instead
      self.logic.startTransformMapping(self.groundTruthTransformSelector.currentNode(), self.mappedTransformSelector.currentNode(), self.outputVisitedPointsModelSelector.currentNode(), self.positionErrorTransformSelector.currentNode(), self.orientationErrorTransformSelector.currentNode())
    else:
      self.logic.stopTransformMapping()

  def onExport(self, clicked):
    self.logic.exportTransformToVectorVolume(self.positionErrorTransformSelector.currentNode(), self.exportRoiSelector.currentNode(), True)
    self.logic.exportTransformToVectorVolume(self.orientationErrorTransformSelector.currentNode(), self.exportRoiSelector.currentNode(), True)

#
# TrackingErrorMappingLogic
#

class TrackingErrorMappingLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent = None):
    ScriptedLoadableModuleLogic.__init__(self, parent)

    self.groundTruthTransformNode = None
    self.transformNodeObserverTags = []
    self.mappedTransformNode = None
    self.outputVisitedPointsModelNode = None
    self.previousGroundTruthPosition_World = [0,0,0]

    # no sample is collected if the pointer moves less than this distance
    # from the previous sample
    self.minimumSamplingDistance = 20

    # spacing of the exported volume
    self.exportVolumeSpacingMm = 3.0

    # history of error values for error statistics
    self.positionErrorMagnitudeList = []
    self.orientationErrorMagnitudeList = []

  def addObservers(self):
    transformModifiedEvent = 15000
    transformNode = self.groundTruthTransformNode
    while transformNode:
      print "Add observer to {0}".format(transformNode.GetName())
      self.transformNodeObserverTags.append([transformNode, transformNode.AddObserver(transformModifiedEvent, self.onGroundTruthTransformNodeModified)])
      transformNode = transformNode.GetParentTransformNode()

  def removeObservers(self):
    print "Remove observers"
    for nodeTagPair in self.transformNodeObserverTags:
      nodeTagPair[0].RemoveObserver(nodeTagPair[1])

  def initializeErrorTransform(self, errorTransformNode):
    if not errorTransformNode:
      return
    alwaysClearOutputTransformOnStart = True
    errorTransform=errorTransformNode.GetTransformToParentAs('vtkThinPlateSplineTransform', False)
    if alwaysClearOutputTransformOnStart or not errorTransform:
      errorTransform=vtk.vtkThinPlateSplineTransform()
      groundTruthPoints=vtk.vtkPoints()
      mappedPoints=vtk.vtkPoints()
      errorTransform.SetSourceLandmarks(groundTruthPoints)
      errorTransform.SetTargetLandmarks(mappedPoints)
      errorTransformNode.SetAndObserveTransformToParent(errorTransform)
      # We need to use R basis function to be able to save the transform
      # VTK does not set the basis for the inverse transform (probably a bug)
      # so we set that manually.
      errorTransformNode.GetTransformToParent().SetBasisToR()
      errorTransformNode.GetTransformFromParent().SetBasisToR()

  def startTransformMapping(self, groundTruthTransformNode, mappedTransformNode, outputVisitedPointsModelNode, positionErrorTransformNode, orientationErrorTransformNode):
    self.removeObservers()
    self.groundTruthTransformNode = groundTruthTransformNode
    self.mappedTransformNode = mappedTransformNode
    self.outputVisitedPointsModelNode = outputVisitedPointsModelNode
    self.positionErrorTransformNode = positionErrorTransformNode
    self.orientationErrorTransformNode = orientationErrorTransformNode

    self.positionErrorMagnitudeList = []
    self.orientationErrorMagnitudeList = []

    if self.outputVisitedPointsModelNode:

      if not self.outputVisitedPointsModelNode.GetDisplayNode():
        modelDisplay = slicer.vtkMRMLModelDisplayNode()
        #modelDisplay.SetSliceIntersectionVisibility(False) # Show in slice view
        #modelDisplay.SetEdgeVisibility(True) # Hide in 3D view
        modelDisplay.SetEdgeVisibility(True)
        slicer.mrmlScene.AddNode(modelDisplay)
        self.outputVisitedPointsModelNode.SetAndObserveDisplayNodeID(modelDisplay.GetID())

      self.visitedPoints = vtk.vtkPoints()
      self.visitedPointsPolydata = vtk.vtkPolyData()
      self.visitedPointsPolydata.SetPoints(self.visitedPoints)
      glyph = vtk.vtkPolyData()
      cubeSource = vtk.vtkCubeSource()
      cubeSource.SetXLength(self.minimumSamplingDistance)
      cubeSource.SetYLength(self.minimumSamplingDistance)
      cubeSource.SetZLength(self.minimumSamplingDistance)
      self.visitedPointsGlyph3d = vtk.vtkGlyph3D()
      self.visitedPointsGlyph3d.SetSourceConnection(cubeSource.GetOutputPort())
      self.visitedPointsGlyph3d.SetInputData(self.visitedPointsPolydata)
      self.visitedPointsGlyph3d.Update()
      self.outputVisitedPointsModelNode.SetPolyDataConnection(self.visitedPointsGlyph3d.GetOutputPort())

    self.initializeErrorTransform(self.positionErrorTransformNode)
    self.initializeErrorTransform(self.orientationErrorTransformNode)

    # Start the updates
    self.addObservers()
    self.onGroundTruthTransformNodeModified(0,0)

  def stopTransformMapping(self):
    self.removeObservers()
    self.updateErrorSummaryDisplay()

  def updateErrorDisplay(self, positionErrorMagnitude, orientationErrorMagnitude, positionErrorX, positionErrorY, positionErrorZ):
    # TODO: it would be nicer to call a GUI callback function instead of manipulating GUI widgets in the logic

    self.positionErrorMappingWidget.errorPositionValueLabel.setText("{0:.3f}".format(positionErrorMagnitude))
    self.positionErrorMappingWidget.errorXValueLabel.setText("{0:.3f}".format(positionErrorX))
    self.positionErrorMappingWidget.errorYValueLabel.setText("{0:.3f}".format(positionErrorY))
    self.positionErrorMappingWidget.errorZValueLabel.setText("{0:.3f}".format(positionErrorZ))

    if orientationErrorMagnitude>=0:
      self.positionErrorMappingWidget.errorOrientationValueLabel.setText("{0:.3f}".format(orientationErrorMagnitude))
    else:
      self.positionErrorMappingWidget.errorOrientationValueLabel.setText("N/A")

  def updateErrorSummaryDisplay(self):
    # TODO: it would be nicer to call a GUI callback function instead of manipulating GUI widgets in the logic

    self.positionErrorMappingWidget.errorPositionValueLabel.setText("{0:.3f} (MAD)".format(np.mean(self.positionErrorMagnitudeList)))
    self.positionErrorMappingWidget.errorXValueLabel.setText("N/A")
    self.positionErrorMappingWidget.errorYValueLabel.setText("N/A")
    self.positionErrorMappingWidget.errorZValueLabel.setText("N/A")

    self.positionErrorMappingWidget.errorOrientationValueLabel.setText("{0:.3f} (MAD)".format(np.mean(self.orientationErrorMagnitudeList)))

    print("positionError = "+repr(self.positionErrorMagnitudeList)+";")
    print("orientationError = "+repr(self.orientationErrorMagnitudeList)+";")

    percentiles = [5, 25, 50, 75, 95]
    positionPercentiles = self.percentile(self.positionErrorMagnitudeList, percentiles)
    for percIndex in range(len(percentiles)):
        print("PositionError {0}th percentile: {1:.3f}".format(percentiles[percIndex], positionPercentiles[percIndex]))
    orientationPercentiles = self.percentile(self.orientationErrorMagnitudeList, percentiles)
    for percIndex in range(len(percentiles)):
        print("OrientationError {0}th percentile: {1:.3f}".format(percentiles[percIndex], orientationPercentiles[percIndex]))

  def percentile(self, a, percentiles):
    result = []
    aSorted = np.sort(a)
    for perc in percentiles:
      result.append(aSorted[int(round(0.01*perc*len(aSorted)))])
    return result

  def onGroundTruthTransformNodeModified(self, observer, eventid):

    mappedToWorldTransform = vtk.vtkMatrix4x4()
    self.mappedTransformNode.GetMatrixTransformToWorld(mappedToWorldTransform)
    mappedPos = [mappedToWorldTransform.GetElement(0,3), mappedToWorldTransform.GetElement(1,3), mappedToWorldTransform.GetElement(2,3)]
    groundTruthToWorldTransform = vtk.vtkMatrix4x4()
    self.groundTruthTransformNode.GetMatrixTransformToWorld(groundTruthToWorldTransform)
    worldToMappedTransform = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(mappedToWorldTransform, worldToMappedTransform)
    groundTruthToMappedTransform = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(worldToMappedTransform, groundTruthToWorldTransform, groundTruthToMappedTransform)

    positionErrorX = groundTruthToMappedTransform.GetElement(0,3)
    positionErrorY = groundTruthToMappedTransform.GetElement(1,3)
    positionErrorZ = groundTruthToMappedTransform.GetElement(2,3)
    positionErrorMagnitude = math.sqrt(positionErrorX*positionErrorX+positionErrorY*positionErrorY+positionErrorZ*positionErrorZ)

    orientationErrorMagnitude = -1 # if negative it means the value is invalid
    try:
      angle, direc, point = self.rotation_from_matrix(self.orthonormalize(self.arrayFromVtkMatrix(groundTruthToMappedTransform)))
      orientationErrorMagnitude = abs(angle) * 180/math.pi
    except ValueError:
      logging.warning("Failed to compute orientation error")

    self.positionErrorMagnitudeList.append(positionErrorMagnitude)
    self.orientationErrorMagnitudeList.append(orientationErrorMagnitude)

    self.updateErrorDisplay(positionErrorMagnitude, orientationErrorMagnitude, positionErrorX, positionErrorY, positionErrorZ)

    groundTruthPosition_World = [groundTruthToWorldTransform.GetElement(0,3), groundTruthToWorldTransform.GetElement(1,3), groundTruthToWorldTransform.GetElement(2,3)]
    mappedPosition_World = [mappedToWorldTransform.GetElement(0,3), mappedToWorldTransform.GetElement(1,3), mappedToWorldTransform.GetElement(2,3)]

    # return if did not move enough compared to the previous sampling position
    if vtk.vtkMath.Distance2BetweenPoints(self.previousGroundTruthPosition_World,groundTruthPosition_World) < self.minimumSamplingDistance*self.minimumSamplingDistance:
      return

    self.previousGroundTruthPosition_World = groundTruthPosition_World

    # Add a box at the visited point position
    if self.outputVisitedPointsModelNode:
      self.visitedPoints.InsertNextPoint(groundTruthPosition_World)
      self.visitedPoints.Modified()

    # Update transforms
    if self.positionErrorTransformNode:
      positionErrorVectorTransform=self.positionErrorTransformNode.GetTransformToParent()
      positionErrorVectorTransform.GetSourceLandmarks().InsertNextPoint(groundTruthPosition_World)
      positionErrorVectorTransform.GetTargetLandmarks().InsertNextPoint(mappedPosition_World)
      self.positionErrorTransformNode.GetTransformToParent().Modified()
    if self.orientationErrorTransformNode:
      orientationErrorMagnitudeTransform=self.orientationErrorTransformNode.GetTransformToParent()
      orientationErrorMagnitudeTransform.GetSourceLandmarks().InsertNextPoint(groundTruthPosition_World)
      orientationErrorMagnitudeTransform.GetTargetLandmarks().InsertNextPoint(groundTruthPosition_World[0]+orientationErrorMagnitude, groundTruthPosition_World[1], groundTruthPosition_World[2])
      self.orientationErrorTransformNode.GetTransformToParent().Modified()

  def exportTransformToVectorVolume(self, positionErrorVectorTransform, exportRoi, magnitudeOnly):
    if not positionErrorVectorTransform:
      return

    roiBounds_Ras = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    exportRoi.GetRASBounds(roiBounds_Ras)
    exportVolumeSize = [(roiBounds_Ras[1]-roiBounds_Ras[0]+1)/self.exportVolumeSpacingMm, (roiBounds_Ras[3]-roiBounds_Ras[2]+1)/self.exportVolumeSpacingMm, (roiBounds_Ras[5]-roiBounds_Ras[4]+1)/self.exportVolumeSpacingMm]
    exportVolumeSize = [int(math.ceil(x)) for x in exportVolumeSize]

    exportImageData = vtk.vtkImageData()
    exportImageData.SetExtent(0, exportVolumeSize[0]-1, 0, exportVolumeSize[1]-1, 0, exportVolumeSize[2]-1)
    if vtk.VTK_MAJOR_VERSION <= 5:
      exportImageData.SetScalarType(vtk.VTK_DOUBLE)
      exportImageData.SetNumberOfScalarComponents(1)
      exportImageData.AllocateScalars()
    else:
      exportImageData.AllocateScalars(vtk.VTK_DOUBLE, 1)

    exportVolume = slicer.vtkMRMLVectorVolumeNode()
    exportVolume.SetAndObserveImageData(exportImageData)
    exportVolume.SetSpacing(self.exportVolumeSpacingMm, self.exportVolumeSpacingMm, self.exportVolumeSpacingMm)
    exportVolume.SetOrigin(roiBounds_Ras[0], roiBounds_Ras[2], roiBounds_Ras[4])

    slicer.modules.transforms.logic().CreateDisplacementVolumeFromTransform(positionErrorVectorTransform, exportVolume, magnitudeOnly)

  # TODO: it is not nice that this function is duplicated in DualModalityCalibration, call the DualModalityCalibration.logic from here instead
  # Source: http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
  def rotation_from_matrix(self, matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    import numpy
    R = numpy.array(matrix, dtype=numpy.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, W = numpy.linalg.eig(R33.T)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = numpy.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, Q = numpy.linalg.eig(R)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = numpy.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (numpy.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point

  # TODO: it is not nice that this function is duplicated in DualModalityCalibration, call the DualModalityCalibration.logic from here instead
  def arrayFromVtkMatrix(self, vtkMatrix):
    npArray = np.zeros((4,4))
    for row in range(4):
      for column in range(4):
          npArray[row][column] = vtkMatrix.GetElement(row,column)
    return npArray

  # TODO: it is not nice that this function is duplicated in DualModalityCalibration, call the DualModalityCalibration.logic from here instead
  def orthonormalize(self, a):
    a_ortho = np.identity(4)

    aRot = a[0:3,0:3]
    u, s, v = np.linalg.svd(aRot, full_matrices=1, compute_uv=1)
    aRot_ortho = np.dot(u,v) # v is transpose of the usual v (for example v in Matlab)
    a_ortho[0:3,0:3] = aRot_ortho

    a_ortho[0:3,3] = a[0:3,3]
    return a_ortho

class TrackingErrorMappingTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_TrackingErrorMapping1()

  def test_TrackingErrorMapping1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = TrackingErrorMappingLogic()
    self.assertTrue( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
