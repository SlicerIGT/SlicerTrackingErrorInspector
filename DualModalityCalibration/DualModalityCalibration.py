import os
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import numpy as np
import logging
import math
import time

#
# DualModalityCalibration
#

class DualModalityCalibration(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Dual Modality Calibration "
    self.parent.categories = ["Tracking Error Inspector"]
    self.parent.dependencies = []
    self.parent.contributors = ["Vinyas Harish, Aidan Baksh, Andras Lasso (PerkLab, Queen's)"]
    self.parent.helpText = "This is a simple example of how to calibrate two tracking systems for use in mapping position tracking error."
    self.parent = parent
    self.parent.acknowledgementText = "This work was was funded by Cancer Care Ontario, the Ontario Consortium for Adaptive Interventions in Radiation Oncology (OCAIRO) \
    the Queen's University Internships in Computing (QUIC), and the Summer Work Experience Program (SWEP) at Queen's Unversity."
    #self.logic = DualModalityCalibrationLogic

#
# DualModalityCalibrationWidget
#

class DualModalityCalibrationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.logic = DualModalityCalibrationLogic()

    # Instantiate and connect widgets

    #
    # Calibration Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Calibration input and output transforms"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # Optical tracking system transform selector
    #
    self.opticalPointerSelectorLabel = qt.QLabel()
    self.opticalPointerSelectorLabel.setText( "OpPointerToOpRef Transform: " )
    self.opticalPointerSelector = slicer.qMRMLNodeComboBox()
    self.opticalPointerSelector.nodeTypes = ( ["vtkMRMLLinearTransformNode"] )
    self.opticalPointerSelector.noneEnabled = False
    self.opticalPointerSelector.addEnabled = False
    self.opticalPointerSelector.removeEnabled = True
    self.opticalPointerSelector.setMRMLScene( slicer.mrmlScene )
    self.opticalPointerSelector.setToolTip( "Pick a known transform corresponding to the optical pointer's coordinate frame" )
    parametersFormLayout.addRow(self.opticalPointerSelectorLabel, self.opticalPointerSelector)

    #
    # Em tracking system transform selector
    #
    self.EmPointerSelectorLabel = qt.QLabel()
    self.EmPointerSelectorLabel.setText( "EmPointerToEmTracker Transform: " )
    self.emPointerSelector = slicer.qMRMLNodeComboBox()
    self.emPointerSelector.nodeTypes = ( ["vtkMRMLTransformNode"] )
    self.emPointerSelector.noneEnabled = False
    self.emPointerSelector.addEnabled = False
    self.emPointerSelector.removeEnabled = True
    self.emPointerSelector.setMRMLScene( slicer.mrmlScene )
    self.emPointerSelector.setToolTip( "Pick a known transform corresponding to the Em pointer's coordinate frame" )
    parametersFormLayout.addRow(self.EmPointerSelectorLabel, self.emPointerSelector)

    #
    # Initial landmark registration result transform selector
    #
    self.initialEmTrackerToOpRefSelectorLabel = qt.QLabel()
    self.initialEmTrackerToOpRefSelectorLabel.setText( "Initial EmTrackerToOpRef Transform: " )
    self.initialEmTrackerToOpRefSelector = slicer.qMRMLNodeComboBox()
    self.initialEmTrackerToOpRefSelector.nodeTypes = ( ["vtkMRMLTransformNode"] )
    self.initialEmTrackerToOpRefSelector.noneEnabled = False
    self.initialEmTrackerToOpRefSelector.addEnabled = True
    self.initialEmTrackerToOpRefSelector.removeEnabled = True
    self.initialEmTrackerToOpRefSelector.baseName = "EmTrackerToOpRefInitial"
    self.initialEmTrackerToOpRefSelector.setMRMLScene( slicer.mrmlScene )
    self.initialEmTrackerToOpRefSelector.setToolTip( "Pick a known transform corresponding to the Em pointer's coordinate frame" )
    parametersFormLayout.addRow(self.initialEmTrackerToOpRefSelectorLabel, self.initialEmTrackerToOpRefSelector)

    self.outputEmTrackerToOpRefTransformSelectorLabel = qt.QLabel("Output EmTrackerToOpRef transform: ")
    self.outputEmTrackerToOpRefTransformSelector = slicer.qMRMLNodeComboBox()
    self.outputEmTrackerToOpRefTransformSelector.nodeTypes = ( ["vtkMRMLTransformNode"] )
    self.outputEmTrackerToOpRefTransformSelector.noneEnabled = True
    self.outputEmTrackerToOpRefTransformSelector.addEnabled = True
    self.outputEmTrackerToOpRefTransformSelector.removeEnabled = True
    self.outputEmTrackerToOpRefTransformSelector.baseName = "EmTrackerToOpRef"
    self.outputEmTrackerToOpRefTransformSelector.setMRMLScene( slicer.mrmlScene )
    self.outputEmTrackerToOpRefTransformSelector.setToolTip("Select the transform to output the EmTrackerToOpRef calibration result to")
    parametersFormLayout.addRow(self.outputEmTrackerToOpRefTransformSelectorLabel, self.outputEmTrackerToOpRefTransformSelector)

    self.outputEmPointerGtruthToOpPointerTransformSelectorLabel = qt.QLabel("Output EmPointerGtruthToOpPointer transform: ")
    self.outputEmPointerGtruthToOpPointerTransformSelector = slicer.qMRMLNodeComboBox()
    self.outputEmPointerGtruthToOpPointerTransformSelector.nodeTypes = ( ["vtkMRMLTransformNode"] )
    self.outputEmPointerGtruthToOpPointerTransformSelector.noneEnabled = True
    self.outputEmPointerGtruthToOpPointerTransformSelector.addEnabled = True
    self.outputEmPointerGtruthToOpPointerTransformSelector.removeEnabled = True
    self.outputEmPointerGtruthToOpPointerTransformSelector.baseName = "EmPointerGtruthToOpPointer"
    self.outputEmPointerGtruthToOpPointerTransformSelector.setMRMLScene( slicer.mrmlScene )
    self.outputEmPointerGtruthToOpPointerTransformSelector.setToolTip("Select the transform to output the EmPointerGtruthToOpPointer calibration result to")
    parametersFormLayout.addRow(self.outputEmPointerGtruthToOpPointerTransformSelectorLabel, self.outputEmPointerGtruthToOpPointerTransformSelector)

    # Select defaults (to make debugging easier)
    opPointerToOpRefNode = slicer.util.getNode('OpPointerToOpRef')
    if opPointerToOpRefNode:
        self.opticalPointerSelector.setCurrentNode(opPointerToOpRefNode)
    emPointerToEmTrackerNode = slicer.util.getNode('EmPointerToEmTracker')
    if emPointerToEmTrackerNode:
        self.emPointerSelector.setCurrentNode(emPointerToEmTrackerNode)
    emTrackerToOpRefInitialNode = slicer.util.getNode('EmTrackerToOpRefInitial')
    if emTrackerToOpRefInitialNode:
        self.initialEmTrackerToOpRefSelector.setCurrentNode(emTrackerToOpRefInitialNode)
    emTrackerToOpRefOutputNode = slicer.util.getNode('EmTrackerToOpRef')
    if emTrackerToOpRefOutputNode:
        self.outputEmTrackerToOpRefTransformSelector.setCurrentNode(emTrackerToOpRefOutputNode)
    emPointerGtruthToOpPointerOutputNode = slicer.util.getNode('EmPointerGtruthToOpPointer')
    if emPointerGtruthToOpPointerOutputNode:
        self.outputEmPointerGtruthToOpPointerTransformSelector.setCurrentNode(emPointerGtruthToOpPointerOutputNode)

    #
    # Controls Area
    #
    controlsCollapsibleButton = ctk.ctkCollapsibleButton()
    controlsCollapsibleButton.text = "Controls"
    self.layout.addWidget(controlsCollapsibleButton)

    # Layout within the dummy collapsible button
    controlsFormLayout = qt.QFormLayout(controlsCollapsibleButton)

    #
    # Number of data points to be collected input
    #

    self.numberDataPointsInputLabel = qt.QLabel()
    self.numberDataPointsInputLabel.setText("Number of Collected Data Points:")
    self.numberDataPointsInput = qt.QSpinBox()
    self.numberDataPointsInput.maximum = 5000
    self.numberDataPointsInput.minimum = 10
    self.numberDataPointsInputLabel.setToolTip("Select how many sample points will be used in calculations")
    controlsFormLayout.addRow(self.numberDataPointsInputLabel, self.numberDataPointsInput)
    #self.numberDataPoints = self.numberDataPointsInput.value

    #
    # Delay spinbox
    #
    self.delaySelectorLabel = qt.QLabel()
    self.delaySelectorLabel.setText("Delay (seconds):")
    self.delaySelector = qt.QSpinBox()
    self.delaySelector.minimum = 1
    self.delaySelector.maximum = 10
    self.delaySelector.value = 5
    self.delaySelector.setToolTip("Time to wait before starting data collection after clicking 'Start data collection' button")
    controlsFormLayout.addRow(self.delaySelectorLabel, self.delaySelector)

    #
    # Start data collection button
    #
    self.startButton = qt.QPushButton("Start data collection")
    self.startButton.toolTip = "Start collecting data that will be used in calculations"
    self.startButton.enabled = True
    self.startButton.connect('clicked(bool)',self.onStartDataCollection)
    controlsFormLayout.addRow(self.startButton)

    #
    # Delay countdown timer
    #
    self.delayTimer = qt.QTimer()
    self.delayTimerLabel = qt.QLabel()
    self.delayTimer.setInterval(1000) # 1 second
    self.delayTimer.setSingleShot(True)
    self.delayTimer.connect('timeout()',self.onDelayTimerTimeout)
    controlsFormLayout.addRow(self.delayTimerLabel)

    #
    # Data collection progress bar
    #
    self.collectionProgressBar = qt.QProgressBar()
    self.collectionProgressBar.setRange(0,100)
    self.collectionProgressBar.setVisible(False)
    controlsFormLayout.addRow(self.collectionProgressBar)

    #
    # Display Area
    #
    displayCollapsibleButton = ctk.ctkCollapsibleButton()
    displayCollapsibleButton.text = "Calibration results"
    self.layout.addWidget(displayCollapsibleButton)

    # Layout within the dummy collapsible button
    displayFormLayout = qt.QFormLayout(displayCollapsibleButton)

    #
    # Error metrics
    #
    self.transErrorLabel = qt.QLabel("Position error: N/A")
    self.rotErrorLabel = qt.QLabel("Orientation error: N/A")
    displayFormLayout.addRow(self.transErrorLabel)
    displayFormLayout.addRow(self.rotErrorLabel)

    # Add vertical spacer
    self.layout.addStretch(1)

  def logicStatusCallback(self, statusCode, percentageCompleted):
    """This method is called by the logic to report progress"""

    if statusCode == self.logic.CALIBRATION_COMPLETE:
        self.transErrorLabel.setText("Position error: {0:.3f} mm".format(self.logic.calibrationErrorTranslationMm))
        self.rotErrorLabel.setText("Orientation error: {0:.3f} deg".format(self.logic.calibrationErrorRotationDeg))
    else:
        self.transErrorLabel.setText("Position error: N/A")
        self.rotErrorLabel.setText("Orientation error: N/A")

    if statusCode == self.logic.CALIBRATION_COMPLETE:
        self.collectionProgressBar.setValue(100)
        self.collectionProgressBar.setVisible(False)
        self.delayTimerLabel.setText("Calibration complete.")
    elif statusCode == self.logic.CALIBRATION_IN_PROGRESS:
        self.collectionProgressBar.setVisible(True)
        self.collectionProgressBar.setValue(percentageCompleted)
        self.delayTimerLabel.setText("Collecting data...")
    else:
        self.collectionProgressBar.setValue(0)
        self.collectionProgressBar.setVisible(False)

    # Refresh the screen
    slicer.app.processEvents()

  def onStartDataCollection(self, moduleName="DualModalityCalibration"):
    self.delayDuration = self.delaySelector.value
    self.delayTimerStopTime = time.time() + float(self.delayDuration)
    self.onDelayTimerTimeout()

  def onDelayTimerTimeout(self, moduleName="DualModalityCalibration"):
    self.delayTimerLabel.setText("Time remaining before data collection begins: {0:.0f} seconds".format(self.delayTimerStopTime - time.time()))
    if (time.time()<self.delayTimerStopTime):
        # Continue
        self.delayTimer.start()
    else:
        # Delay over
        self.startDataCollection()

  def startDataCollection(self, moduleName="DualModalityCalibration"):
    logging.debug("Start Data Collection")
    self.logic.statusCallback = self.logicStatusCallback
    self.logic.startDataCollection(self.numberDataPointsInput.value, self.emPointerSelector.currentNode(),self.opticalPointerSelector.currentNode(), \
    self.initialEmTrackerToOpRefSelector.currentNode(),self.outputEmTrackerToOpRefTransformSelector.currentNode() ,self.outputEmPointerGtruthToOpPointerTransformSelector.currentNode())

  def cleanup(self):
    logging.debug("DualModalityCalibration cleanup")
    self.logic.removeObservers()

#
# DualModalityCalibrationLogic
#

class DualModalityCalibrationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):

    # Observed transform nodes
    self.opPointerToOpRefNode = None
    self.emPointerToEmTrackerNode = None
    self.transformNodeObserverTags = []

    # Input parameters
    self.initialEmTrackerToOpRefTransform = None
    self.numberOfDataPointsToCollect = 20

    # Collected data
    self.numberDataPointsCollected = 0
    self.opPointerToOpRefTransformArray = np.zeros((0,4,4))
    self.emPointerToEmTrackerTransformArray = np.zeros((0,4,4))
    self.lastUpdateTimeSec = 0

    # Output
    self.emPointerToOpPointer = None

    self.calibrationErrorTranslationMm = None
    self.calibrationErrorRotationDeg = None

    self.statusCallback = None
    self.CALIBRATION_NOT_STARTED = 0
    self.CALIBRATION_IN_PROGRESS = 1
    self.CALIBRATION_COMPLETE = 2

  def addObservers(self):
      transformModifiedEvent = slicer.vtkMRMLLinearTransformNode.TransformModifiedEvent
      transformNode = self.opPointerToOpRefNode
      while transformNode:
          logging.debug("Add observer to {0}".format(transformNode.GetName()))
          self.transformNodeObserverTags.append([transformNode, transformNode.AddObserver(transformModifiedEvent, self.onOpticalTransformNodeModified)])
          transformNode = transformNode.GetParentTransformNode()

  def removeObservers(self):
      logging.debug("Removing observers")
      for nodeTagPair in self.transformNodeObserverTags:
          nodeTagPair[0].RemoveObserver(nodeTagPair[1])

  def arrayFromVtkMatrix(self, vtkMatrix):
      npArray = np.zeros((4,4))
      for row in range(4):
        for column in range(4):
            npArray[row][column] = vtkMatrix.GetElement(row,column)
      return npArray

  def vtkMatrixFromArray(self, array):
      aVTKMatrix = vtk.vtkMatrix4x4()
      for row in range(4):
          for column in range(4):
              aVTKMatrix.SetElement(row,column, array[row][column])
      return aVTKMatrix

  def reportStatus(self, statusCode, percentageCompleted):
    if self.statusCallback:
      self.statusCallback(statusCode, percentageCompleted)

  def startDataCollection(self, numberOfDataPointsToCollect, emPointerToEmTrackerNode, opPointerToOpRefNode, initialEmTrackerToOpRefTransformNode, \
  emTrackerToOpRefNode, emPointerToOpPointerNode):
      self.removeObservers()

      self.opPointerToOpRefNode = opPointerToOpRefNode
      self.emPointerToEmTrackerNode = emPointerToEmTrackerNode
      self.numberOfDataPointsToCollect = numberOfDataPointsToCollect

      self.outputEmTrackerToOpRefNode = emTrackerToOpRefNode
      self.outputEmPointerToOpPointerNode = emPointerToOpPointerNode

      initialEmTrackerToOpRefVtkMatrix = vtk.vtkMatrix4x4()
      initialEmTrackerToOpRefTransformNode.GetMatrixTransformToParent(initialEmTrackerToOpRefVtkMatrix)
      self.initialEmTrackerToOpRefTransform = self.arrayFromVtkMatrix(initialEmTrackerToOpRefVtkMatrix)
      logging.debug("initialEmTrackerToOpRefTransformNode = {0}".format(self.initialEmTrackerToOpRefTransform))

      self.numberDataPointsCollected = 0
      self.opPointerToOpRefTransformArray = np.zeros((self.numberOfDataPointsToCollect,4,4))
      self.emPointerToEmTrackerTransformArray = np.zeros((self.numberOfDataPointsToCollect,4,4))

      self.addObservers()

      self.reportStatus(self.CALIBRATION_IN_PROGRESS,0)

  #Since optical tracking is our ground truth we only want to query both transform nodes when the optical node is changing
  def onOpticalTransformNodeModified(self,observer,eventid):

      minimumTimeBetweenUpdatesSec = 0.1
      currentTimeSec = vtk.vtkTimerLog.GetUniversalTime()
      if (currentTimeSec<self.lastUpdateTimeSec+minimumTimeBetweenUpdatesSec):
        # too close to last update
        return

      self.lastUpdateTimeSec = currentTimeSec

      # Check to ensure you do not overstep the bounds of the array, if done creating arrays, start calculations
      if self.numberDataPointsCollected >= self.numberOfDataPointsToCollect:
          logging.debug("Finished data collection")
          self.removeObservers()
          self.calculateCalibrationOutput()
          return

      # Store the two transform matrices into their corresponding arrays for calculations

      opPointerToOpRefVtkMatrix = vtk.vtkMatrix4x4()
      self.opPointerToOpRefNode.GetMatrixTransformToParent(opPointerToOpRefVtkMatrix)
      self.opPointerToOpRefTransformArray[self.numberDataPointsCollected] = self.arrayFromVtkMatrix(opPointerToOpRefVtkMatrix)

      emPointerToEmTrackerVtkMatrix = vtk.vtkMatrix4x4()
      self.emPointerToEmTrackerNode.GetMatrixTransformToParent(emPointerToEmTrackerVtkMatrix)
      self.emPointerToEmTrackerTransformArray[self.numberDataPointsCollected] = self.arrayFromVtkMatrix(emPointerToEmTrackerVtkMatrix)

      self.numberDataPointsCollected = self.numberDataPointsCollected + 1

      if self.numberDataPointsCollected % 10 == 0: # report status after every 10th point to avoid too frequent screen updates
        self.reportStatus(self.CALIBRATION_IN_PROGRESS,100*self.numberDataPointsCollected/self.numberOfDataPointsToCollect)

  def orthonormalize(self, a):
      a_ortho = np.identity(4)

      aRot = a[0:3,0:3]
      u, s, v = np.linalg.svd(aRot, full_matrices=1, compute_uv=1)
      aRot_ortho = np.dot(u,v) # v is transpose of the usual v (for example v in Matlab)
      a_ortho[0:3,0:3] = aRot_ortho

      a_ortho[0:3,3] = a[0:3,3]
      return a_ortho

  def orthonormalize3x3(self, a):
    u, s, v = np.linalg.svd(a, full_matrices=1, compute_uv=1)
    a_ortho = np.dot(u,v) # v is transpose of the usual v (for example v in Matlab)
    return a_ortho

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

  def calculateCalibrationError(self, opPointerToEmPointer, opRefToEmTracker):
    # OpRefToEmTracker * OpPointerToOpRef = EmPointerToEmTracker * OpPointerToEmPointer
    # ErrorMatrix = OpRefToEmTracker * OpPointerToOpRef * inv(EmPointerToEmTracker * OpPointerToEmPointer)

    numberOfTransforms = np.shape(self.opPointerToOpRefTransformArray)[0]
    positionErrorMm = np.zeros(numberOfTransforms)
    angleErrorDeg = np.zeros(numberOfTransforms)
    for transformIndex in range(numberOfTransforms):
        opPointerToEmTrackerFromOp = np.dot(opRefToEmTracker, self.opPointerToOpRefTransformArray[transformIndex])
        opPointerToEmTrackerFromEm = np.dot(self.emPointerToEmTrackerTransformArray[transformIndex], opPointerToEmPointer)
        errorMatrix = np.dot(opPointerToEmTrackerFromOp, np.linalg.inv(opPointerToEmTrackerFromEm))
        errorMatrix = self.orthonormalize(errorMatrix)
        positionErrorMm[transformIndex] = np.linalg.norm(errorMatrix[0:3,3])
        try:
          angle, direc, point = self.rotation_from_matrix(errorMatrix)
        except ValueError:
          loggin.warning("Invalid errorMatrix: "+repr(errorMatrix))
          angle = math.pi
        angleErrorDeg[transformIndex] = abs(angle) * 180/math.pi

    return np.mean(positionErrorMm), np.mean(angleErrorDeg)

  def getOpPointerToEmPointerFromSingleSample(self, sampleIndex):
    emPointerToEmTracker = self.emPointerToEmTrackerTransformArray[sampleIndex]
    opRefToEmTracker = np.linalg.inv(self.initialEmTrackerToOpRefTransform)
    opPointerToOpRef = self.opPointerToOpRefTransformArray[sampleIndex]
    opPointerToEmPointerComputed = np.dot(np.linalg.inv(emPointerToEmTracker),np.dot(opRefToEmTracker,opPointerToOpRef))
    return opPointerToEmPointerComputed

  def transformSolveAX_B(self, Aarray, Barray):
    # Solves least squares to find the transformation X such that AX=B
    A = np.vstack(Aarray)
    B = np.vstack(Barray)

    Rb = B[:,0:3]
    Ra = A[:,0:3]
    Rx = np.linalg.lstsq(Ra,Rb)[0]
    Rx = self.orthonormalize3x3(Rx) # Maintain orthogonality

    Pb = B[:,3]
    Pa = A[:,3]
    Px = np.linalg.lstsq(Ra,Pb-Pa)[0]

    X = np.identity(4)
    X[0:3,0:3] = Rx
    X[0:3,3] = Px

    return X

  def lstsqLeft(self,A,B):
    # Solves XA=B using AX=B
    # X * A = B
    # A.T * X.T = B.T
    xT = np.linalg.lstsq(A.T,B.T)[0]
    return xT.T

  def transformSolveXA_B(self, Aarray, Barray):
    # Solves least squares to find the transformation X such that XA=B

    #Rb = B(:,1:3); Rb(4:4:end,:) = []; Rb = C2R(Rb);
    #Ra = A(:,1:3); Ra(4:4:end,:) = []; Ra = C2R(Ra);
    Rb = np.hstack(Barray[:,0:3,0:3])
    Ra = np.hstack(Aarray[:,0:3,0:3])
    #Rx = Rb/Ra;
    Rx = self.lstsqLeft(Ra, Rb)
    Rx = self.orthonormalize3x3(Rx) # Maintain orthogonality

    #Pb = reshape(B(:,4),4,[]); Pb = Pb(1:3,:);
    #Pa = reshape(A(:,4),4,[]); Pa = Pa(1:3,:);
    Pb = Barray[:,0:3,3].T # position vector columns stacked horizontally next to each other
    Pa = Aarray[:,0:3,3].T
    #Px = Pb-Rx*Pa; Px = mean(Px')';
    #Px = Pb-np.asmatrix(Rx)*np.asmatrix(Pa)
    Px = Pb-np.dot(Rx,Pa)
    Px = np.mean(Px,1)

    X = np.identity(4)
    X[0:3,0:3] = Rx
    X[0:3,3] = Px

    return X

  def multiplyTransformArrayByTransformFromLeft(self, transform, transformArray):
    numberOfTransforms = np.shape(transformArray)[0]
    output = np.zeros(np.shape(transformArray))
    for transformIndex in range(numberOfTransforms):
      output[transformIndex] = np.dot(transform, transformArray[transformIndex])
    return output

  def multiplyTransformArrayByTransformFromRight(self, transformArray, transform):
    numberOfTransforms = np.shape(transformArray)[0]
    output = np.zeros(np.shape(transformArray))
    for transformIndex in range(numberOfTransforms):
      output[transformIndex] = np.dot(transformArray[transformIndex], transform)
    return output

  def calculateCalibrationOutput(self):
    # OpRefToEmTracker * OpPointerToOpRef = EmPointerToEmTracker * OpPointerToEmPointer

    logging.info("------- DualModalityCalibration --------")

    # From initial guess:
    opRefToEmTracker = np.linalg.inv(self.initialEmTrackerToOpRefTransform)

    # Compute calibration matrices from the first sample to get a baseline (results that we would get without calibration)
    A = np.expand_dims(self.emPointerToEmTrackerTransformArray[0],0)
    b = self.multiplyTransformArrayByTransformFromLeft(opRefToEmTracker, np.expand_dims(self.opPointerToOpRefTransformArray[0],0))
    opPointerToEmPointer = self.transformSolveAX_B(A,b)
    positionOrientationError = self.calculateCalibrationError(opPointerToEmPointer, opRefToEmTracker)
    logging.info("Initial error (from first sample, without iteration): Position and orientation error: {0}".format(positionOrientationError))
    opRefToEmTrackerInitial = opRefToEmTracker
    opPointerToEmPointerInitial = opPointerToEmPointer

    for iterationIndex in range(20):

        # (OpRefToEmTracker * OpPointerToOpRef) = (EmPointerToEmTracker) * OpPointerToEmPointer
        #                  b                    =           A                       x

        A = self.emPointerToEmTrackerTransformArray
        b = self.multiplyTransformArrayByTransformFromLeft(opRefToEmTracker, self.opPointerToOpRefTransformArray)
        opPointerToEmPointer = self.transformSolveAX_B(A,b)

        positionOrientationError = self.calculateCalibrationError(opPointerToEmPointer, opRefToEmTracker)
        logging.info("Iteration: {0},   after first LSQR, before orthonormalization - Position and orientation error: {1}".format(iterationIndex, positionOrientationError))

        opPointerToEmPointer = self.orthonormalize(opPointerToEmPointer)

        positionOrientationError = self.calculateCalibrationError(opPointerToEmPointer, opRefToEmTracker)
        logging.info("Iteration: {0},   after first LSQR,  after orthonormalization - Position and orientation error: {1}".format(iterationIndex, positionOrientationError))

        ###############################################

        # OpRefToEmTracker * OpPointerToOpRef = EmPointerToEmTracker * OpPointerToEmPointer
        #    (unknown mx)  *   (mx array)     =     (mx array)       *     (known mx)
        #         x                A                                 B
        A = self.opPointerToOpRefTransformArray
        b = self.multiplyTransformArrayByTransformFromRight(self.emPointerToEmTrackerTransformArray, opPointerToEmPointer)
        opRefToEmTracker = self.transformSolveXA_B(A,b)

        positionOrientationError = self.calculateCalibrationError(opPointerToEmPointer, opRefToEmTracker)
        logging.info("Iteration: {0},  after second LSQR, before orthonormalization - Position and orientation error: {1}".format(iterationIndex, positionOrientationError))

        opRefToEmTracker = self.orthonormalize(opRefToEmTracker)

        positionOrientationError = self.calculateCalibrationError(opPointerToEmPointer, opRefToEmTracker)
        logging.info("Iteration: {0},  after second LSQR,  after orthonormalization - Position and orientation error: {1}".format(iterationIndex, positionOrientationError))

    self.calibrationErrorTranslationMm = positionOrientationError[0]
    self.calibrationErrorRotationDeg = positionOrientationError[1]

    logging.info("opRefToEmTracker (initial guess):\n  "+repr(opRefToEmTrackerInitial))
    logging.info("opPointerToEmPointer (initial computation):\n  "+repr(opPointerToEmPointerInitial))

    logging.info("opRefToEmTracker (final):\n  "+repr(opRefToEmTracker))
    logging.info("opPointerToEmPointer (final):\n  "+repr(opPointerToEmPointer))

    emTrackerToOpRef = np.linalg.inv(opRefToEmTracker)
    emPointerToOpPointer = np.linalg.inv(opPointerToEmPointer)

    emTrackerToOpRefVTKMatrix = self.vtkMatrixFromArray(emTrackerToOpRef)
    emPointerToOpPointerVTKMatrix = self.vtkMatrixFromArray(emPointerToOpPointer)

    # Save results to output nodes

    if self.outputEmTrackerToOpRefNode:
      self.outputEmTrackerToOpRefNode.SetMatrixTransformToParent(emTrackerToOpRefVTKMatrix)

    if self.outputEmPointerToOpPointerNode:
      self.outputEmPointerToOpPointerNode.SetMatrixTransformToParent(emPointerToOpPointerVTKMatrix)

    self.reportStatus(self.CALIBRATION_COMPLETE,100)

class DualModalityCalibrationTest(ScriptedLoadableModuleTest):
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
