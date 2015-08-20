import os, math, numpy
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

#
# CompareDisplacementFields
#

class CompareDisplacementFields(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    parent.title = "Compare displacement fields"
    parent.categories = ["IGT"]
    parent.dependencies = []
    parent.contributors = ["Andras Lasso (PerkLab, Queen's)"]
    parent.helpText = "Compare displacement fields"
    parent.acknowledgementText = ""

#
# CompareDisplacementFieldsWidget
#

class CompareDisplacementFieldsWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.logic = CompareDisplacementFieldsLogic()

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Common Area
    #
    commonCollapsibleButton = ctk.ctkCollapsibleButton()
    commonCollapsibleButton.text = "Common"
    self.layout.addWidget(commonCollapsibleButton)
    commonFormLayout = qt.QFormLayout(commonCollapsibleButton)
    
    # ROI selector
    self.roiSelectorLabel = qt.QLabel()
    self.roiSelectorLabel.setText( "Region of interest: " )
    self.exportRoiSelector = slicer.qMRMLNodeComboBox()
    self.exportRoiSelector.nodeTypes = ( "vtkMRMLAnnotationROINode", "" )
    self.exportRoiSelector.noneEnabled = False
    self.exportRoiSelector.addEnabled = False
    self.exportRoiSelector.removeEnabled = True
    self.exportRoiSelector.setMRMLScene( slicer.mrmlScene )
    self.exportRoiSelector.setToolTip( "Pick the input region of interest for comparison" )
    commonFormLayout.addRow(self.roiSelectorLabel, self.exportRoiSelector)

    # Spacing
    self.compareVolumeSpacingLabel = qt.QLabel()
    self.compareVolumeSpacingLabel.setText( "Comparison volume spacing: " )
    self.compareVolumeSpacing = ctk.ctkDoubleSpinBox()
    self.compareVolumeSpacing.minimum = 0.01
    self.compareVolumeSpacing.maximum = 30
    self.compareVolumeSpacing.suffix = 'mm'
    self.compareVolumeSpacing.singleStep = 0.5
    self.compareVolumeSpacing.value = 3
    self.compareVolumeSpacing.setToolTip( "Resolution for comparison. Smaller values mean higher accuracy but more computation time." )
    commonFormLayout.addRow(self.compareVolumeSpacingLabel, self.compareVolumeSpacing)

    #
    # Average Area
    #
    averageCollapsibleButton = ctk.ctkCollapsibleButton()
    averageCollapsibleButton.text = "Average"
    self.layout.addWidget(averageCollapsibleButton)
    averageFormLayout = qt.QFormLayout(averageCollapsibleButton)

    # ground truth field selector
    self.maxNumberOfInputFieldsForAveraging = 10
    self.averageInputFieldSelectors = []
    for averageInputFieldIndex in xrange(0, self.maxNumberOfInputFieldsForAveraging):
      averageInputFieldSelectorLabel = qt.QLabel()
      averageInputFieldSelectorLabel.setText( 'Field {0}: '.format(averageInputFieldIndex+1) )
      averageInputFieldSelector = slicer.qMRMLNodeComboBox()
      averageInputFieldSelector.nodeTypes = ( "vtkMRMLVectorVolumeNode", "" )
      averageInputFieldSelector.noneEnabled = True
      averageInputFieldSelector.addEnabled = False
      averageInputFieldSelector.removeEnabled = True
      averageInputFieldSelector.setMRMLScene( slicer.mrmlScene )
      averageInputFieldSelector.setToolTip( "Pick the field that will be include in the average computation" )
      averageFormLayout.addRow(averageInputFieldSelectorLabel, averageInputFieldSelector)
      self.averageInputFieldSelectors.append(averageInputFieldSelector)

    self.averageOutputFieldLabel = qt.QLabel()
    self.averageOutputFieldLabel.setText( 'Output field mean:')
    self.averageOutputFieldSelector = slicer.qMRMLNodeComboBox()
    self.averageOutputFieldSelector.nodeTypes = ( "vtkMRMLVectorVolumeNode", "" )
    self.averageOutputFieldSelector.noneEnabled = True
    self.averageOutputFieldSelector.addEnabled = True
    self.averageOutputFieldSelector.removeEnabled = True
    self.averageOutputFieldSelector.renameEnabled = True
    self.averageOutputFieldSelector.baseName = 'Mean'
    self.averageOutputFieldSelector.setMRMLScene( slicer.mrmlScene )
    self.averageOutputFieldSelector.setToolTip( "Computed mean of the displacement fields" )
    averageFormLayout.addRow(self.averageOutputFieldLabel, self.averageOutputFieldSelector)

    self.varianceOutputFieldLabel = qt.QLabel()
    self.varianceOutputFieldLabel.setText( 'Output field mean error:')
    self.varianceOutputFieldSelector = slicer.qMRMLNodeComboBox()
    self.varianceOutputFieldSelector.nodeTypes = ( "vtkMRMLScalarVolumeNode", "" )
    self.varianceOutputFieldSelector.noneEnabled = True
    self.varianceOutputFieldSelector.addEnabled = True
    self.varianceOutputFieldSelector.removeEnabled = True
    self.varianceOutputFieldSelector.renameEnabled = True
    self.varianceOutputFieldSelector.showChildNodeTypes = False
    self.varianceOutputFieldSelector.baseName = 'MeanError'
    self.varianceOutputFieldSelector.setMRMLScene( slicer.mrmlScene )
    self.varianceOutputFieldSelector.setToolTip( "Computed variance of the displacement fields" )
    averageFormLayout.addRow(self.varianceOutputFieldLabel, self.varianceOutputFieldSelector)

    # Compute button
    self.averageComputeButton = qt.QPushButton("Compute")
    self.averageComputeButton.toolTip = "Compute average and standard deviation"
    self.averageComputeButton.enabled = True
    averageFormLayout.addRow(self.averageComputeButton)
    self.averageComputeButton.connect('clicked(bool)', self.computeAverage)

    
    #
    # Difference Area
    #
    differenceCollapsibleButton = ctk.ctkCollapsibleButton()
    differenceCollapsibleButton.text = "Difference"
    self.layout.addWidget(differenceCollapsibleButton)
    differenceFormLayout = qt.QFormLayout(differenceCollapsibleButton)

    self.differenceInputFieldALabel = qt.QLabel()
    self.differenceInputFieldALabel.setText( 'Displacement field A')
    self.differenceInputFieldASelector = slicer.qMRMLNodeComboBox()
    self.differenceInputFieldASelector.nodeTypes = ( "vtkMRMLVectorVolumeNode", "" )
    self.differenceInputFieldASelector.noneEnabled = False
    self.differenceInputFieldASelector.addEnabled = False
    self.differenceInputFieldASelector.removeEnabled = True
    self.differenceInputFieldASelector.setMRMLScene( slicer.mrmlScene )
    self.differenceInputFieldASelector.setToolTip( "Pick the field that the other will be subtracted from" )
    differenceFormLayout.addRow(self.differenceInputFieldALabel, self.differenceInputFieldASelector)

    self.differenceInputFieldBLabel = qt.QLabel()
    self.differenceInputFieldBLabel.setText( 'Displacement field B' )
    self.differenceInputFieldBSelector = slicer.qMRMLNodeComboBox()
    self.differenceInputFieldBSelector.nodeTypes = ( "vtkMRMLVectorVolumeNode", "" )
    self.differenceInputFieldBSelector.noneEnabled = False
    self.differenceInputFieldBSelector.addEnabled = False
    self.differenceInputFieldBSelector.removeEnabled = True
    self.differenceInputFieldBSelector.setMRMLScene( slicer.mrmlScene )
    self.differenceInputFieldBSelector.setToolTip( "Pick the field to subtract from the other" )
    differenceFormLayout.addRow(self.differenceInputFieldBLabel, self.differenceInputFieldBSelector)

    self.differenceOutputFieldLabel = qt.QLabel()
    self.differenceOutputFieldLabel.setText( 'Output difference:')
    self.differenceOutputFieldSelector = slicer.qMRMLNodeComboBox()
    self.differenceOutputFieldSelector.nodeTypes = ( "vtkMRMLVectorVolumeNode", "" )
    self.differenceOutputFieldSelector.noneEnabled = True
    self.differenceOutputFieldSelector.addEnabled = True
    self.differenceOutputFieldSelector.removeEnabled = True
    self.differenceOutputFieldSelector.renameEnabled = True
    self.differenceOutputFieldSelector.baseName = 'Difference'
    self.differenceOutputFieldSelector.setMRMLScene( slicer.mrmlScene )
    self.differenceOutputFieldSelector.setToolTip( "Computed difference of the displacement fields" )
    differenceFormLayout.addRow(self.differenceOutputFieldLabel, self.differenceOutputFieldSelector)
    
    # Compute button
    self.differenceComputeButton = qt.QPushButton("Compute")
    self.differenceComputeButton.toolTip = "Compute difference between fields (FieldA-FieldB)"
    self.differenceComputeButton.enabled = True
    differenceFormLayout.addRow(self.differenceComputeButton)
    self.differenceComputeButton.connect('clicked(bool)', self.computeDifference)

    # Add vertical spacer
    self.layout.addStretch(1)

  def cleanup(self):
    pass

  def computeAverage(self, clicked):

    inputFieldNodes=[]
    for fieldSelector in self.averageInputFieldSelectors:
      fieldNode = fieldSelector.currentNode()
      if fieldNode:
        inputFieldNodes.append(fieldNode)

    self.logic.ReferenceVolumeSpacingMm = self.compareVolumeSpacing.value
    self.logic.computeAverage(inputFieldNodes, self.exportRoiSelector.currentNode(), self.averageOutputFieldSelector.currentNode(), self.varianceOutputFieldSelector.currentNode())
  
  def computeDifference(self, clicked):
    self.logic.ReferenceVolumeSpacingMm = self.compareVolumeSpacing.value
    self.logic.computeDifference(self.differenceInputFieldASelector.currentNode(), self.differenceInputFieldBSelector.currentNode(), self.exportRoiSelector.currentNode(), self.differenceOutputFieldSelector.currentNode())
 

#
# CompareDisplacementFieldsLogic
#

class CompareDisplacementFieldsLogic(ScriptedLoadableModuleLogic):
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

    # spacing of the exported volume
    self.ReferenceVolumeSpacingMm = 3.0

  def createVectorVolumeFromRoi(self, exportRoi, spacingMm, numberOfComponents=3):
    roiCenter = [0, 0, 0]
    exportRoi.GetXYZ( roiCenter )
    roiRadius = [0, 0, 0]
    exportRoi.GetRadiusXYZ( roiRadius )
    roiOrigin_Roi = [roiCenter[0] - roiRadius[0], roiCenter[1] - roiRadius[1], roiCenter[2] - roiRadius[2], 1 ]
    
    roiToRas = vtk.vtkMatrix4x4()
    if exportRoi.GetTransformNodeID() != None:
      roiBoxTransformNode = slicer.mrmlScene.GetNodeByID(exportRoi.GetTransformNodeID())
      roiBoxTransformNode.GetMatrixTransformToWorld(roiToRas)
        
    exportVolumeSize = [roiRadius[0]*2/spacingMm, roiRadius[1]*2/spacingMm, roiRadius[2]*2/spacingMm]
    exportVolumeSize = [int(math.ceil(x)) for x in exportVolumeSize]
    
    exportImageData = vtk.vtkImageData()
    exportImageData.SetExtent(0, exportVolumeSize[0]-1, 0, exportVolumeSize[1]-1, 0, exportVolumeSize[2]-1)
    if vtk.VTK_MAJOR_VERSION <= 5:
      exportImageData.SetScalarType(vtk.VTK_DOUBLE)
      exportImageData.SetNumberOfScalarComponents(numberOfComponents)
      exportImageData.AllocateScalars()
    else:
      exportImageData.AllocateScalars(vtk.VTK_DOUBLE, numberOfComponents)

    exportVolume = None
    if numberOfComponents==1:
      exportVolume = slicer.vtkMRMLScalarVolumeNode()
    else:
      exportVolume = slicer.vtkMRMLVectorVolumeNode()
    exportVolume.SetAndObserveImageData(exportImageData)
    exportVolume.SetIJKToRASDirections( roiToRas.GetElement(0,0), roiToRas.GetElement(0,1), roiToRas.GetElement(0,2), roiToRas.GetElement(1,0), roiToRas.GetElement(1,1), roiToRas.GetElement(1,2), roiToRas.GetElement(2,0), roiToRas.GetElement(2,1), roiToRas.GetElement(2,2))
    exportVolume.SetSpacing(spacingMm, spacingMm, spacingMm)
    roiOrigin_Ras = roiToRas.MultiplyPoint(roiOrigin_Roi)
    exportVolume.SetOrigin(roiOrigin_Ras[0:3])
    
    return exportVolume

    
  def resampleVolume(self, inputVolume, referenceVolume):
    parameters = {}
    parameters["inputVolume"] = inputVolume.GetID()    
    parameters["referenceVolume"] = referenceVolume.GetID()
    outputVolume = slicer.vtkMRMLVectorVolumeNode()
    outputVolume.SetName('ResampledVolume')
    slicer.mrmlScene.AddNode( outputVolume )
    parameters["outputVolume"] = outputVolume.GetID()
#   Instead of
#    slicer.cli.run(slicer.modules.resamplescalarvectordwivolume, None, parameters, wait_for_completion=True)
#   apply using custom code to allow disabling DisplayData in ApplyAndWait
    module = slicer.modules.resamplescalarvectordwivolume
    node = slicer.cli.createNode(module, parameters)
    logic = module.logic()
    logic.ApplyAndWait(node, False)

    return outputVolume

  def computeDifference(self, fieldA, fieldB, roi, differenceVolume):
    referenceVolume = self.createVectorVolumeFromRoi(roi, self.ReferenceVolumeSpacingMm)
    referenceVolume.SetName('ReferenceVolume')
    slicer.mrmlScene.AddNode( referenceVolume )
    resampledFieldA = self.resampleVolume(fieldA, referenceVolume)
    resampledFieldB = self.resampleVolume(fieldB, referenceVolume)
    subtractor = vtk.vtkImageMathematics()
    subtractor.SetOperationToSubtract()
    subtractor.SetInput1Data(resampledFieldA.GetImageData())
    subtractor.SetInput2Data(resampledFieldB.GetImageData())
    differenceVolume.SetImageDataConnection(subtractor.GetOutputPort())
    ijkToRasMatrix = vtk.vtkMatrix4x4()
    referenceVolume.GetIJKToRASMatrix(ijkToRasMatrix)
    differenceVolume.SetIJKToRASMatrix(ijkToRasMatrix)
    
    differenceVolumeDisplayNode = slicer.vtkMRMLVectorVolumeDisplayNode()
    slicer.mrmlScene.AddNode( differenceVolumeDisplayNode )
    differenceVolumeDisplayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow");
    differenceVolume.SetAndObserveNthDisplayNodeID(0, differenceVolumeDisplayNode.GetID()); 
    
    slicer.mrmlScene.RemoveNode( resampledFieldA )
    slicer.mrmlScene.RemoveNode( resampledFieldB )
    slicer.mrmlScene.RemoveNode( referenceVolume )

  def computeAverage(self, inputFieldNodes, roi, outputAverageVolume, outputVarianceVolume):
  
    referenceVolume = self.createVectorVolumeFromRoi(roi, self.ReferenceVolumeSpacingMm)
    referenceVolume.SetName('ReferenceVolume')
    slicer.mrmlScene.AddNode( referenceVolume )
    
    fieldNodes=[]
    fieldImageData=[]
    for fieldNode in inputFieldNodes:
      resampledFieldNode = self.resampleVolume(fieldNode, referenceVolume)
      fieldNodes.append(resampledFieldNode)
      fieldImageData.append(resampledFieldNode.GetImageData())
 
    ijkToRasMatrix = vtk.vtkMatrix4x4()
      
    # Average volume
    averageImageData = vtk.vtkImageData()
    averageImageData.DeepCopy(referenceVolume.GetImageData())
    outputAverageVolume.SetAndObserveImageData(averageImageData)
    referenceVolume.GetIJKToRASMatrix(ijkToRasMatrix)
    outputAverageVolume.SetIJKToRASMatrix(ijkToRasMatrix)
    
    # Variance volume
    varianceImageData = vtk.vtkImageData()    
    varianceImageData.SetExtent(averageImageData.GetExtent())
    if vtk.VTK_MAJOR_VERSION <= 5:
      varianceImageData.SetScalarType(vtk.VTK_DOUBLE)
      varianceImageData.SetNumberOfScalarComponents(1)
      varianceImageData.AllocateScalars()
    else:
      varianceImageData.AllocateScalars(vtk.VTK_DOUBLE, 1)
    outputVarianceVolume.SetIJKToRASMatrix(ijkToRasMatrix)
    outputVarianceVolume.SetAndObserveImageData(varianceImageData)

    # Compute
    
    dims = averageImageData.GetDimensions()
    # [field, component]
    voxelValues = numpy.zeros([len(fieldImageData), 3])
    for z in xrange(dims[2]):
      for y in xrange(dims[1]):
        for x in xrange(dims[0]):
          fieldIndex = 0
          for imageData in fieldImageData:
            voxelValues[fieldIndex,0] = imageData.GetScalarComponentAsDouble(x, y, z, 0)
            voxelValues[fieldIndex,1] = imageData.GetScalarComponentAsDouble(x, y, z, 1)
            voxelValues[fieldIndex,2] = imageData.GetScalarComponentAsDouble(x, y, z, 2)
            fieldIndex = fieldIndex+1
          meanVoxelValues = numpy.mean(voxelValues, axis = 0)
          averageImageData.SetScalarComponentFromDouble(x, y, z, 0, meanVoxelValues[0])
          averageImageData.SetScalarComponentFromDouble(x, y, z, 1, meanVoxelValues[1])
          averageImageData.SetScalarComponentFromDouble(x, y, z, 2, meanVoxelValues[2])
          # Compute the mean of the magnitude of the error vectors
          errorValues = voxelValues-meanVoxelValues
          errorVectorMagnitudes = numpy.sqrt(numpy.sum(errorValues*errorValues, axis=1))
          varianceImageData.SetScalarComponentFromDouble(x, y, z, 0, numpy.mean(errorVectorMagnitudes))

    averageImageData.Modified()
    varianceImageData.Modified()
    
    # Create display node if they have not created yet
    
    if not outputAverageVolume.GetNthDisplayNode(0):
      outputAverageVolumeDisplayNode = slicer.vtkMRMLVectorVolumeDisplayNode()
      slicer.mrmlScene.AddNode( outputAverageVolumeDisplayNode )
      outputAverageVolumeDisplayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow");
      outputAverageVolume.SetAndObserveNthDisplayNodeID(0, outputAverageVolumeDisplayNode.GetID()); 
    
    if not outputVarianceVolume.GetNthDisplayNode(0):
      outputVarianceVolumeDisplayNode = slicer.vtkMRMLScalarVolumeDisplayNode()
      slicer.mrmlScene.AddNode( outputVarianceVolumeDisplayNode )
      outputVarianceVolumeDisplayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow");
      outputVarianceVolume.SetAndObserveNthDisplayNodeID(0, outputVarianceVolumeDisplayNode.GetID()); 

    # Clean up temporary nodes
    
    for fieldNode in fieldNodes:
      slicer.mrmlScene.RemoveNode( fieldNode )

    slicer.mrmlScene.RemoveNode( referenceVolume )


class CompareDisplacementFieldsTest(ScriptedLoadableModuleTest):
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
    self.test_CompareDisplacementFields1()

  def test_CompareDisplacementFields1(self):
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
    logic = CompareDisplacementFieldsLogic()
    self.assertTrue( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
