"""
Unit tests for ENG2006 coursework 2
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def question1a(points,pointLabels):
    if 'points' in locals():
        assert type(points) == np.ndarray, 'points is not a numpy array'
        assert points.shape == (4000,2),'points array is not of the expected size'
    else:
        raise Exception('points array not defined')
        
    if 'pointLabels' in locals():
        assert type(pointLabels) == np.ndarray, 'pointLabels is not a numpy array'
        assert pointLabels.shape == (4000,),'pointLabels array is not of the expected size'
    else:
        raise Exception('pointLabels array not defined')
        
    print('points and labels seem to be defined properly')
    

def question1b(pointsTrain,pointsVal,pointsTest,pointLabelsTrain,pointLabelsVal,pointLabelsTest):
    if 'pointsTrain' in locals():
        assert type(pointsTrain) == np.ndarray, 'pointsTrain is not a numpy array'
        assert pointsTrain.shape == (1400,2),'pointsTrain array is not of the expected size'
    else:
        raise Exception('pointsTrain array not defined')
        
    if 'pointsVal' in locals():
        assert type(pointsVal) == np.ndarray, 'pointsVal is not a numpy array'
        assert pointsVal.shape == (600,2),'pointsVal array is not of the expected size'
    else:
        raise Exception('pointsVal array not defined')
        
    if 'pointsTest' in locals():
        assert type(pointsTest) == np.ndarray, 'pointsTest is not a numpy array'
        assert pointsTest.shape == (2000,2),'pointsTest array is not of the expected size'
    else:
        raise Exception('pointsTest array not defined')
        
    if 'pointLabelsTrain' in locals():
        assert type(pointLabelsTrain) == np.ndarray, 'pointLabelsTrain is not a numpy array'
        assert pointLabelsTrain.shape == (1400,),'pointLabelsTrain array is not of the expected size'
    else:
        raise Exception('pointLabelsTrain array not defined')
        
    if 'pointLabelsVal' in locals():
        assert type(pointLabelsVal) == np.ndarray, 'pointLabelsVal is not a numpy array'
        assert pointLabelsVal.shape == (600,),'pointLabelsVal array is not of the expected size'
    else:
        raise Exception('pointLabelsVal array not defined')
        
    if 'pointLabelsTest' in locals():
        assert type(pointLabelsTest) == np.ndarray, 'pointLabelsTest is not a numpy array'
        assert pointLabelsTest.shape == (2000,),'pointLabelsTest array is not of the expected size'
    else:
        raise Exception('pointLabelsTest array not defined')
        
    print('Training validation and test sets seem to be defined properly')
    

def question1c(layersOpt,unitsOpt,lossOpt,accOpt,modelOpt,pointsVal,pointLabelsVal):
    if 'layersOpt' in locals():
        assert type(layersOpt) == int, 'layersOpt is not an integer'
    else:
        raise Exception('layersOpt not defined')
        
    if 'unitsOpt' in locals():
        assert type(unitsOpt) == int, 'unitsOpt is not an integer'
    else:
        raise Exception('unitsOpt not defined')
    
    if 'lossOpt' in locals():
        assert type(lossOpt) == float, 'lossOpt is not a float'
    else:
        raise Exception('lossOpt not defined')
        
    if 'accOpt' in locals():
        assert type(accOpt) == float, 'accOpt is not a float'
    else:
        raise Exception('accOpt not defined')
        
    if 'modelOpt' in locals():
        assert type(modelOpt) == tf.keras.Sequential, 'modelOpt is not a keras model'
        assert len(modelOpt.layers)==layersOpt+1,'The number of hidden layers in the model is not the same as variable unitsOpt'
        assert modelOpt.layers[0].units==unitsOpt,'The number of units in the model is not the same as variable unitsOpt'
        
        assert len(modelOpt.layers) in [2,3,5], 'The number of hidden layers in the model is outside the specified values'
        assert modelOpt.layers[0].units in [64,128,256,512], 'The number of hidden units in the model is outside the specified values'
        
        lossVal, accVal = modelOpt.evaluate(pointsVal,  pointLabelsVal)
        
        assert lossVal==lossOpt and accVal==accOpt, 'The accuracy of the model is different than what is stored in accOpt'
        
        assert accVal>0.8, 'The accuracy of your model seems very low. Make sure you have trained it properly.'
    else:
        raise Exception('modelOpt array not defined')
        
    print('The keras model seems to be defined properly!')
    
def question1d(accTest,accOpt,lossTest,pointsConfusionMatrix,pointsConfusionMatrixPlot):
    if 'accTest' in locals():
        assert type(accTest) == float, 'accTest is not an float'    
        assert accTest>accOpt-0.04, 'Your test accuracy seems significantly smaller than your validation accuracy. Make sure your model is trained properly.'
    else:
        raise Exception('accTest not defined')
        
    if 'lossTest' in locals():
        assert type(lossTest) == float, 'lossTest is not an float'
    else:
        raise Exception('lossTest not defined')
        
    if 'pointsConfusionMatrix' in locals():
        assert type(pointsConfusionMatrix) == np.ndarray, 'pointsConfusionMatrix is of the proper type'
        assert pointsConfusionMatrix.shape == (3,3),'pointsConfusionMatrix is not of the expected size'
    else:
        raise Exception('pointsConfusionMatrix not defined')
        
    if 'pointsConfusionMatrixPlot' in locals():
        assert type(pointsConfusionMatrixPlot) == ConfusionMatrixDisplay, 'pointsConfusionMatrixPlot is of the proper type'
    else:
        raise Exception('pointsConfusionMatrixPlot not defined')
        
    print('The test set accuracy and confusion matrix seem to be defined properly.')
    
def question2a(imagesTrain,imageLabelsTrain,imagesTest,imageLabelsTest):
    if 'imagesTrain' in locals():
        assert type(imagesTrain) == np.ndarray, 'imagesTrain is not a numpy array'
        assert imagesTrain.shape == (6633, 150, 150),'imagesTrain array is not of the expected size'
    else:
        raise Exception('imagesTrain array not defined')
        
    if 'imageLabelsTrain' in locals():
        assert type(imageLabelsTrain) == np.ndarray, 'imageLabelsTrain is not a numpy array'
        assert imageLabelsTrain.shape == (6633,),'imageLabelsTrain array is not of the expected size'
    else:
        raise Exception('imageLabelsTrain array not defined')
        
    if 'imagesTest' in locals():
        assert type(imagesTest) == np.ndarray, 'imagesTest is not a numpy array'
        assert imagesTest.shape == (715, 150, 150),'imagesTest array is not of the expected size'
    else:
        raise Exception('imagesTest array not defined')
        
    if 'imageLabelsTest' in locals():
        assert type(imageLabelsTest) == np.ndarray, 'imageLabelsTest is not a numpy array'
        assert imageLabelsTest.shape == (715,),'imageLabelsTest array is not of the expected size'
    else:
        raise Exception('imageLabelsTest array not defined')
        
    print('Images and labels seem to be defined properly.')
    
def question2b(imagesTest,imageLabelsTest,imagesTrain,imageLabelsTrain,imagesVal,imageLabelsVal):
    if 'imagesTest' in locals():
        assert type(imagesTest) == np.ndarray, 'imagesTest is not a numpy array'
        assert imagesTest.shape == (715, 150, 150),'imagesTest array is not of the expected size'
        assert imagesTest.max()<=1 and imagesTest.min()>=0, 'imagesTest is not normalised'
    else:
        raise Exception('imagesTest array not defined')
        
    if 'imageLabelsTest' in locals():
        assert type(imageLabelsTest) == np.ndarray, 'imageLabelsTest is not a numpy array'
        assert imageLabelsTest.shape == (715,),'imageLabelsTest array is not of the expected size'
    else:
        raise Exception('imageLabelsTest array not defined')
    
    if 'imagesTrain' in locals():
        assert type(imagesTrain) == np.ndarray, 'imagesTrain is not a numpy array'
        assert imagesTrain.shape == (4974, 150, 150),'imagesTrain array is not of the expected size'
        assert imagesTrain.max()<=1 and imagesTrain.min()>=0, 'imagesTrain is not normalised'
    else:
        raise Exception('imagesTrain array not defined')
        
    if 'imageLabelsTrain' in locals():
        assert type(imageLabelsTrain) == np.ndarray, 'imageLabelsTrain is not a numpy array'
        assert imageLabelsTrain.shape == (4974,),'imageLabelsTrain array is not of the expected size'
    else:
        raise Exception('imageLabelsTrain array not defined')
    
    
    if 'imagesVal' in locals():
        assert type(imagesVal) == np.ndarray, 'imagesVal is not a numpy array'
        assert imagesVal.shape == (1659, 150, 150),'imagesVal array is not of the expected size'
        assert imagesVal.max()<=1 and imagesVal.min()>=0, 'imagesVal is not normalised'
    else:
        raise Exception('imagesVal array not defined')
        
    if 'imageLabelsVal' in locals():
        assert type(imageLabelsVal) == np.ndarray, 'imageLabelsVal is not a numpy array'
        assert imageLabelsVal.shape == (1659,),'imageLabelsVal array is not of the expected size'
    else:
        raise Exception('imageLabelsVal array not defined')
        
    print('Images labels and training/validation sets seem to be defined properly.')
    
    
def question2c(imageModelMLPOpt,imageMLPLossTest,imageMLPAccTest,imagesTest,imageLabelsTest):
    imageModelMLPOpt.summary()

    if 'imageMLPLossTest' in locals():
        assert type(imageMLPLossTest) == float, 'imageMLPLossTest is not a float'
    else:
        raise Exception('imageMLPLossTest not defined')
        
    if 'imageMLPAccTest' in locals():
        assert type(imageMLPAccTest) == float, 'imageMLPAccTest is not a float'
    else:
        raise Exception('imageMLPAccTest not defined')
        
    if 'imageModelMLPOpt' in locals():
        assert type(imageModelMLPOpt) == tf.keras.Sequential, 'imageModelMLPOpt is not a keras model'
        
        assert len(imageModelMLPOpt.layers) in [4,6,10], 'The number of hidden layers in the model is outside the specified values'
        
        imageLossTestRef, imageAccTestRef = imageModelMLPOpt.evaluate(imagesTest,  imageLabelsTest)
        
        assert imageMLPAccTest==imageAccTestRef, 'The value stored in imageAccTest seems to be different than the accuracy of your model for the test set.'
        
        assert imageAccTestRef>0.70, 'The accuracy of your model seems very low. Make sure you have trained it properly.'
    else:
        raise Exception('imageModelMLPOpt not properly defined')
        
    print('The keras model seems to be defined and trained properly!')



def question2d(imageModelCNN,imageCNNLossTest,imageCNNAccTest,imagesTest,imageLabelsTest):
    imageModelCNN.summary()

    if 'imageCNNLossTest' in locals():
        assert type(imageCNNLossTest) == float, 'imageCNNLossTest is not a float'
    else:
        raise Exception('imageCNNLossTest not defined')
        
    if 'imageCNNAccTest' in locals():
        assert type(imageCNNAccTest) == float, 'imageCNNAccTest is not a float'
    else:
        raise Exception('imageCNNAccTest not defined')
        
    if 'imageModelCNN' in locals():
        assert type(imageModelCNN) == tf.keras.Sequential, 'imageModelCNN is not a keras model'
        
    
        assert type(imageModelCNN.layers[0])==tf.keras.layers.Conv2D, 'The first layer of your network should be convolutional'
        assert type(imageModelCNN.layers[1])==tf.keras.layers.MaxPooling2D, 'The second layer of your network should be max pooling'
        assert type(imageModelCNN.layers[2])==tf.keras.layers.Conv2D, 'The third layer of your network should be convolutional'
        assert type(imageModelCNN.layers[3])==tf.keras.layers.MaxPooling2D, 'The fourth layer of your network should be max pooling'
        
        imageLossTestRef, imageAccTestRef = imageModelCNN.evaluate(imagesTest,  imageLabelsTest)
        
        assert imageCNNAccTest==imageAccTestRef, 'The value stored in imageCNNAccTest seems to be different than the accuracy of your model for the test set.'
        
        assert imageAccTestRef>0.95, 'The accuracy of your model seems very low. Make sure you have trained it properly.'
    else:
        raise Exception('imageModelCNN not properly defined')
        
    print('The keras model seems to be defined and trained properly!')