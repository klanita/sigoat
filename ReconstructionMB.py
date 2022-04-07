#!/usr/bin/env python

import os
import math
import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix, csc_matrix, vstack
from scipy.sparse.linalg import lsqr
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

def createModelMatrix(geometry='multisegmentCup.mat', idx=None, resolutionXY=256):
    #===== define paths =====#
    main_dir    = os.getcwd()
    array_dir   = f'/home/anna/OptoAcoustics/arrayInfo/{geometry}'

    #===== load transducer positions =====#
    arrayFile = sio.loadmat(array_dir)
    transducerPos = arrayFile['transducerPos']
    if not idx is None:
        transducerPos = transducerPos[idx, :]
        
    speedOfSound    = 1525    
    reconDimsXY     = 0.0256                             # recon dimensions x and y [m]
    xSensor         = np.transpose(transducerPos[:,0])   # positions in x
    ySensor         = np.transpose(transducerPos[:,1])   # positions in y
    rSensor         = np.sqrt(xSensor**2 + ySensor**2)
    fSampling       = 40e6
    delayInSamples  = 61                                 # DAQ delay
    nSamples        = 2030;                              # number of samples
    timePoints      = np.arange(
        0, (nSamples)/fSampling, 1/fSampling) + delayInSamples/fSampling
    nAngles         = 2*resolutionXY
    theta           = np.arctan2(ySensor,xSensor)
    angleSensor     = theta + 2*math.pi*(np.multiply((xSensor>0),(ySensor<0)))
    lambdaReg       = 15e6
    iterationNum    = 10

    modelMatrix = calculateModelMatrix(speedOfSound, resolutionXY, reconDimsXY,
        timePoints, rSensor, angleSensor, nAngles)

    return modelMatrix


def cropMatrix(matrixIn, xSensor, ySensor, reconDimsXY, fSampling, speedOfSound, delayInSamples):
    
    r1      = ySensor[119]
    r2      = np.sqrt(xSensor[0]**2+ySensor[0]**2)
    
    limits  = [r1-(reconDimsXY)/np.sqrt(2), r2+(reconDimsXY)/np.sqrt(2)]
    
    # extract delay
    conversionFactor    = fSampling/speedOfSound
    limitsInSamples     = [np.floor(limits[0]*conversionFactor - delayInSamples), 
    np.ceil(limits[1]*conversionFactor - delayInSamples)]
    
    limitsInSamples[0]  = max(0, int(limitsInSamples[0])-1)
    limitsInSamples[1]  = min(nSamples, int(limitsInSamples[1]))
    
    matrixOut           = matrixIn[limitsInSamples[0]:limitsInSamples[1]]

    return matrixOut


def calculateProjection(xPoint,yPoint,rPoint,theta,reconDimsXY,resolutionXY,nRows,i):
    
    nCols               = resolutionXY*resolutionXY
    lt                  = np.shape(xPoint)[1]                 # length of the time vector
    nAngles             = np.shape(xPoint)[0]                 # number of points on the curve
    pixelSize           = reconDimsXY/(resolutionXY-1)        # sampling distance in x and y
    
    ##### map points to original grid #####
    xPointUnrotated     = xPoint*np.cos(theta) - yPoint*np.sin(theta); # horizontal position of the points of the curve in the original grid (not rotated)
    yPointUnrotated     = xPoint*np.sin(theta) + yPoint*np.cos(theta); # vertical position of the points of the curve in the original grid (not rotated)
    
    # pad zeros to x and y positions
    xPaddedRight    = np.concatenate((xPointUnrotated, np.zeros((1,lt))))
    yPaddedRight    = np.concatenate((yPointUnrotated, np.zeros((1,lt))))
    xPaddedLeft     = np.concatenate((np.zeros((1,lt)), xPointUnrotated))
    yPaddedLeft     = np.concatenate((np.zeros((1,lt)), yPointUnrotated))
    
    distToOrigin    = np.sqrt(np.power((xPaddedRight - xPaddedLeft),2) + np.power((yPaddedRight - yPaddedLeft),2))
    distToOrigin    = distToOrigin[1:nAngles,:]
    del xPaddedRight, xPaddedLeft, yPaddedRight, yPaddedLeft
    
    vecIntegral = (1/2)*np.divide((np.concatenate((distToOrigin, np.zeros((1,lt))))+np.concatenate((np.zeros((1,lt)), distToOrigin))),rPoint) # vector for calculating the integral
    del distToOrigin
    
    xNorm = (xPointUnrotated + (reconDimsXY/2))/pixelSize+1 # horizontal position of the points of the curve in normalized coordinates
    del xPointUnrotated
    
    yNorm = (yPointUnrotated + (reconDimsXY/2))/pixelSize+1 # vertical position of the points of the curve in normalized coordinates
    del yPointUnrotated
    
    xBefore = np.floor(xNorm)   # horizontal position of the point of the grid at the left of the point (normalized coordinates)
    xAfter  = np.floor(xNorm+1) # horizontal position of the point of the grid at the right of the point (normalized coordinates)
    xDiff   = xNorm-xBefore
    del xNorm

    yBefore = np.floor(yNorm)      # vertical position of the point of the grid below of the point (normalized coordinates)
    yAfter  = np.floor(yNorm+1)    # vertical position of the point of the grid above of the point (normalized coordinates)
    yDiff   = yNorm - yBefore
    del yNorm
    
    ##### define square #####

    # position of the first point of the square
    xSquare1 = xBefore.astype(int)
    ySquare1 = yBefore.astype(int)
    
    # position of the second point of the square
    xSquare2 = xAfter.astype(int)
    ySquare2 = yBefore.astype(int)
    
    # position of the third point of the square
    xSquare3 = xBefore.astype(int)
    ySquare3 = yAfter.astype(int)
    
    # position of the fourth point of the square
    xSquare4 = xAfter.astype(int)
    ySquare4 = yAfter.astype(int)
    
    ##### decide points are inside or outside of the rectangle #####
    inPoint1 = (xSquare1>0) & (xSquare1<=resolutionXY) & (ySquare1>0) & (ySquare1<=resolutionXY) # boolean to decide 1. point of the square is inside the grid
    inPoint2 = (xSquare2>0) & (xSquare2<=resolutionXY) & (ySquare2>0) & (ySquare2<=resolutionXY) # boolean to decide 2. point of the square is inside the grid
    inPoint3 = (xSquare3>0) & (xSquare3<=resolutionXY) & (ySquare3>0) & (ySquare3<=resolutionXY) # boolean to decide 3. point of the square is inside the grid
    inPoint4 = (xSquare4>0) & (xSquare4<=resolutionXY) & (ySquare4>0) & (ySquare4<=resolutionXY) # boolean to decide 4. point of the square is inside the grid

    inVec1 = np.transpose(inPoint1).reshape(1, -1) # convert to vector
    inVec2 = np.transpose(inPoint2).reshape(1, -1) # convert to vector
    inVec3 = np.transpose(inPoint3).reshape(1, -1) # convert to vector
    inVec4 = np.transpose(inPoint4).reshape(1, -1) # convert to vector
    
    ##### define points on grid #####
    pos1 = resolutionXY*(xSquare1-1)+ySquare1 # one dimensional position of the first points of the squares in the grid
    pos2 = resolutionXY*(xSquare2-1)+ySquare2 # one dimensional position of the first points of the squares in the grid
    pos3 = resolutionXY*(xSquare3-1)+ySquare3 # one dimensional position of the first points of the squares in the grid
    pos4 = resolutionXY*(xSquare4-1)+ySquare4 # one dimensional position of the first points of the squares in the grid
    del xSquare1, xSquare2, xSquare3, xSquare4, ySquare1, ySquare2, ySquare3, ySquare4
    
    ##### convert to vector format #####
    posVec1 = np.transpose(pos1).reshape(1, -1) # Pos_triang_1_t in vector form
    posVec2 = np.transpose(pos2).reshape(1, -1) # Pos_triang_1_t in vector form
    posVec3 = np.transpose(pos3).reshape(1, -1) # Pos_triang_1_t in vector form
    posVec4 = np.transpose(pos4).reshape(1, -1) # Pos_triang_1_t in vector form
    del pos1, pos2, pos3, pos4
    
    weight1 = (1-xDiff)*(1-yDiff)*vecIntegral # weight of the first point of the triangle
    weight2 = (xDiff)*(1-yDiff)*vecIntegral # weight of the second point of the triangle
    weight3 = (1-xDiff)*(yDiff)*vecIntegral # weight of the third point of the triangle
    weight4 = (xDiff)*(yDiff)*vecIntegral # weight of the fourth point of the triangle
    
    weightVec1 = np.transpose(weight1).reshape(1, -1) # weight_sq_1 in vector form
    weightVec2 = np.transpose(weight2).reshape(1, -1) # weight_sq_1 in vector form
    weightVec3 = np.transpose(weight3).reshape(1, -1) # weight_sq_1 in vector form
    weightVec4 = np.transpose(weight4).reshape(1, -1) # weight_sq_1 in vector form
    del weight1, weight2, weight3, weight4
    
    rowMatrix       = np.transpose( np.transpose(np.expand_dims(np.linspace(1,lt,lt, dtype=int),axis=0)) * np.ones((1,nAngles), dtype=int) ) # rows of the sparse matrix
    rowMatrixVec    = np.transpose(np.transpose(rowMatrix).reshape(-1, 1))-1 # rows of the sparse matrix in vector form
    del rowMatrix

    rowMat              = np.concatenate((rowMatrixVec[inVec1]+(i*lt), rowMatrixVec[inVec2]+(i*lt),
        rowMatrixVec[inVec3]+(i*lt), rowMatrixVec[inVec4]+(i*lt)))

    posMat              = np.concatenate((posVec1[inVec1]-1, posVec2[inVec2]-1, posVec3[inVec3]-1,
        posVec4[inVec4]-1))

    weightMat           = np.concatenate((weightVec1[inVec1], weightVec2[inVec2], weightVec3[inVec3],
        weightVec4[inVec4]))

    projectionMatrix    = csc_matrix((weightMat, (rowMat, posMat)), shape=(nRows, nCols))
    
    return projectionMatrix


# Function to calculate model matrix
def calculateModelMatrix(speedOfSound, resolutionXY, reconDimsXY, timePoints, rSensor, angleSensor, nAngles):
    
    nCols       = resolutionXY*resolutionXY                 # number of columns of the matrix
    nRows       = len(timePoints)*len(angleSensor)            # number of rows of the matrix
    pixelSize   = reconDimsXY/(resolutionXY-1)              # one pixel size
    dt          = 1e-15                                     # diferential of time to perform derivation
    tPlusdt     = timePoints+dt                             # time instants for t+dt
    tMinusdt    = timePoints-dt                             # time instants for t-dt
    
    # max angle required to cover all grid for each of the transducers
    angleMax = np.arcsin(((reconDimsXY+2*pixelSize)*np.sqrt(2))/(2*np.amin(rSensor)))
    
    minusDistSensor     = speedOfSound*tMinusdt
    plusDistSensor      = speedOfSound*tPlusdt

    angles              = np.transpose(np.expand_dims(np.linspace(-angleMax,angleMax,nAngles),axis=0))*np.ones((1,len(timePoints)))
    
    pbar = tqdm(range(len(angleSensor)), ncols=80)

    for i in pbar:
        
        # print('Projection Number: {}'.format(i+1))
        
        theta               = angleSensor[i]                        # angle to (0,0) point
        
        rMinus              = np.ones((nAngles,1))*minusDistSensor  # -t distance from sensor to curve
        rPlus               = np.ones((nAngles,1))*plusDistSensor   # +t distance from sensor to curve
        
        xMinust             = rSensor[i]-(rMinus)*np.cos(angles)   # x distance at -t based on (0,0) to transducer coordinate system
        yMinust             = (rMinus)*np.sin(angles)              # y distance at +t based on (0,0) to transducer coordinate system
        
        xPlust              = rSensor[i]-(rPlus)*np.cos(angles)    # x distance at +t based on (0,0) to transducer coordinate system
        yPlust              = (rPlus)*np.sin(angles)               # y distance at +t based on (0,0) to transducer coordinate system

        projectionMinust    = calculateProjection(xMinust, yMinust, rMinus, theta, reconDimsXY, resolutionXY, nRows, i)
        projectionPlust     = calculateProjection(xPlust, yPlust, rPlus, theta, reconDimsXY, resolutionXY, nRows, i)

        if i > 0:
            modelMatrix     = modelMatrix + (1/(2*dt))*(projectionPlust - projectionMinust)
        else:
            modelMatrix     = (1/(2*dt))*(projectionPlust - projectionMinust)
    
    # clear variables
    del xMinust, yMinust, rMinus, xPlust, yPlust, rPlus
    
    return modelMatrix



def sigMatFilter(sigMat, lowCutOff, highCutOff, fSampling, fOrder):
    
    nyquistRatio    = 0.5 * fSampling
    lowCutOff       = lowCutOff / nyquistRatio
    highCutOff      = highCutOff / nyquistRatio
    lowF, highF     = butter(fOrder, [lowCutOff, highCutOff], btype='band')
    
    for i in range(np.shape(sigMat)[2]):
        for j in range(np.shape(sigMat)[1]):
            sigMat[:,j,i] = lfilter(lowF, highF, sigMat[:,j,i])

    return sigMat


def sigMatNormalize(sigMatIn):

    sigMatOut = np.zeros(np.shape(sigMatIn))

    for i in range(np.shape(sigMatIn)[2]):
        singleF             = sigMatIn[:,:,i]
        meanF               = np.mean(singleF, axis=0)
        sigMatOut[:,:,i]    = singleF - np.tile(meanF, (np.shape(singleF)[0], 1))

    return sigMatOut


def sigMatSmooth(sigMatIn):
    
    sigMatOut = np.zeros(np.shape(sigMatIn))

    # weighting for linear par
    xLinear             = 64.5
    posLinear           = np.linspace(0,127,128,dtype=int)
    sigmaLinear         = 54
    weightingLinear     = np.exp(-(np.abs(posLinear - xLinear))**10 / (2*sigmaLinear**10))
    
    # weighting for concave parts
    xConcave            = 32.5
    posConcave          = np.linspace(0,63,64,dtype=int)
    sigmaConcave        = 27
    weightingConcave    = np.exp(-(np.abs(posConcave - xConcave))**10/(2*sigmaConcave**10))

    for i in range(np.shape(sigMat)[0]):
        sigMatOut[i,64:192,:] = np.multiply(np.expand_dims(weightingLinear,axis=1),sigMatIn[i,64:192,:])
        sigMatOut[i,0:64,:]   = np.multiply(np.expand_dims(weightingConcave,axis=1),sigMatIn[i,0:64,:])
        sigMatOut[i,192:256,:]= np.multiply(np.expand_dims(weightingConcave,axis=1),sigMatIn[i,192:256,:])

    return sigMat


def calculateRegularizationMatrix(resolutionXY, lambdaReg):
    
    nRows       = resolutionXY*resolutionXY
    nCols       = resolutionXY*resolutionXY
    rows        = np.linspace(0,nRows-1, nRows)
    cols        = np.linspace(0,nCols-1, nCols)
    
    matrixVal   = np.ones((nRows,))*lambdaReg
    regMatrix   = csc_matrix((matrixVal, (rows, cols)), shape=(nRows, nCols))
    
    return regMatrix


def sigMatRecon(sigMat, modelMatrix, regMatrix, iterationNum, resolutionXY):

    AMat        = vstack((modelMatrix, regMatrix))
    
    sigMatVec   = np.expand_dims(np.transpose(sigMat).reshape(-1),axis=1)

    bVec        = np.concatenate((sigMatVec, np.zeros((resolutionXY*resolutionXY, 1)) ))
    
    recon, reasonTerm, iterNum, normR = lsqr(AMat, bVec, iter_lim=iterationNum)[:4]

    imageRecon  = np.reshape(recon, (resolutionXY, resolutionXY))

    return imageRecon
