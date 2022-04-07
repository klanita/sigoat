#!/usr/bin/env python3

import os
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def sigMatFilter(
    sigMat,
    lowCutOff=0.1e6,
    highCutOff=6e6,
    fSampling=40e6,
    fOrder=3,
    conRatio=0.5):

    sigMatF         = np.zeros(np.shape(sigMat))
    
    nyquistRatio    = conRatio * fSampling
    lowCutOff       = lowCutOff / nyquistRatio
    highCutOff      = highCutOff / nyquistRatio
    lowF, highF     = butter(fOrder, [lowCutOff, highCutOff], btype='bandpass')
    
    for i in range(np.shape(sigMat)[2]):
        for j in range(np.shape(sigMat)[1]):
            sigMatF[:,j,i] = filtfilt(lowF, highF, sigMat[:,j,i])

    return sigMatF


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


def sigMatRecon(sigMat, resolutionXY, xSensor, ySensor, speedOfSound, timePoints, fSampling, reconDimsXY):

    imageRecon = np.zeros((resolutionXY, resolutionXY))

    Dxy = reconDimsXY/(resolutionXY-1)

    x = np.linspace(((-1)*(resolutionXY/2-0.5)*Dxy),((resolutionXY/2-0.5)*Dxy),resolutionXY)
    y = np.linspace(((-1)*(resolutionXY/2-0.5)*Dxy),((resolutionXY/2-0.5)*Dxy),resolutionXY)

    meshX, meshY = np.meshgrid(x,y)

    for i in range(0,len(xSensor)):
    # for i in range(0,1):

        singleSignal    = sigMat[2:,i]
        diffSignal      = np.concatenate((singleSignal[1:]-singleSignal[0:-1], [[0]]), axis=0)
        derSignal       = np.multiply(diffSignal, np.expand_dims(timePoints, axis=1))*fSampling

        # print(np.shape(derSignal))

        distX           = meshX - xSensor[i]
        distY           = meshY - ySensor[i]
        dist            = np.sqrt(distX**2 + distY**2)

        timeSample      = np.ceil((dist*fSampling)/speedOfSound - timePoints[0]*fSampling + 1)
        timeSample      = timeSample.astype(int)

        # print(np.shape(timeSample))

        timeSample[timeSample<=0]       = 0
        timeSample[timeSample>=2030]    = 2030

        imageRecon      = imageRecon + np.squeeze(singleSignal[timeSample] - derSignal[timeSample])

    return imageRecon

def get_normal_signal(sigMat, fSampling=40e6,
    delayInSamples=61,
    nSamples=2028,
    lowCutOff=0.1e6,
    highCutOff=6e6,
    fOrder=3):
    if sigMat.ndim != 3:
        sigMat = np.expand_dims(sigMat, axis=2)

    # filter sigMat (we can skip for now, doesnt change output a lot)
    sigMatF         = (-1)*sigMatFilter(sigMat, lowCutOff, highCutOff, fSampling, fOrder, 0.5)
    
    # normalize sigMat around 0
    return sigMatNormalize(sigMatF)[:, :, 0]


def backProject(sigMat,
    geometry='multisegmentCup',
    array_dir='./arrayInfo/',
    idx=None,
    resolutionXY=256, 
    reconDimsXY=0.0256,
    speedOfSound=1525,
    fSampling=40e6,
    delayInSamples=61,
    nSamples=2028,
    lowCutOff=0.1e6,
    highCutOff=6e6,
    fOrder=3):

    """ resolutionXY: pixel numbers in x y
        reconDimsXY: recon dimensions x and y [m]
        speedOfSound: m/s
        fSampling: sampling frequency
        delayInSamples: DAQ delay
        nSamples: number of samples
    """
    
    arrayFile = sio.loadmat(f"{array_dir}/{geometry}.mat")
    transducerPos = arrayFile['transducerPos']
    if not idx is None:
        transducerPos = transducerPos[idx, :]

    xSensor = np.transpose(transducerPos[:,0])
    ySensor = np.transpose(transducerPos[:,1])
    timePoints = np.arange(0, (nSamples)/fSampling, 1/fSampling) + delayInSamples/fSampling

    if sigMat.ndim != 3:
        sigMat = np.expand_dims(sigMat, axis=2)

    # filter sigMat (we can skip for now, doesnt change output a lot)
    sigMatF         = (-1)*sigMatFilter(sigMat, lowCutOff, highCutOff, fSampling, fOrder, 0.5)
    
    # normalize sigMat around 0
    sigMatN         = sigMatNormalize(sigMatF)

    # smooth for gap between linear and concave parts (we can skip for now, doesnt change output a lot)
    # sigMatS         = sigMatSmooth(sigMatN)

    # reconstruction
    imageRecon      = sigMatRecon(sigMatN, resolutionXY, xSensor, ySensor, speedOfSound, timePoints, fSampling, reconDimsXY)

    imageRecon      = np.rot90(imageRecon,1)

    return imageRecon
