import h5py
import os
import numpy as np

def extract_signals(filename, signalName):
    '''
    _
    :param filename: .mat data 
    :param signalName: Raw RF data from Ultrasound 
    :return: data in shape (number of samples, channel number, time series length, frame)
    '''
    fData = h5py.File(filename, 'r')
    inData = fData.get(signalName)
    print('loading training data %s in shape: (%d, %d, %d)' % (signalName, inData.shape[0], inData.shape[1], inData.shape[2]))
    data = np.array(inData).reshape(inData.shape[0], inData.shape[1], inData.shape[2], 1)

    return data


def extract_signals_all(filenames, signalName):
    '''
    :param filenames: list of .mat data file paths
    :param signalName: Raw RF data from Ultrasound 
    :return: data in shape (number of samples, channel number, time series length, frame)
    '''
    all_data = []

    for filename in filenames:
        fData = h5py.File(filename, 'r')
        inData = fData.get(signalName)
        print('loading training data %s from file %s in shape: (%d, %d, %d)' % (signalName, filename, inData.shape[0], inData.shape[1], inData.shape[2]))
        data = np.array(inData).reshape(inData.shape[0], inData.shape[1], inData.shape[2], 1)
        all_data.append(data)

    # 合并所有数据
    combined_data = np.concatenate(all_data, axis=0)
    return combined_data


def extract_signals_rf(filename, signalName):
    '''
    _
    :param filename: .mat data 
    :param signalName: Raw RF data from Ultrasound 
    :return: data in shape (number of samples, channel number, time series length, frame)
    '''
    fData = h5py.File(filename, 'r')
    inData = fData.get(signalName)
    print('loading training data %s in shape: (%d, %d, %d)' % (signalName, inData.shape[0], inData.shape[1], inData.shape[2]))    
    inData = np.transpose(inData, (2, 1, 0))

    data = np.array(inData).reshape(inData.shape[0], inData.shape[1], inData.shape[2], 1)

    return data

def extract_signals_sos(filename, signalName):
    '''
    
    :param filename: .mat data 
    :param signalName: Raw RF data from Ultrasound 
    :return: data in shape (number of samples, channel number, time series length, frame)
    '''
    fData = h5py.File(filename, 'r')
    inData = fData.get(signalName)
    print('loading training data %s in shape: (%d, %d, %d)' % (signalName, inData.shape[0], inData.shape[1], inData.shape[2]))
    inData = np.transpose(inData, (2, 0, 1))
    data = np.array(inData).reshape(inData.shape[0], inData.shape[1], inData.shape[2], 1)


    return data


import os
import numpy as np
from scipy.io import loadmat

def extract_signals_scipy(filename, signalName):
    '''
    
    :param filename: .mat data 
    :param signalName: Raw RF data from Ultrasound 
    :return: data in shape (number of samples, channel number, time series length, frame)
    '''
    mat_data = loadmat(filename)
    inData = mat_data[signalName]
    print('loading training data %s in shape: (%d, %d, %d)' % (signalName, inData.shape[0], inData.shape[1], inData.shape[2]))
    data = np.array(inData).reshape(inData.shape[0], inData.shape[1], inData.shape[2], 1)

    return data