# featurise.py
import numpy as np

def featuriseArray(data):
    header = data[0]  #keeps the header
    numericalData = np.array(data[1:], dtype=float)
    
    #normalise
    normalisedData = (numericalData - numericalData.min(axis=0)) / \
        (numericalData.max(axis=0) - numericalData.min(axis=0))
    
    #combines the header onto the rest of the data
    featurised_data = np.vstack([header, normalisedData])
    return featurised_data
