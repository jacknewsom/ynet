import torch
import numpy as np

def purity(pred, label):
    '''
    Calculate clustering purity from predicted fragment labels and true fragment labels
    
    args:
        pred: (N,) tensor where entries are between 0, p_max
        label: (N,) tensor where entries are between 0, l_max
    
    return:
        purity: average clustering purity
    '''
    purities = []
    for p in np.unique(pred):
        # find predictions corresponding to particular label
        fragment_predictions = pred[pred == p]
        # actual fragment label
        fragment_truth = label[pred == p]
        biggest = max(np.unique(fragment_truth))
        purity = len(fragment_truth[fragment_truth == biggest]) / len(fragment_truth)
        purities.append(purity)
    return sum(purities) / len(purities)

def efficiency(pred, label):
    '''
    Calculate clustering efficiency from predicted fragment labels and true fragment labels
    
    args:
        pred: (N,) tensor where entries are between 0, p_max
        label: (N,) tensor where entries are between 0, l_max
    
    return:
        efficiency: average clustering efficiency
    '''
    efficiencies = []
    for l in np.unique(label):
        # actual fragment label
        fragment_truth = label[label == l]
        # predicted labels for each voxel in this cluster
        fragment_pred = pred[label == l]
        biggest = max(np.unique(fragment_pred))
        efficiency = len(fragment_pred[fragment_pred == biggest]) / len(fragment_pred)
        efficiencies.append(efficiency)
    return sum(efficiencies) / len(efficiencies)