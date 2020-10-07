import keras
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import util as util
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K 
from util import *
import streamlit as st

def dice_coefficient(y_true, y_pred, axis=(1, 2, 3),epsilon=0.00001):
    dice_numerator   = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + epsilon
    dice_coefficient = K.mean((dice_numerator)/(dice_denominator))
    return dice_coefficient

def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), epsilon=0.00001):
    dice_numerator   = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true**2, axis=axis) + K.sum(y_pred**2, axis=axis) + epsilon
    dice_loss        = 1 - K.mean((dice_numerator)/(dice_denominator))
    return dice_loss

def load_model():
    model_weights = 'model.hdf5'
    model_json    = 'model.json'
    with open(model_json) as json_file:
        loaded_model = model_from_json(json_file.read())
    loaded_model.load_weights(model_weights)
    return loaded_model

def load_case(image_nifty_file, label_nifty_file):
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())    
    return image, label

model = load_model()
image, label = load_case("imagesTr/BRATS_003.nii.gz", "labelsTr/BRATS_003.nii.gz")
pred = util.predict_and_viz(image, label, model, .5, loc=(130, 130, 77))                    