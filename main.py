#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:10:39 2024

@author: sspringe
"""

import numpy as np
import os
from Data_class1 import Data
from IPython.display import display
import pickle

def get_longitude_indices(center_deg, window_width_deg, discretization_step=2.5):
    """
    Calculate the range of longitude indices based on the central longitude in degrees
    and the window width in degrees, considering the discretization step.
    
    Parameters:
    - center_deg (float): Central longitude in degrees.
    - window_width_deg (float): Total width of the window in degrees.
    - discretization_step (float): Degrees per index step, default is 2.5.
    
    Returns:
    - list: Indices representing the window of longitudes.
    - int: Size of the longitude window.
    """
    total_points = int(360 / discretization_step)
    
    # Convert degrees to index
    center_idx = int(center_deg / discretization_step) % total_points
    half_window_idx = int(window_width_deg / (2 * discretization_step))
    
    # Calculate the start and end indices
    start_idx = (center_idx - half_window_idx) % total_points
    end_idx = (center_idx + half_window_idx + 1) % total_points

    # Determine the range of longitudes
    if start_idx < end_idx:
        longi = list(range(start_idx, end_idx))
    else:
        # The range wraps around the circular buffer
        longi = list(range(start_idx, total_points)) + list(range(0, end_idx))

    # Calculate the size of the window for possible further use
    window_size = len(longi)
    
    return longi, window_size

data_name = "hgh_NH_Winter_real"
data_extension = ".npy"

# Construct the path to the file
# Get the current directory of the script
current_script_path = os.path.dirname(__file__)

# Navigate to the parent directory and then to the DATA directory
data_path = os.path.join(current_script_path, '..', 'DATA', data_name + data_extension)

grid_discretization = [37,144]
scale = 360/grid_discretization[1]
lati = range(13,29+1)
filters = [ "Lati_area", "Longi_Gaussian" ]


if  ('Winter' in data_name):
    season = "Winter" 
elif ('Summer' in data_name):
    season = "Summer" 
    
if  ('model' in data_name):
    n_microstates = 2000
    # n_microstates = 360
    # n_microstates = 180
elif ('real' in data_name):
    n_microstates = 180



# Example usage
center_longitude = 0  # in degrees
window_width = 60  # window width in degrees
longi, window_size = get_longitude_indices(center_longitude, window_width)


Data_ = Data(current_script_path, data_name, grid_discretization, data_extension, scale, lati, longi, window_size, season, filters, n_microstates)
Data_.load_data()
Data_.detrend_and_deseasonalize_data()
Data_.get_blocked_days()
Data_.get_microstates()
Data_.get_perc_blocked_days_per_micro()
Data_.get_transition_matrix_and_eigen()
which_eigen = 1
Data_.get_extreme_microstates( which_eigen )
Data_.visualize_microstate_mean_Z500( which_micro = Data_.inj[0], color_local = 'royalblue')

# # Data_.append( Data_ )
# namePickle = 'Real_Winter_72_180_tau1_Rev_Req_grid_' + str(ii) + '_' + str(jj)+'_14_11_23.pkl'
# with open(namePickle, 'wb') as outp:
#     pickle.dump(Data_, outp, pickle.HIGHEST_PROTOCOL)