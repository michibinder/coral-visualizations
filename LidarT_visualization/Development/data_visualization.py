#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:56:32 2020

@author: tennismichel
"""

import os
# from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime
# import pandas as pd
import xarray as xr
import seaborn
from scipy import signal

# fileLocation = '/Users/tennismichel/Coding/Python/DLR_LIDAR/Data'
fileLocation = '../Data'
fileName = "20191015-0014_T15Z900.nc"
path = os.path.join(fileLocation, fileName)


DS = xr.open_dataset(path, decode_times=False)
# attrs = {'units': 'seconds since 2019-10-15 00:00:00'}
# ds = xr.Dataset({'time': ('time', DS.time, attrs)})
DS.time.values = DS.time.values / 1000
DS.integration_start_time.values = DS.integration_start_time.values / 1000
DS.integration_end_time.values = DS.integration_end_time.values / 1000
DS.time.attrs['units'] = 'seconds since 2019-10-15'
DS.integration_start_time.attrs['units'] = 'seconds since 2019-10-15 00:00:00'
DS.integration_end_time.attrs['units'] = 'seconds since 2019-10-15 00:00:00'
DS = xr.decode_cf(DS, decode_coords = True, decode_times = True) 
# drop_variables = ['integration_start_time', 'integration_end_time'])

## Altitude ## 
DS['alt_plot'] = DS.altitude/1000 + DS.altitude_offset + DS.station_height #km

## Temperature ## 
# Change 0 to NaN
DS.temperature.values = np.where(DS.temperature == 0, np.nan, DS.temperature)

def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high_crit = highcut / nyq
    print(high_crit)
    b, a = signal.butter(order, high_crit, btype='lowpass')
    return b, a

## Example
n = 12  # the larger n is, the smoother curve will be
b = [1.0 / n] * n # Vector of size n filled with 1/n
a = [1]

highcut = 40 # 20km critical wavelength
fs = 1000  # 100m sample wavelength
b, a = butter_lowpass(highcut, fs, order=5)

print(b,a)

# x = DS.temperature.values + np.flip(DS.temperature.values, axis=1)


mirror_flipped = signal.lfilter(b,a,np.flip(DS.temperature.values, axis=1), axis=1) #filtfilt for both side approach
DS['tmp_bg'] = (['time', 'altitude'], np.flip(mirror_flipped, axis=1))
DS['tmp_perturbation'] = DS.temperature - DS.tmp_bg
# print(DS['tmp_bg'][50:100][510:510].values)