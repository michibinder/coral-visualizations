################################################################################
# Copyright 2023 German Aerospace Center                                       #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

import os
import sys
import glob
import shutil
import psutil
import configparser
import datetime
import cdsapi
import multiprocessing
from ecmwfapi import ECMWFDataServer

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import filter, cmaps, era5_processor, lidar_processor

"""Config"""
DATE = "2018-06-16/to/2018-06-18"
AREA = '-46/-87/-56/-77' # # North, West, South, East. Default: global

## KEYWORDS:
## https://confluence.ecmwf.int/display/UDOC/Keywords+in+MARS+and+Dissemination+requests


# retrieve,
# class=od,
# date=2018-06-16,
# expver=1,
# levelist=1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60/61/62/63/64/65/66/67/68/69/70/71/72/73/74/75/76/77/78/79/80/81/82/83/84/85/86/87/88/89/90/91/92/93/94/95/96/97/98/99/100/101/102/103/104/105/106/107/108/109/110/111/112/113/114/115/116/117/118/119/120/121/122/123/124/125/126/127/128/129/130/131/132/133/134/135/136/137,
# levtype=ml,
# param=130/131/132/135/138/152/155,
# step=0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60/61/62/63/64/65/66/67/68/69/70/71/72,
# stream=oper,
# time=00:00:00,
# type=fc,
# target="output"

def download_era5():
    """Download data from ECMWF Tape archive"""
    # c = cdsapi.Client(quiet=True)
    c = cdsapi.Client()
    # server = ECMWFDataServer()

    datadir = "./data/eulag"
    file_ml = os.path.join(datadir, "era5_ml.nc")
    file_ml_T21 = os.path.join(datadir, "era5_ml_T21.nc")
    file_ml_int = os.path.join(datadir, "era5_ml_int.nc")

    print("[i] Retrieving full model level data...")
    c.retrieve('reanalysis-era5-complete', {
        'class'   : 'ea',
        'date'    : DATE,
        'expver'  : '1',
        'levelist': '1/to/137',
        'levtype' : 'ml',
        'param'   : '129/130/131/132/133/152',
        'stream'  : 'oper',
        'time'    : '00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00',
        #'time'    : '00:00:00/to/23:00:00',
        'type'    : 'an',
        'area'    : AREA,
        'grid'    : '0.25/0.25',               # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
        'format'  : 'netcdf', # 'short'??
        'truncation' : 'av' # 'av', '639', '21' vs 'resol':
    }, file_ml)
    
    #### ECMWF TAPE ARCHIVE (not ERA5)
    # server.retrieve({
    #     'dataset' : "copernicus",
    #     'class'   : 'ea',
    #     'date'    : DATE,
    #     'expver'  : '1',
    #     'levelist': '1/to/137',
    #     'levtype' : 'ml',
    #     'param'   : '129/130/131/132/133/152',
    #     'stream'  : 'oper',
    #     'time'    : '00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00',
    #     #'time'    : '00:00:00/to/23:00:00',
    #     'type'    : 'an',
    #     'area'    : AREA,
    #     'grid'    : '0.25/0.25',               # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
    #     'format'  : 'netcdf', # 'short'??
    #     'truncation' : 'av', # 'av', '639', '21' vs 'resol':
    #     'target'    : file_ml
    # })

    print(f"[i] Retrieving T21 model level data...")
    # - Request model level data - #
    c.retrieve('reanalysis-era5-complete', {
        'class'   : 'ea',
        'date'    : DATE,
        'expver'  : '1',
        'levelist': '1/to/137',
        'levtype' : 'ml',
        'param'   : '129/130/131/132/133/152',
        'stream'  : 'oper',
        'time'    : '00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00',
        # 'time'    : '00:00:00/to/23:00:00',
        'type'    : 'an',
        'area'    : AREA,
        'grid'    : '0.25/0.25',               # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
        'format'  : 'netcdf', # 'short'??
        'truncation' : '21' # 'av', '639', '21', T1279
    }, file_ml_T21)

    file_ml_coeff = 'input/era5-ml-coeff.csv'
    print("[i] Interpolating model levels...")
    era5_processor.prepare_interpolated_ml_ds(file_ml,file_ml_T21,file_ml_coeff,file_ml_int)             

if __name__ == '__main__':
    """Try changing working directory for Crontab"""
    try:
        os.chdir(os.path.dirname(sys.argv[0]))
    except:
        print('[i]   Working directory already set!')
    download_era5()