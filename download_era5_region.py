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

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import filter, cmaps, era5_processor


def download_era5_region(CONFIG_FILE):
    """Visualize lidar measurements (time-height diagrams + absolute temperature measurements)"""
    
    """Settings"""
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    if config.get("INPUT","OBS_FILE") == "NONE":
        obs_list = sorted(glob.glob(os.path.join(config.get("INPUT","OBS_FOLDER") , config.get("GENERAL","RESOLUTION"))))
    else:
        obs_list = os.path.join(config.get("INPUT","OBS_FOLDER"), config.get("INPUT","OBS_FILE"))
    
    config['GENERAL']['NTASKS'] = str(int(multiprocessing.cpu_count()-2))
    print("[i]   CPUs available: {}".format(multiprocessing.cpu_count()))
    print("[i]   CPUs used: {}".format(config.get("GENERAL","NTASKS")))
    print("[i]   Observations (without duration limit): {}".format(len(obs_list)))

    """Define ERA5 data folder"""
    config["OUTPUT"]["ERA5-FOLDER"] = os.path.join(config.get("OUTPUT","FOLDER"),"era5-region")
    os.makedirs(config.get("OUTPUT","ERA5-FOLDER"), exist_ok=True)

    procs = []
    sema = multiprocessing.Semaphore(config.getint("GENERAL","NTASKS"))
    ii = 0
    # - Start processes - #
    for ii, obs in enumerate(obs_list):
        sema.acquire()
        proc = multiprocessing.Process(target=download_and_interpolate_era5_data, args=(ii, config, obs, sema))
        procs.append(proc)
        proc.start()   

    # - Complete processes - #
    for proc in procs:
        proc.join()


def download_and_interpolate_era5_data(ii,config,obs,sema):
    file_name = os.path.split(obs)[-1]

    with xr.open_dataset(obs, decode_times=False) as ds:
        """Decode time with time offset"""
        # - Change from milliseconds to seconds - #
        # ds.assign_coords({'time':ds.time.values / 1000})
        ds.coords['time'] = ds.time.values / 1000
        ds.integration_start_time.values = ds.integration_start_time.values / 1000
        ds.integration_end_time.values = ds.integration_end_time.values / 1000
        
        # - Set reference date - #
        ### Reference date is first reference
        ### 'Time offset' is 'seconds' after reference date
        ### Time is 'seconds' after time offset
        unit_str = ds.time_offset.attrs['units']
        ds.attrs['reference_date'] = unit_str[14:-6]
        
        # - Set reference date in units attribute - #
        time_reference = datetime.datetime.strptime(ds.reference_date, '%Y-%m-%d %H:%M:%S.%f')
        time_offset = datetime.timedelta(seconds=float(ds.time_offset.values[0]))
        new_time_reference = time_reference + time_offset
        time_reference_str = datetime.datetime.strftime(new_time_reference, '%Y-%m-%d %H:%M:%S')

        ds.time.attrs['units'] = 'seconds since ' + time_reference_str
        ds.integration_start_time.attrs['units'] = 'seconds since ' + time_reference_str
        ds.integration_end_time.attrs['units'] = 'seconds since ' + time_reference_str
        ds.time.attrs['resolution'] = "15"

        ds = xr.decode_cf(ds, decode_coords = True, decode_times = True) 

        """Define timeframe"""
        # - Date for plotting should always refer to beginning of the plot (04:00 UTC) - #
        start_date = datetime.datetime.utcfromtimestamp(ds.time.values[0].astype('O')/1e9)
        duration = datetime.datetime.utcfromtimestamp(ds.integration_end_time.values[-1].astype('O')/1e9) -  datetime.datetime.utcfromtimestamp(ds.integration_start_time.values[0].astype('O')/1e9)# for calendar

    if duration > datetime.timedelta(hours=6):
        """Check if file already exists"""
        nc_file_name = file_name[:13]
        file_ml     = os.path.join(config.get("OUTPUT","ERA5-FOLDER"), nc_file_name + '-ml.nc')
        file_ml_T21 = os.path.join(config.get("OUTPUT","ERA5-FOLDER"), nc_file_name + '-ml-T21.nc')
        file_ml_int = os.path.join(config.get("OUTPUT","ERA5-FOLDER"), nc_file_name + '-ml-int.nc')
        file_pl     = os.path.join(config.get("OUTPUT","ERA5-FOLDER"), nc_file_name + '-pl.nc')
        file_pvu    = os.path.join(config.get("OUTPUT","ERA5-FOLDER"), nc_file_name + '-pvu.nc')

        """Download ERA5 data"""
        DATE = start_date.strftime("%Y-%m-%d") + "/to/" + (start_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        ## AREA = '-25/-120/-85/-30',          
        AREA = config.get("ERA5","AREA")
        c = cdsapi.Client(quiet=True)

        if not os.path.exists(file_ml_int):
            if not os.path.exists(file_ml):
                print("[i][{}]   Retrieving full model data...".format(ii))
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
                    'resol'   : 'av' # 'av', '639', '21'
                }, file_ml)

            if not os.path.exists(file_ml_T21):
                print("[i][{}]   Retrieving T21 model data...".format(ii))
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
                    'resol'   : '21' # 'av', '639', '21'
                }, file_ml_T21)

            file_ml_coeff = 'input/era5-ml-coeff.csv'
            print("[i][{}]   Interpolating model levels...".format(ii))
            era5_processor.prepare_interpolated_ml_ds(file_ml,file_ml_T21,file_ml_coeff,file_ml_int)
            os.remove(file_ml)
            os.remove(file_ml_T21)

        if (not os.path.exists(file_pl))
            print("[i][{}]   Retrieving pressure level data...".format(ii))
            c.retrieve('reanalysis-era5-complete', {
                'class'   : 'ea',
                'date'    : DATE,
                'expver'  : '1',
                'levelist': '1/to/1000',
                'levtype' : 'pl',
                'param'   : '60.128/129.128/131/132', # '60.128/129.128/130.128/131/132/133.128/138.128/155.128/157.128'
                'stream'  : 'oper',
                'time'    : '00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00',
                # 'time'    : '00:00:00/to/23:00:00',
                'type'    : 'an',
                'area'    : AREA,          # North, West, South, East. Default: global
                'grid'    : '0.25/0.25',               # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
                'format'  : 'netcdf',                # Output needs to be regular lat-lon, so only works in combination with 'grid'!
            }, file_pl)

        if (not os.path.exists(file_pvu))
            print("[i][{}]   Retrieving 2PVU level data...".format(ii))
            c.retrieve('reanalysis-era5-complete', {
                'class'   : 'ea',
                'date'    : DATE,
                'expver'  : '1',
                'levelist': '2000',
                'levtype' : 'pv',
                'param'   : '3.128/54.128/129.128/131.128/132.128/133.128',
                'stream'  : 'oper',
                'time'    : '00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00',
                'type'    : 'an',
                'area'    : AREA,          # North, West, South, East. Default: global
                'grid'    : '0.25/0.25',               # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
                'format'  : 'netcdf',
            }, file_pvu)

        print("[i][{}]   ERA5 data prepared for observation: {}".format(ii,obs))
    else:
        print("[i][{}]   Duration below limit for observation: {}".format(ii,obs))
    sema.release()

if __name__ == '__main__':
    """provide ini file as argument and pass it to function"""
    download_era5_region(sys.argv[1])