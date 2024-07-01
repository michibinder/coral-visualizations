################################################################################
# Copyright 2023 German Aerospace Center                                       #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

import datetime
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import filter

"""Config"""
reference_hour  = 15 # 15 for CORAL
fixed_timeframe = 24 # h

def open_and_decode_lidar_measurement(obs: str):
    """Open and decode time of NC-file (lidar obs)"""

    ds = xr.open_dataset(obs, decode_times=False)

    """Decode time with time offset"""
    # - Change from milliseconds to seconds - #
    # ds.assign_coords({'time':ds.time.values / 1000})
    ds.coords['time'] = ds.time.values / 1000
     
    # - Set reference date - #
    ## 'Time offset' is 'seconds' after reference date
    ## 'Time' is 'seconds' after 'Time offset'
    # unit_str = ds.time_offset.attrs['units']
    unit_str = "seconds since 1970-01-01 00:00:00.00 00:00"
    ds.attrs['reference_date'] = unit_str[14:-6]
    
    # - Set reference date in units attribute - #
    time_reference = datetime.datetime.strptime(ds.reference_date, '%Y-%m-%d %H:%M:%S.%f')
    #time_offset = datetime.timedelta(seconds=float(ds.time_offset.values[0]))
    time_offset = datetime.timedelta(seconds=float(ds.time_offset.values))
    new_time_reference = time_reference + time_offset
    time_reference_str = datetime.datetime.strftime(new_time_reference, '%Y-%m-%d %H:%M:%S')

    ds.time.attrs['units'] = 'seconds since ' + time_reference_str
    if "integration_start_time" in ds.variables:
        ds.integration_start_time.values         = ds.integration_start_time.values / 1000
        ds.integration_end_time.values           = ds.integration_end_time.values / 1000
        ds.integration_start_time.attrs['units'] = 'seconds since ' + time_reference_str
        ds.integration_end_time.attrs['units']   = 'seconds since ' + time_reference_str

    ds = xr.decode_cf(ds, decode_coords = True, decode_times = True) 

    if len(ds.time) > 1:
        ds.time.attrs['resolution']     = (ds.time.values[1]-ds.time.values[0]).astype('timedelta64[m]')
        ds.altitude.attrs['resolution'] = ds.altitude.values[1]-ds.altitude.values[0]
    else:
        print(f"[i]  No data available for: {obs.split('/')[-1]}")
        # tqdm.write(f"[i]   No data available for: {obs.split('/')[-1]}")

        return

    """Define timeframe"""
    # - Date for plotting should always refer to beginning of the plot (04:00 UTC) - #
    # ds.attrs["start_time_utc"] = datetime.datetime.utcfromtimestamp(ds.integration_start_time.values[0].astype('O')/1e9)
    # ds.attrs["end_time_utc"]   = datetime.datetime.utcfromtimestamp(ds.integration_end_time.values[-1].astype('O')/1e9)
    ds.attrs["start_time_utc"] = datetime.datetime.utcfromtimestamp(ds.time.values[0].astype('O')/1e9)
    ds.attrs["end_time_utc"]   = datetime.datetime.utcfromtimestamp(ds.time.values[-1].astype('O')/1e9)
    ds.attrs["duration"]       = ds.end_time_utc - ds.start_time_utc

    """Compose duration string"""
    hours = ds.duration.seconds // 3600
    minutes = (ds.duration.seconds % 3600) // 60
    duration_str = ''
    if hours <= 9:
        duration_str = duration_str + '0' + str(int(hours))
    else:
        duration_str = duration_str + str(int(hours))
    duration_str = duration_str + 'h'
    if minutes <= 9:
        duration_str = duration_str + '0' + str(int(minutes))
    else:
        duration_str = duration_str + str(int(minutes))
    ds.attrs["duration_str"] = duration_str + 'min'
    
    return ds


def process_lidar_measurement(config: dict, ds: object):
    """Process lidar measurement (time decoding, altitude for plots, filter,...)"""

    """Define timeframe for plot"""
    if config.get("GENERAL","TIMEFRAME_NIGHT") != "NONE":
        timeframe = eval(config.get("GENERAL", "TIMEFRAME_NIGHT"))
        if timeframe[1] < timeframe[0]:
            fixed_intervall = timeframe[1] + 24 - timeframe[0]
        else: 
            fixed_intervall = timeframe[1] - timeframe[0]
            
        fixed_start_date = datetime.datetime(ds.start_time_utc.year, ds.start_time_utc.month, ds.start_time_utc.day, timeframe[0], 0,0)

        if timeframe[0] == 0 and timeframe[1] == 24:
            # - TELMA AT SOUTHPOLE - #
            ds['date_startp'] = fixed_start_date
            ds['date_endp']   = fixed_start_date + datetime.timedelta(hours=fixed_intervall)
        else:
            # - CORAL, ... - #
            if (ds.start_time_utc.hour > reference_hour) and (fixed_start_date.hour > reference_hour):
                ds['date_startp'] = fixed_start_date
                ds['date_endp']   = fixed_start_date + datetime.timedelta(hours=fixed_intervall)
            elif (ds.start_time_utc.hour > reference_hour) and (fixed_start_date.hour < reference_hour): # prob in range of 0 to 10
                ds['date_startp'] = fixed_start_date + datetime.timedelta(hours=24)
                ds['date_endp']   = fixed_start_date + datetime.timedelta(hours=24+fixed_intervall)
            elif (ds.start_time_utc.hour < reference_hour) and (fixed_start_date.hour > reference_hour):
                ds['date_startp'] = fixed_start_date - datetime.timedelta(hours=24)
                ds['date_endp']   = fixed_start_date - datetime.timedelta(hours=24-fixed_intervall)
            else: # (start_time_utc.hour < 15) and (fixed_start_date.hour < 15):
                ds['date_startp'] = fixed_start_date
                ds['date_endp']   = fixed_start_date + datetime.timedelta(hours=fixed_intervall)
            
    else:
        ds['date_startp'] = ds.start_time_utc
        ds['date_endp']   = ds.start_time_utc + datetime.timedelta(hours=fixed_timeframe)
        
    """ Temperature missing values (Change 0 to NaN)"""
    ds.temperature.values = np.where(ds.temperature == 0, np.nan, ds.temperature)
    ds.temperature_err.values = np.where(ds.temperature_err == 0, np.nan, ds.temperature_err)

    """Measurement data for plot"""
    if "altitude_offset" in ds.variables:
        ds['alt_plot'] = (ds.altitude + ds.altitude_offset.values + ds.station_height.values) / 1000 #km
    else:
        ds['alt_plot'] = ds.altitude / 1000
    ds.attrs['vres'] = (ds['alt_plot'][1]-ds['alt_plot'][0]).values # in km
    ds.attrs['tres'] = (ds['time'][1]-ds['time'][0]).values.astype("timedelta64[m]").astype('int') # in minutes

    return ds

def calculate_primes(ds, temporal_cutoff, vertical_cutoff):
    """Calculate temporal and vertical Butterworth filter"""

    # - Vertical BW filter - #
    tprime_vbwf, tbg_vbwf = filter.butterworth_filter(ds["temperature"].values, cutoff=1/vertical_cutoff, fs=1/ds.vres, order=5, mode='both')
    ds["tprime_vbwf"] = (('t', 'z'), tprime_vbwf)
    ds["tbg_vbwf"]    = (('t', 'z'), tbg_vbwf)

    # - Temporal BW filter (Interpolate data gaps and remove again later) - #
    ds["temperature_interp"] = ds.temperature.interpolate_na(dim='time', method='linear', limit=None, use_coordinate='time', max_gap=None)
    tprime_tbwf, tbg_tbwf = filter.butterworth_filter(ds["temperature_interp"].values.T, cutoff=1/temporal_cutoff, fs=1/ds.tres, order=5, mode='both')
    tprime_tbwf = tprime_tbwf.T
    tbg_tbwf    = tbg_tbwf.T
    tprime_tbwf  = np.where(~np.isnan(ds.temperature.values), tprime_tbwf, np.nan) 
    tbg_tbwf     = np.where(~np.isnan(ds.temperature.values), tbg_tbwf, np.nan) 
    ds["tprime_tbwf"] = (('t', 'z'), tprime_tbwf)
    ds["tbg_tbwf"]    = (('t', 'z'), tbg_tbwf)

    # - Subtract nightly mean - #
    ds["tprime_nm"]  = (ds["temperature"]-ds["temperature"].mean(dim='time'))

    return ds