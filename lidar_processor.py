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

import filter

def open_and_decode_lidar_measurement(obs: str):
    """Open and decode time of NC-file (lidar obs)"""

    ds = xr.open_dataset(obs, decode_times=False)

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
    ds.attrs["start_date"]     = datetime.datetime.utcfromtimestamp(ds.time.values[0].astype('O')/1e9)
    ds.attrs["start_time_utc"] = datetime.datetime.utcfromtimestamp(ds.integration_start_time.values[0].astype('O')/1e9)
    ds.attrs["end_time_utc"]   = datetime.datetime.utcfromtimestamp(ds.integration_end_time.values[-1].astype('O')/1e9)
    ds.attrs["duration"]       = ds.end_time_utc - ds.start_time_utc
    return ds


def process_lidar_measurement(config: dict, ds: object):
    """Process lidar measurement (time decoding, altitude for plots, filter,...)"""

    if config.get("GENERAL","TIMEFRAME_NIGHT") != "NONE":
        timeframe = eval(config.get("GENERAL", "TIMEFRAME_NIGHT"))
        if timeframe[1] < timeframe[0]:
            fixed_intervall = timeframe[1] + 24 - timeframe[0]
        else: 
            fixed_intervall = timeframe[1] - timeframe[0]
            
        fixed_start_date = datetime.datetime(ds.start_date.year, ds.start_date.month, ds.start_date.day, timeframe[0], 0,0)
        reference_hour   = 15
        if (ds.start_date.hour > reference_hour) and (fixed_start_date.hour > reference_hour):
            ds['date_startp'] = fixed_start_date
            ds['date_endp']   = fixed_start_date + datetime.timedelta(hours=fixed_intervall)
        elif (ds.start_date.hour > reference_hour) and (fixed_start_date.hour < reference_hour): # prob in range of 0 to 10
            ds['date_startp'] = fixed_start_date + datetime.timedelta(hours=24)
            ds['date_endp']   = fixed_start_date + datetime.timedelta(hours=24+fixed_intervall)
        elif (ds.start_date.hour < reference_hour) and (fixed_start_date.hour > reference_hour):
            ds['date_startp'] = fixed_start_date - datetime.timedelta(hours=24)
            ds['date_endp']   = fixed_start_date - datetime.timedelta(hours=24-fixed_intervall)
        else: # (start_date.hour < 15) and (fixed_start_date.hour < 15):
            ds['date_startp'] = fixed_start_date
            ds['date_endp']   = fixed_start_date + datetime.timedelta(hours=fixed_intervall)
            
        ds['fixed_timeframe'] = 1
    else:
        timeframe = config.getint("GENERAL", "TIMEFRAME")
        start_date = datetime.datetime.utcfromtimestamp(ds.time.values[0].astype('O')/1e9)
        ds['date_startp'] = start_date
        ds['date_endp']   = start_date + datetime.timedelta(hours=timeframe)
        ds['fixed_timeframe'] = 1
        
    """ Temperature missing values (Change 0 to NaN)"""
    ds.temperature.values = np.where(ds.temperature == 0, np.nan, ds.temperature)
    ds.temperature_err.values = np.where(ds.temperature_err == 0, np.nan, ds.temperature_err)

    """Measurement data for plot"""
    ds['alt_plot'] = (ds.altitude + ds.altitude_offset.values + ds.station_height.values) / 1000 #km
    vert_res_obs   = (ds['alt_plot'][1]-ds['alt_plot'][0]).values
    tprime_bwf15, tbg15 = filter.butterworth_filter(ds["temperature"].values, highcut=1/15, fs=1/vert_res_obs, order=5, mode='high')
    ds["tprime_bwf15"] = (["time", "altitude"], tprime_bwf15)
    #meanT = ds["temperature"].mean(dim='time')
    ds["tprime_temp"]  = ds["temperature"]-ds["temperature"].mean(dim='time')

    return ds