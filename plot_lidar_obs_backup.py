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
import configparser
import datetime

import numpy as np
import xarray as xr
from scipy import signal

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

from eratools import era_filter, era_cmaps


def plot_lidar_obs(CONFIG_FILE):
    """Visualize lidar measurements (time-height diagrams + absolute temperature measurements)"""
    
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    
    if config.get("INPUT","OBS_FILE") == "NONE":
        obs_list = sorted(glob.glob(os.path.join(config.get("INPUT","OBS_FOLDER") , '*.nc')))
    else:
        obs_list = os.path.join(config.get("INPUT","OBS_FOLDER"), config.get("INPUT","OBS_FILE"))
        
        
    for obs in obs_list:
        file_name = os.path.split("/")[-1]
        ds = xr.open_dataset(obs, decode_times=False)

        """Decode time with time offset"""
        ds.assign_coords({'time':ds.time.values / 1000})
        ds.integration_start_time.values = ds.integration_start_time.values / 1000
        ds.integration_end_time.values = ds.integration_end_time.values / 1000
        
        unit_str = ds.time_offset.attrs['units']
        ds.attrs['reference_date'] = unit_str[14:-6]
        # Reference date is first reference
        # 'Time offset' is 'seconds' after reference date
        # Time is 'seconds' after time offset
        
        time_reference = datetime.datetime.strptime(ds.reference_date, '%Y-%m-%d %H:%M:%S.%f')
        time_offset = datetime.timedelta(seconds=float(ds.time_offset.values[0]))
        new_time_reference = time_reference + time_offset
        time_reference_str = datetime.datetime.strftime(new_time_reference, '%Y-%m-%d %H:%M:%S')
        
        ds.time.attrs['units'] = 'seconds since ' + time_reference_str
        ds.integration_start_time.attrs['units'] = 'seconds since ' + time_reference_str
        ds.integration_end_time.attrs['units'] = 'seconds since ' + time_reference_str
        
        ds = xr.decode_cf(ds, decode_coords = True, decode_times = True) 
        
        """Define timeframe"""
        # - Date for plotting should always refer to the center of the plot (04:00 UTC) - #
        if config.get("GENERAL","FIXED_TIMEFRAME") != "NONE":
            timeframe = eval(config.get("GENERAL", "FIXED_TIMEFRAME"))
            fixed_start = timeframe[0]
            fixed_end = timeframe[1]
            if fixed_end < fixed_start:
                fixed_intervall = fixed_end + 24 - fixed_start
            else: 
                fixed_intervall = fixed_end - fixed_start
                
            start_date = datetime.datetime.utcfromtimestamp(ds.time.values[0].astype('O')/1e9)
            fixed_start_date = datetime.datetime(start_date.year, start_date.month, start_date.day, fixed_start, 0,0)
            duration = datetime.datetime.utcfromtimestamp(ds.integration_end_time.values[-1].astype('O')/1e9) -  datetime.datetime.utcfromtimestamp(ds.integration_start_time.values[0].astype('O')/1e9)# for calendar
            
            reference_hour = 15
            if (start_date.hour > reference_hour) and (fixed_start_date.hour > reference_hour):
                ds['date_startp'] = fixed_start_date
                ds['date_endp'] = fixed_start_date + datetime.timedelta(hours=fixed_intervall)
            elif (start_date.hour > reference_hour) and (fixed_start_date.hour < reference_hour): # prob in range of 0 to 10
                ds['date_startp'] = fixed_start_date + datetime.timedelta(hours=24)
                ds['date_endp'] = fixed_start_date + datetime.timedelta(hours=24+fixed_intervall)
            elif (start_date.hour < reference_hour) and (fixed_start_date.hour > reference_hour):
                ds['date_startp'] = fixed_start_date - datetime.timedelta(hours=24)
                ds['date_endp'] = fixed_start_date - datetime.timedelta(hours=24-fixed_intervall)
            else: # (start_date.hour < 18) and (fixed_start_date.hour < 18):
                ds['date_startp'] = fixed_start_date
                ds['date_endp'] = fixed_start_date - datetime.timedelta(hours=fixed_intervall)
                
            ds['fixed_timeframe'] = 1
        else:
            ds['fixed_timeframe'] = 0
            
        """ Temperature missing values (Change 0 to NaN)"""
        ds.temperature.values = np.where(ds.temperature == 0, np.nan, ds.temperature)
        ds.temperature_err.values = np.where(ds.temperature_err == 0, np.nan, ds.temperature_err)

        """Altitude for plotting"""
        ds['alt_plot'] = (ds.altitude + ds.altitude_offset + ds.station_height) / 1000 #km

        if config.get("GENERAL", "PLOT_CONTENT") == "TEMP":
            ds = butterworthf(ds, highcut=1/20, fs=1/0.1, order=5, single_column_filter=True)
            var = "temperature"
            var_label = "Temperature / K"
            cb_range  = eval(config.get("GENERAL", "TRANGE"))
            clev      = np.arange(180,310,10)
            cmap      = plt.get_cmap('turbo')

        elif config.get("GENERAL", "PLOT_CONTENT") == "bwf20":
            ds = butterworthf(ds, highcut=1/20, fs=1/0.1, order=5, single_column_filter=True)
            var = "temperature"
            var_label = "T' (BWF20km) / K"
            cb_range = eval(config.get("GENERAL", "TRANGE_PERT"))
            clev   = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32] # 32
            clev_l = [-16,-4,-1,1,4,16]
            cmap   = era_cmaps.get_wave_cmap()

        elif config.get("GENERAL", "PLOT_CONTENT") == "bwf15":
            ds = butterworthf(ds, highcut=1/20, fs=1/0.1, order=5, single_column_filter=True)
            var = "tmp_pert"
            var_label = "T' (BWF15km) / K"
            cb_range = eval(config.get("GENERAL", "TRANGE_PERT"))
            clev   = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32] # 32
            clev_l = [-16,-4,-1,1,4,16]
            cmap   = era_cmaps.get_wave_cmap()

        elif config.get("GENERAL", "PLOT_CONTENT") == "tm":
            tmp_mean = ds.temperature.mean(dim='time')
            ds['tmp_tfilter'] = ds.temperature-tmp_mean
            clev   = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32] # 32
            clev_l = [-16,-4,-1,1,4,16]
            cmap   = era_cmaps.get_wave_cmap()

        elif config.get("GENERAL", "PLOT_CONTENT") == "rm":
            tmp_mean = ds.temperature.rolling(time=30, center = True).mean(dim='time')
            ds['tmp_runningM'] = ds.temperature-tmp_mean
            clev   = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32] # 32
            clev_l = [-16,-4,-1,1,4,16]
            cmap   = era_cmaps.get_wave_cmap()

        norm = BoundaryNorm(boundaries=clev, ncolors=cmap.N, clip=True)
        fig, axes = plt.subplots(1,2,figsize=(9,4.5), sharey=True, gridspec_kw={'hspace':0.15, 'wspace':0.04, 'width_ratios': [1,1,1]})

        pcolor0 = axes[0].pcolormesh(ds.time.values, ds.alt_plot.values, np.matrix.transpose(ds[var].values),
                                cmap=cmap, norm=norm)
        cbar = fig.colorbar(pcolor0, ax=axes[0],orientation='horizontal', shrink = 0.9, pad=0.03, extend='both')
        cbar.set_label(var_label)
        
        axes[1].plot(ds.temperature.mean(axis=0), ds.alt_plot)

        y_range = eval(config.get("GENERAL","ALTITUDE_RANGE"))
        axes[0].set_ylim(y_range[0],y_range[1])
        axes[0].set_ylabel('Altitude / km')

    
        """Formatting"""
        if ds.fixed_timeframe.values:
            date = datetime.datetime.utcfromtimestamp(ds.date_endp.values.astype('O')/1e9)
        else: 
            date = datetime.datetime.utcfromtimestamp(ds.time.values[-1].astype('O')/1e9)
        # use date of last measurement (date of morning day)
        axes[0].set_xlabel('Hours (UTC) on {}'.format(datetime.datetime.strftime(date, '%b %d, %Y')))

        h_fmt = mdates.DateFormatter('%H')
        h_interv = mdates.HourLocator(interval = 2)
        numb_str = ['a','b','c']
        for i, ax in enumerate(axes):
            ax.grid()
            ax.yaxis.set_minor_locator(AutoMinorLocator()) 
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_label_position('top') 
            ax.tick_params(which='both', top=True, labelbottom=False,labeltop=True)
            ax.xaxis.set_major_locator(h_interv)
            ax.xaxis.set_major_formatter(h_fmt)   

        axes[0].grid()
        if ds.instrument_name == "":
            ds.instrument_name = "LIDAR"
        fig.suptitle('          German Aerospace Center (DLR)\n \
        {}, {}\n \
        ------------------------------\n \
        Vertical resolution: {} km\n \
        Temporal resolution: {} min'.format(ds.instrument_name, ds.station_name, ds.altitude.resolution / 1000, ds.time.resolution / (1000*60)), fontsize=12)

        """Save figure"""
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        duration_str = ''
        if hours <= 9:
            duration_str = duration_str + '0' + str(int(hours))
        else:
            duration_str = duration_str + str(int(hours))
        if minutes <= 9:
            duration_str = duration_str + '0' + str(int(minutes))
        else:
            duration_str = duration_str + str(int(minutes))
        
        fig_name = file_name[:14] + duration_str + 'h.png'
        folder   = config.get("GENERAL", "PLOT_CONTENT")
        fig.savefig(os.path.join("output",folder,fig_name), facecolor='w', edgecolor='w', format='png', dpi=150, bbox_inches='tight') # orientation='portrait'
    return


def butterworthf(ds, highcut=1/20, fs=1/0.1, order=3, single_column_filter=True):
    """butterworth filter applied to matrix or each column seperately
        - uses the signal.butter and signal.lfilter functions of the SCIPY library
        - applies a low pass filter based on the given order and highcut frequency
    Input:
        - ds
        - highcut frequency (1/wavelength) 5 work good 20km?
        - fs (sampling frequency) -> 100m 
        - order of filter = 3
    Output:
        - ds_tmp, which now includes the filtered temperature background and the perturbation
    """
    
    b, a = butter_lowpass(highcut, fs, order=order) 
    # print(b,a)
    # print("filter stable!", np.all(np.abs(np.roots(a))<1))

    if single_column_filter:
        # filter each column (returns matrix with Nans at bottom and top)
        columns_bg = np.full(ds.temperature.values.shape, np.NaN)
        for col, column in enumerate(ds.temperature):
            mask = np.isnan(column) # dataarray
            c_masked = column[~mask] # dataarray
            if len(c_masked) >= 10:
                c_mirrored = np.append(np.flip(c_masked, axis=0), c_masked, axis=0) # numpy array
                # c_filtered = signal.lfilter(b,a,c_mirrored, axis=0)
                c_filtered = signal.filtfilt(b,a,c_mirrored, axis=0, padtype='even') # 'even'
                c_filtered = c_filtered[len(c_masked):]
                column_bg = column.copy()
                column_bg[~mask] = c_filtered
                columns_bg[col,:] = column_bg
            else: # column of NANs is just passed through
                columns_bg[col,:] = column
        ds['tmp_bg'] = (['time', 'altitude'], columns_bg)
    else: # not available @ mom
        da_tmp = ds.temperature
        ds_tmp = da_tmp.to_dataset()
        # filter as matrix (some data is lost at the upper boundary)
        da_tmp = ds_tmp.temperature.dropna(dim='altitude', how="any")
        ds_tmp_2 = da_tmp.to_dataset() # required for original window
        #ds_tmp['tmp_bg'] = ds_tmp.temperature
        tmp_mirrored = np.append(np.flip(ds_tmp_2.temperature, axis=1), ds_tmp_2.temperature, axis=1)
        # tmp_filtered = signal.lfilter(b,a,tmp_mirrored, axis=1)
        tmp_filtered = signal.filtfilt(b,a,tmp_mirrored, axis=1)
        tmp_filtered = tmp_filtered[:,len(da_tmp[0]):]
        ds_tmp_2['tmp_bg'] = (['time', 'altitude'], tmp_filtered)
        ds_tmp['tmp_bg'] = ds_tmp_2['tmp_bg']

    ds['tmp_pert'] = ds.temperature - ds.tmp_bg
    return ds


def butter_lowpass(highcut, fs, order=5):
    """
    defines the butterworth filter coefficient based on 
    sample frequency, cut_off frequency and filter order
    """
    nyq = 0.5 * fs # Nyquist frequency
    # highcut = 1/20
    # lowcut = 1/2000000
    # low_crit = lowcut / nyq
   
    high_crit = highcut / nyq # critical frequency ratio
    # Wn = [low_crit, high_crit] # bandpass
    
    b, a = signal.butter(order, high_crit, btype='low', analog=False)
    return b, a


def plot_obs_overview(file_location='../..', file_name="v12means.nc", SETTINGS=None, save_fig=False):
    """
    Visualize overview (nightly mean plot for whole period with measurements)
    """
    
    path = os.path.join(file_location, file_name)
    
    ds = xr.open_dataset(path, decode_times=True)
    ds['alt_plot'] = (ds.altitude + ds.altitude_offset + ds.station_height) / 1000 #km
    ds.temperature.values = np.where(ds.temperature == 0, np.nan, ds.temperature) # Change 0 to NaN
        
    fig, ax0 = plt.subplots(figsize=(10,4))
    im_temp = ax0.pcolormesh(ds.time, ds.alt_plot, np.matrix.transpose(ds.temperature.values),
                             cmap='jet', vmin=140, vmax=280)
    ax0.set_ylim(20,100)
    
    # Labels
    cbar = fig.colorbar(im_temp, ax=ax0)
    cbar.set_label('Temperature (K)')
    ax0.set_xlabel('Time')
    ax0.set_ylabel('Altitude (km)')
    
    ax0.xaxis.tick_bottom()
    plt.grid()
    
    
    # X-Ticks
    # interv = mdates.MonthLocator(interval = 3)
    # ax0.xaxis.set_major_locator(interv)
    fmt = mdates.DateFormatter('%b-%y')
    # fmt = mdates.ConciseDateFormatter(mdates.AutoDateLocator())
    ax0.xaxis.set_major_formatter(fmt)
    # ax0.xaxis.set_tick_params(rotation=30)
    
    # X-limits
    if (SETTINGS['NIGHTLY_MEAN_FIXED'] == '1'):
        timeframe = eval(SETTINGS['FIXED_TIMEFRAME_NM'])
        timeframe[0] = datetime.datetime.strptime(timeframe[0], '%Y-%m')
        timeframe[1] = datetime.datetime.strptime(timeframe[1], '%Y-%m')
        
        ax0.set_xlim(timeframe[0],timeframe[1])
        
    # ax.set_title()
    fig.suptitle('German Aerospace Center (DLR) - Nightly mean temperature profiles           \n\
                 {}, {}           '.format(ds.instrument_name, ds.station_name), fontsize=12)
    fig.subplots_adjust(top=0.85)
    # fig.tight_layout(rect=[1, 1, 1, 0.6])
    # fig.tight_layout()
    
    if save_fig:
        fig_name = 'nightly_means.png'
        fig.savefig(SETTINGS['PLOT_FOLDER_NM'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png') # orientation='portrait'
    
    return fig


if __name__ == '__main__':
    """provide ini file as argument and pass it to function"""
    plot_lidar_obs(sys.argv[1])