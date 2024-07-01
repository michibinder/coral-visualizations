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
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

from eratools import era_filter, era_cmaps

plt.style.use('latex_default.mplstyle')


def plot_lidar_filter(CONFIG_FILE):
    """Visualize lidar measurements (time-height diagrams + absolute temperature measurements)"""
    
    """Settings"""
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    if config.get("INPUT","OBS_FILE") == "NONE":
        obs_list = sorted(glob.glob(os.path.join(config.get("INPUT","OBS_FOLDER") , '*T15Z900.nc')))
    else:
        obs_list = os.path.join(config.get("INPUT","OBS_FOLDER"), config.get("INPUT","OBS_FILE"))
    
    os.makedirs(config.get("OUTPUT","FOLDER"), exist_ok=True)
    zrange = eval(config.get("GENERAL","ALTITUDE_RANGE"))
    trange = eval(config.get("GENERAL","TRANGE"))
        
    for obs in obs_list:
        file_name = os.path.split(obs)[-1]
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
        # - Date for plotting should always refer to the center of the plot (04:00 UTC) - #
        if config.get("GENERAL","TIMEFRAME_NIGHT") != "NONE":
            timeframe = eval(config.get("GENERAL", "TIMEFRAME_NIGHT"))
            if timeframe[1] < timeframe[0]:
                fixed_intervall = timeframe[1] + 24 - timeframe[0]
            else: 
                fixed_intervall = timeframe[1] - timeframe[0]
                
            start_date = datetime.datetime.utcfromtimestamp(ds.time.values[0].astype('O')/1e9)
            fixed_start_date = datetime.datetime(start_date.year, start_date.month, start_date.day, timeframe[0], 0,0)
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

        """Data for plotting"""
        ds['alt_plot'] = (ds.altitude + ds.altitude_offset + ds.station_height) / 1000 #km
        vert_res       = (ds['alt_plot'][1]-ds['alt_plot'][0]).values[0]
        tprime_bwf20, tbg20 = era_filter.butterworth_filter(ds["temperature"].values, highcut=1/20, fs=1/vert_res, order=5, mode='low')
        tprime_bwf15, tbg15 = era_filter.butterworth_filter(ds["temperature"].values, highcut=1/15, fs=1/vert_res, order=5, mode='low')
        #ds['tprime_bwf20'] = butterworthf(ds, highcut=1/20, fs=1/0.1, order=5, single_column_filter=True)['tmp_pert']
        #tprime_bwf15, tbg15 = butterworthf(ds, highcut=1/15, fs=1/0.1, order=5, single_column_filter=True)['tmp_pert']
        #meanT = ds["temperature"].mean(dim='time')
        tprime_temp  = (ds["temperature"]-ds["temperature"].mean(dim='time')).values

        vars = [ds["temperature"].values, tprime_temp, tprime_bwf15, tprime_bwf20]
        """Figure"""
        gskw = {'hspace':0.05, 'wspace':0.02, 'width_ratios': [4,2], 'height_ratios': [6,6,6,6,1]} #  , 'width_ratios': [5,5]}
        fig, axes = plt.subplots(5,2, figsize=(7,12), sharey=True, gridspec_kw=gskw)
        axes[4,0].axis('off')
        axes[4,1].axis('off')

        h_fmt      = mdates.DateFormatter('%H')
        h_interv   = mdates.HourLocator(interval = 2)
        filter_str =['Temperature','Filter: Temp-mean','Filter: 15km-BW','Filter: 20km-BW']
        for k in range(0,4):
            ax_lid = axes[k,0]
            ax0    = axes[k,1]
            if k==0:
                trange = eval(config.get("GENERAL", "TRANGE"))
                clev   = np.arange(trange[0],trange[1],10)
                cmap   = plt.get_cmap('turbo')
            elif k==1:
                clev = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32] # 32
                clev_l = [-16,-4,-1,1,4,16]
                cmap = era_cmaps.get_wave_cmap()
            elif k==2:
                clev = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32] # 32
                clev_l = [-16,-4,-1,1,4,16]
                cmap = era_cmaps.get_wave_cmap()
            elif k==3:
                clev = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32] # 32
                clev_l = [-16,-4,-1,1,4,16]
                cmap = era_cmaps.get_wave_cmap()

            norm = BoundaryNorm(boundaries=clev, ncolors=cmap.N, clip=True)
            pcolor0 = ax_lid.pcolormesh(ds.time.values, ds.alt_plot.values, np.matrix.transpose(vars[k]),
                                cmap=cmap, norm=norm)

            ax_lid.set_xlim(ds['date_startp'],ds['date_endp'])
            ax_lid.xaxis.set_major_locator(h_interv)
            ax_lid.yaxis.set_major_locator(MultipleLocator(10))
            ax_lid.xaxis.set_major_formatter(h_fmt)
            ax_lid.yaxis.set_minor_locator(AutoMinorLocator()) 
            ax_lid.xaxis.set_minor_locator(AutoMinorLocator())
            ax_lid.xaxis.set_label_position('top')
            if k==0:
                ax_lid.tick_params(which='both', labelbottom=False,labeltop=True)
            else:
                ax_lid.tick_params(which='both', labelbottom=False,labeltop=False)
            ax_lid.set_ylabel('altitude / km')

            ypos = 0.945
            ax_lid.text(0.02, ypos, filter_str[k], transform=ax_lid.transAxes, verticalalignment='top', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
            # if k==0:
            #     info_str = ""
            #    ax_lid.text(0.5, ypos, info_str, transform=ax_lid.transAxes, verticalalignment='top', horizontalalignment='center', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
            ax_lid.grid()

            # ---- T-axis ---- #
            lw_thin=0.1
            lw_thick=2

            ax1 = ax0.twiny()
            ax1.axvline(x=0,c='grey',lw=lw_thick-1)
            trange_prof = eval(config.get("GENERAL", "TRANGE_PROF"))
            ax0.set_xlim(trange_prof[0],trange_prof[1])
            ax1.set_xlim([-49.5,25])

            ax0.yaxis.set_minor_locator(AutoMinorLocator()) 
            ax0.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            
            ax0.set_ylabel('altitude / km')
            ax0.yaxis.set_label_position("right")
            ax0.xaxis.set_label_position('top')
            ax0.yaxis.tick_right()
            ax0.tick_params(which='both', labelbottom=False, labeltop=True, labelright=True, left=True, top=True, bottom=False)
            
            ax1.tick_params(which='both', axis='x', bottom=True, top=False, labeltop=False, labelbottom=True, colors='red')
            ax1.spines['bottom'].set_color('red')
            ax1.xaxis.set_label_position('bottom')

            if k==0:
                ax0.set_xlabel('temperature / K')
            else:
                ax0.tick_params(labeltop=False)
            if k==3:
                ax1.set_xlabel("T' / K", color='red')
            else:
                ax1.tick_params(labelbottom=False, colors='red')
            
            for t in range(0,np.shape(ds['temperature'])[0],4):      
                ax0.plot(ds["temperature"][t],ds['alt_plot'],lw=lw_thin,color='black')
                if k > 0:
                    ax1.plot(vars[k][t],ds['alt_plot'],lw=lw_thin,color='red')
            ax0.plot(np.mean(ds["temperature"],axis=0),ds['alt_plot'],lw=lw_thick,color='black')
            if k>0:
                ax1.plot(np.nanmean(vars[k],axis=0),ds['alt_plot'], lw=lw_thick, color='red')
            ax0.grid()

            numb_str = ['a','b','c','d','e','f','g','h']
            xpp0 = 0.975
            xpp1 = 0.92
            ypp = 0.94
            ax_lid.text(xpp0, ypp, numb_str[2*k], verticalalignment='top', horizontalalignment='right', transform=ax_lid.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
            ax1.text(xpp1, ypp, numb_str[2*k+1], verticalalignment='top', horizontalalignment='right', transform=ax1.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
      

        # - COLORBAR - #
        cbar = fig.colorbar(pcolor0, ax=axes[4,0], location='bottom', ticks=clev_l, fraction=1, shrink=0.8, aspect=30, extend='both') # aspect=30
        cbar.set_label(r"$T'$ / K")

        axes[0,0].set_ylim(zrange[0],zrange[1])
    
        """Formatting"""
        # if ds.fixed_timeframe.values:
        #     date = datetime.datetime.utcfromtimestamp(ds.date_endp.values.astype('O')/1e9)
        # else: 
        #     date = datetime.datetime.utcfromtimestamp(ds.time.values[-1].astype('O')/1e9)
        
        # - Use date of first measurement - #
        date = datetime.datetime.utcfromtimestamp(ds.time.values[0].astype('O')/1e9)
        axes[0,0].set_xlabel('hours (UTC) starting on {}'.format(datetime.datetime.strftime(date, '%b %d, %Y')))  

        if ds.instrument_name == "":
            ds.instrument_name = "LIDAR"
        fig.suptitle('          German Aerospace Center (DLR)\n \
        {}, {}\n \
        ------------------------------\n \
        Resolution: {}$\,$km  x  {}$\,$min'.format(ds.instrument_name, ds.station_name, ds.altitude.resolution / 1000, ds.time.resolution))

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
        fig.savefig(os.path.join(config.get("OUTPUT","FOLDER"),fig_name), facecolor='w', edgecolor='w', format='png', dpi=150, bbox_inches='tight') # orientation='portrait'
    return


if __name__ == '__main__':
    """provide ini file as argument and pass it to function"""
    plot_lidar_filter(sys.argv[1])