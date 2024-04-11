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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import filter, cmaps, lidar_processor

plt.style.use('latex_default.mplstyle')


def timelab_format_func(value, tick_number):
    dt = mdates.num2date(value)
    if dt.hour == 0:
        return "{}\n{}".format(dt.strftime("%Y-%b-%d"), dt.strftime("%H"))
    else:
        return dt.strftime("%H")


def plot_lidar_stacked_filter(CONFIG_FILE):
    """Visualize lidar measurements (time-height diagrams + absolute temperature measurements)"""
    
    """Settings"""
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    if config.get("INPUT","OBS_FILE") == "NONE":
        obs_list = sorted(glob.glob(os.path.join(config.get("INPUT","OBS_FOLDER") , config.get("GENERAL","RESOLUTION"))))
    else:
        obs_list = os.path.join(config.get("INPUT","OBS_FOLDER"), config.get("INPUT","OBS_FILE"))
    
    config["INPUT"]["ERA5-FOLDER"] = os.path.join(config.get("OUTPUT","FOLDER"),"era5-profiles")

    folder = "filter-stacked"
    os.makedirs(os.path.join(config.get("OUTPUT","FOLDER"),folder), exist_ok=True)
    zrange = eval(config.get("GENERAL","ALTITUDE_RANGE"))
    trange = eval(config.get("GENERAL","TRANGE"))
    
    for ii, obs in enumerate(obs_list):
        file_name = os.path.split(obs)[-1]
        ds = lidar_processor.open_and_decode_lidar_measurement(obs)

        """File name with time and duration"""
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
        duration_str   = duration_str + 'min'

        """Define timeframe"""
        # - Date for plotting should always refer to beginning of the plot (04:00 UTC) - #
        start_date = datetime.datetime.utcfromtimestamp(ds.time.values[0].astype('O')/1e9)
        start      = datetime.datetime.utcfromtimestamp(ds.integration_start_time.values[0].astype('O')/1e9)
        end        = datetime.datetime.utcfromtimestamp(ds.integration_end_time.values[-1].astype('O')/1e9)

        if config.get("GENERAL","TIMEFRAME_NIGHT") != "NONE":
            timeframe = eval(config.get("GENERAL", "TIMEFRAME_NIGHT"))
            if timeframe[1] < timeframe[0]:
                fixed_intervall = timeframe[1] + 24 - timeframe[0]
            else: 
                fixed_intervall = timeframe[1] - timeframe[0]
                
            fixed_start_date = datetime.datetime(start_date.year, start_date.month, start_date.day, timeframe[0], 0,0)
            reference_hour   = 15
            if (start_date.hour > reference_hour) and (fixed_start_date.hour > reference_hour):
                ds['date_startp'] = fixed_start_date
                ds['date_endp']   = fixed_start_date + datetime.timedelta(hours=fixed_intervall)
            elif (start_date.hour > reference_hour) and (fixed_start_date.hour < reference_hour): # prob in range of 0 to 10
                ds['date_startp'] = fixed_start_date + datetime.timedelta(hours=24)
                ds['date_endp']   = fixed_start_date + datetime.timedelta(hours=24+fixed_intervall)
            elif (start_date.hour < reference_hour) and (fixed_start_date.hour > reference_hour):
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
            
        """Temperature missing values (Change 0 to NaN)"""
        ds.temperature.values = np.where(ds.temperature == 0, np.nan, ds.temperature)
        ds.temperature_err.values = np.where(ds.temperature_err == 0, np.nan, ds.temperature_err)
        
        ds['alt_plot'] = (ds.altitude + ds.altitude_offset + ds.station_height) / 1000 #km
        vert_res_obs   = (ds['alt_plot'][1]-ds['alt_plot'][0]).values[0]
        minutes   = (ds['time'][1]-ds['time'][0]).values.astype("timedelta64[m]")
        minutes  = minutes.astype('int')
        print(minutes)

        """Temporal BW filter (Interpolate data gaps and remove again later)"""
        ds["temperature_interp"] = ds.temperature.interpolate_na(dim='time', method='linear', limit=None, use_coordinate='time', max_gap=None)
        temporal_bw_primes, temporal_bw_bg = filter.butterworth_filter(ds["temperature_interp"].values.T, highcut=1/15, fs=1/temp_res_obs, order=5, mode='high')
        
        ds["temperature_interp"] = ds["temperature_interp"].where(~np.isnan(ds.temperature.values), other=np.nan) 

        """Measurement data for plot"""
        tprime_bwf15, tbg15 = filter.butterworth_filter(ds["temperature"].values, highcut=1/15, fs=1/vert_res_obs, order=5, mode='high')
        #meanT = ds["temperature"].mean(dim='time')
        tprime_temp  = (ds["temperature"]-ds["temperature"].mean(dim='time')).values


        # - k=0 is place holder - #
        #vars = [tprime_temp, tprime_temp, tprime_bwf15]
        vars = [tprime_temp, ds["temperature_interp"].values, temporal_bw_primes, tprime_bwf15]

        """ERA5 data for plot"""
        era5_file_path = os.path.join(config["INPUT"]["ERA5-FOLDER"], file_name[0:13] + "-ml-int.nc")
        plot_era5 = False
        if os.path.exists(era5_file_path):
            ds_era5 = xr.open_dataset(era5_file_path)
            ds_era5 = ds_era5.sel(latitude=-53.75,longitude=292.25) # method='nearest'
            tprime_era5_T21  = ds_era5['tprime'].values
            tprime_era5_temp = (ds_era5["t"]-ds_era5["t"].rolling(time=10,center=True).mean()).values
            vert_res_era5    = (ds_era5['level'][1]-ds_era5['level'][0]).values / 1000 # km
            tprime_era5_bwf15, tbg15 = filter.butterworth_filter(ds_era5["t"].values, highcut=1/15, fs=1/vert_res_era5, order=5, mode='high')

            plot_era5 = True

        """Figure"""
        gskw = {'hspace':0.04, 'wspace':0.03, 'width_ratios': [4,2], 'height_ratios': [5,5,5,1]} #  , 'width_ratios': [5,5]}
        fig, axes = plt.subplots(4,2, figsize=(7,12), sharey=True, gridspec_kw=gskw)
        axes[3,0].axis('off')
        axes[3,1].axis('off')

        h_fmt      = mdates.DateFormatter('%H')
        hlocator   = mdates.HourLocator(byhour=range(0,24,2))
        filter_str =['T-T$_{T21}$ (ERA5)','T-T$_{tmean}$','T$_{BW:15km}$']
        
        clev = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32] # 32
        clev_contour = [-32,-16,-8,-4,-2,-1,1,2,4,8,16,32] 
        clev_l = [-16,-4,-1,1,4,16]
        cmap = cmaps.get_wave_cmap()
        norm = BoundaryNorm(boundaries=clev, ncolors=cmap.N, clip=True)

        for k in range(0,3):
            ax_lid = axes[k,0]
            ax0    = axes[k,1]
            
            if k==0:
                if plot_era5:
                    pcolor0 = ax_lid.pcolormesh(ds_era5['time'].values, ds_era5['level']/1000, tprime_era5_T21.T,
                                cmap=cmap, norm=norm)
                else:
                    ax_lid.text(0.5, 0.5, "No ERA5 data", transform=ax_lid.transAxes, horizontalalignment='center', verticalalignment='center', 
                                bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

            elif k==1:
                pcolor0 = ax_lid.pcolormesh(ds.time.values, ds.alt_plot.values, np.matrix.transpose(vars[k]),
                                cmap=cmap, norm=norm)
                if plot_era5:
                    ax_lid.contour(ds_era5['time'].values, ds_era5['level']/1000, tprime_era5_temp.T, levels=clev_contour,
                                colors='k', linewidths=0.3)

            else: # k==2
                pcolor0 = ax_lid.pcolormesh(ds.time.values, ds.alt_plot.values, np.matrix.transpose(vars[k]),
                                cmap=cmap, norm=norm)
                if plot_era5:
                    ax_lid.contour(ds_era5['time'].values, ds_era5['level']/1000, tprime_era5_bwf15.T, levels=clev_contour,
                                colors='k', linewidths=0.3)


            ax_lid.set_xlim(ds['date_startp'],ds['date_endp'])
            # ax_lid.xaxis.set_major_formatter(h_fmt)
            ax_lid.xaxis.set_major_formatter(plt.FuncFormatter(timelab_format_func))
            ax_lid.xaxis.set_major_locator(hlocator)
            ax_lid.yaxis.set_major_locator(MultipleLocator(10))
            ax_lid.yaxis.set_minor_locator(AutoMinorLocator()) 
            ax_lid.xaxis.set_minor_locator(AutoMinorLocator())
            ax_lid.xaxis.set_label_position('top')
            if k==0:
                ax_lid.tick_params(which='both', labelbottom=False,labeltop=True)
            else:
                ax_lid.tick_params(which='both', labelbottom=False,labeltop=False)
            ax_lid.set_ylabel('altitude / km')

            ypp = 0.96
            lw_thin=0.1
            lw_thick=2
            if k==0 and plot_era5:
                ds_era5_cut = ds_era5.sel(time=slice(start,end))

                cwind = 'darkorchid'
                ax_wind = ax_lid.twiny()
                ax_wind.set_xlim([-35,750])
                ax_wind.tick_params(axis='x', which='both', top=False, bottom=True, labelbottom=True,labeltop=False)
                ax_wind.set_xticks([0,50,100])
                wind_xticks = ax_wind.get_xticks()
                plt.xticks(wind_xticks[1:3], labels=['50', '100'], fontweight='normal', visible=True)
                ax_wind.tick_params(axis="x", top=False, bottom=True, labelbottom=True,labeltop=False)
                ax_wind.axvline(x=0,c=cwind,ls='--')
                ax_wind.xaxis.label.set_color(cwind)
                ax_wind.tick_params(axis='x', pad=-18, colors=cwind)
                ax_wind.text(0.21,0.0, 'u / ms$^{-1}$', color=cwind, verticalalignment='bottom', transform=ax_wind.transAxes)
                ax_lid.tick_params(which='both', top=True, bottom=False, labelbottom=False,labeltop=True)
                
                ax_wind.plot(np.mean(ds_era5["u"],axis=0),ds_era5_cut['level']/1000,lw=lw_thick,color=cwind)
                for jj in range(0,np.shape(ds_era5["u"])[0]):
                    ax_wind.plot(ds_era5["u"][jj],ds_era5['level']/1000,lw=lw_thin,color=cwind)
                ax_wind.text(0.03, ypp, filter_str[k], transform=ax_lid.transAxes, verticalalignment='top', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
            else:
                ax_lid.text(0.03, ypp, filter_str[k], transform=ax_lid.transAxes, verticalalignment='top', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})            
            ax_lid.grid()

            """T and T' (second axis)"""
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
            if k==2:
                ax1.set_xlabel("T' / K", color='red')
            else:
                ax1.tick_params(labelbottom=False, colors='red')
            
            if k==0:
                if plot_era5:
                    for t in range(0,np.shape(ds_era5_cut['t'])[0]):      
                        ax0.plot(ds_era5_cut["t"][t],ds_era5_cut['level']/1000,lw=lw_thin,color='black')
                        ax1.plot(ds_era5_cut["tprime"][t],ds_era5_cut['level']/1000,lw=lw_thin,color='red')
                    ax0.plot(np.mean(ds_era5_cut["t"],axis=0),ds_era5_cut['level']/1000,lw=lw_thick,color='black')
                    ax1.plot(np.mean(ds_era5_cut["tprime"],axis=0),ds_era5_cut['level']/1000, lw=lw_thick, color='red')
            else:
                for t in range(0,np.shape(ds['temperature'])[0],4):      
                    ax0.plot(ds["temperature"][t],ds['alt_plot'],lw=lw_thin,color='black')
                    ax1.plot(vars[k][t],ds['alt_plot'],lw=lw_thin,color='red')
                ax0.plot(np.mean(ds["temperature"],axis=0),ds['alt_plot'],lw=lw_thick,color='black')
                ax1.plot(np.nanmean(vars[k],axis=0),ds['alt_plot'], lw=lw_thick, color='red')
            ax0.grid()

            numb_str = ['a','b','c','d','e','f','g','h']
            xpp0 = 0.95
            xpp1 = 0.92
            ax_lid.text(xpp0, ypp, numb_str[2*k], verticalalignment='top', horizontalalignment='right', transform=ax_lid.transAxes, 
                                weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
            ax1.text(xpp1, ypp, numb_str[2*k+1], verticalalignment='top', horizontalalignment='right', transform=ax1.transAxes, 
                                weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
      

        # - COLORBAR - #
        cbar = fig.colorbar(pcolor0, ax=axes[-1,0], location='bottom', ticks=clev_l, fraction=1, shrink=0.9, aspect=25, extend='both') # aspect=30
        cbar.set_label(r"$T'$ / K")

        axes[0,0].set_ylim(zrange[0],zrange[1])
    
        """Formatting"""
        # if ds.fixed_timeframe.values:
        #     date = datetime.datetime.utcfromtimestamp(ds.date_endp.values.astype('O')/1e9)
        # else: 
        #     date = datetime.datetime.utcfromtimestamp(ds.time.values[-1].astype('O')/1e9)
        
        # - Use date of first measurement - #
        date = datetime.datetime.utcfromtimestamp(ds.time.values[0].astype('O')/1e9)
        # axes[0,0].set_xlabel('hours (UTC) starting on {}'.format(datetime.datetime.strftime(date, '%b %d, %Y')))  
        axes[0,0].text(-0.015, 1.0, "UTC", horizontalalignment='right', verticalalignment='bottom', transform=axes[0,0].transAxes)

        if ds.instrument_name == "":
            ds.instrument_name = "LIDAR"
        fig.suptitle('          German Aerospace Center (DLR)\n \
        {}, {}\n \
        ------------------------------\n \
        Resolution: {}$\,$km  x  {}$\,$min'.format(ds.instrument_name, ds.station_name, ds.altitude.resolution / 1000, ds.time.resolution))

        """Save figure"""
        fig_name = file_name[:14] + duration_str + '.png'
        fig.savefig(os.path.join(config.get("OUTPUT","FOLDER"),folder,fig_name), 
                    facecolor='w', edgecolor='w', format='png', dpi=150, bbox_inches='tight') # orientation='portrait'

        if (ii % 50) == 0:
            print("Plotted measurement: {}".format(ii))
        print("ERA5 plotted")


if __name__ == '__main__':
    """provide ini file as argument and pass it to function"""
    """Try changing working directory for Crontab"""
    try:
        os.chdir(os.path.dirname(sys.argv[0]))
    except:
        print('[i]  Working directory already set!')
    plot_lidar_stacked_filter(sys.argv[1])