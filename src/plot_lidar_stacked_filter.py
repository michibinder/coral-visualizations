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

import filter, cmaps, lidar_processor, plt_helper

plt.style.use('latex_default.mplstyle')

"""Config"""
VERTICAL_CUTOFF = 15 # km (LAMBDA_CUT)
TEMPORAL_CUTOFF = 8*60 # min (TAU_CUT)

def plot_lidar_stacked_filter(config, obs, pbar):
    file_name = os.path.split(obs)[-1]

    zrange = eval(config.get("GENERAL","ALTITUDE_RANGE"))
    trange = eval(config.get("GENERAL","TRANGE"))
    ds = lidar_processor.open_and_decode_lidar_measurement(obs)
    if ds is None:
        return
    ds = lidar_processor.process_lidar_measurement(config, ds)

    """Process data for plotting"""
    # - Temporal BW filter (Interpolate data gaps and remove again later) - #
    ds["temperature_interp"] = ds.temperature.interpolate_na(dim='time', method='linear', limit=None, use_coordinate='time', max_gap=None)
    temporal_bw_primes, temporal_bw_bg = filter.butterworth_filter(ds["temperature_interp"].values.T, cutoff=1/TEMPORAL_CUTOFF, fs=1/ds.tres, order=5, mode='both')
    temporal_bw_primes = temporal_bw_primes.T
    temporal_bw_bg     = temporal_bw_bg.T
    temporal_bw_primes = np.where(~np.isnan(ds.temperature.values), temporal_bw_primes, np.nan) 
    temporal_bw_bg     = np.where(~np.isnan(ds.temperature.values), temporal_bw_bg, np.nan) 
    # ds["temperature_interp"] = ds["temperature_interp"].where(~np.isnan(ds.temperature.values), other=np.nan) 

    # - Vertical BW filter - #
    # vertical_bw_primes, vertical_bw_bg = filter.butterworth_filter(ds["temperature"].values, highcut=1/VERTICAL_CUTOFF, fs=1/ds.vres, order=5, mode='high')
    Q3, Q1 = filter.butterworth_filter(temporal_bw_primes, cutoff=1/VERTICAL_CUTOFF, fs=1/ds.vres, order=5, mode='both')
    Q4, Q2 = filter.butterworth_filter(temporal_bw_bg, cutoff=1/VERTICAL_CUTOFF, fs=1/ds.vres, order=5, mode='both')

    plot_vars  = [Q2, Q1, Q4, Q3]
    filter_str = ['$\lambda$ > $\lambda_{c}$, T > T$_{c}$ (BG)', '$\lambda$ > $\lambda_{c}$, T < T$_{c}$', '$\lambda$ < $\lambda_{c}$, T > T$_{c}$ (MWs)', '$\lambda$ < $\lambda_{c}$, T < T$_{c}$']

    # ---- DEFINITION OF QUADRANTS ---- #
    # Q1: lambda > lambda_cut, tau < tau_cut (Large lambda, short tau)
    # Q2: lambda > lambda_cut, tau > tau_cut (Background)
    # Q3: lambda < lambda_cut, tau < tau_cut (Short lambda, short tau)
    # Q4: lambda < lambda_cut, tau > tau_cut (MWs)        

    """ERA5 data for plot"""
    era5_file_path = os.path.join(config["INPUT"]["ERA5-FOLDER"], file_name[0:13] + "-ml-int.nc")
    plot_era5 = False
    if os.path.exists(era5_file_path):
        ds_era5 = xr.open_dataset(era5_file_path)
        if config.getboolean("ERA5","WESTERN_COORDS"):
            lon = config.getfloat("ERA5","LON") + 360
        else:
            lon = config.getfloat("ERA5","LON")
        ds_era5 = ds_era5.sel(latitude=config.getfloat("ERA5","LAT"),longitude=lon)
        
        # tprime_era5_T21  = ds_era5['tprime'].values
        # tprime_era5_temp = (ds_era5["t"]-ds_era5["t"].rolling(time=10,center=True).mean()).values
        # tprime_era5_bwf15, tbg15 = filter.butterworth_filter(ds_era5["t"].values, cutoff=1/15, fs=1/vert_res_era5, order=5, mode='high')

        vert_res_era5    = (ds_era5['level'][1]-ds_era5['level'][0]).values / 1000 # km
        temp_res_era5    = 60 # min
        temporal_bw_primes_era5, temporal_bw_bg_era5 = filter.butterworth_filter(ds_era5["t"].values.T, cutoff=1/TEMPORAL_CUTOFF, fs=1/temp_res_era5, order=5, mode='both')
        temporal_bw_primes_era5 = temporal_bw_primes_era5.T
        temporal_bw_bg_era5     = temporal_bw_bg_era5.T
        Q3_era5, Q1_era5 = filter.butterworth_filter(temporal_bw_primes_era5, cutoff=1/VERTICAL_CUTOFF, fs=1/vert_res_era5, order=5, mode='both')
        Q4_era5, Q2_era5 = filter.butterworth_filter(temporal_bw_bg_era5, cutoff=1/VERTICAL_CUTOFF, fs=1/vert_res_era5, order=5, mode='both')

        plot_vars_era5  = [ds_era5["t"].values, Q1_era5, Q4_era5, Q3_era5]
        plot_era5       = True


    """Figure"""
    gskw = {'hspace':0.04, 'wspace':0.03, 'width_ratios': [4.5,2], 'height_ratios': [5,5,5,5,1]} #  , 'width_ratios': [5,5]}
    fig, axes = plt.subplots(5,2, figsize=(7,12), sharey=True, gridspec_kw=gskw)
    axes[-1,0].axis('off')
    axes[-1,1].axis('off')

    h_fmt      = mdates.DateFormatter('%H')
    hlocator   = mdates.HourLocator(byhour=range(0,24,2))
    clev = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32] # 32
    # clev_contour = [-32,-16,-8,-4,-2,-1,1,2,4,8,16,32]
    clev_contour = clev
    clev_l = [-16,-4,-1,1,4,16]
    cbar_l = "T' / K" 
    cmap   = cmaps.get_wave_cmap()
    norm   = BoundaryNorm(boundaries=clev, ncolors=cmap.N, clip=True)

    cb_range  = eval(config.get("GENERAL", "TRANGE"))
    clev_T    = np.arange(cb_range[0],cb_range[1],10)
    clev_l_T  = np.arange(cb_range[0]+10,cb_range[1]-10,20)
    cbar_l_T  = "temperature / K" 
    cmap_T    = plt.get_cmap('turbo')
    norm_T    = BoundaryNorm(boundaries=clev_T, ncolors=cmap_T.N, clip=True)

    for k in range(0,4):
        ax_lid = axes[k,0]
        ax0    = axes[k,1]

        if k==0:
            pcolor_T = ax_lid.pcolormesh(ds.time.values, ds.alt_plot.values, np.matrix.transpose(plot_vars[k]),
                            cmap=cmap_T, norm=norm_T)

        else:
            pcolor0 = ax_lid.pcolormesh(ds.time.values, ds.alt_plot.values, np.matrix.transpose(plot_vars[k]),
                            cmap=cmap, norm=norm)
            if plot_era5:
                ax_lid.contour(ds_era5['time'].values, ds_era5['level']/1000, plot_vars_era5[k].T, levels=clev_contour,
                            colors='k', linewidths=0.3)
        
        ax_lid.set_xlim(ds['date_startp'],ds['date_endp'])
        # ax_lid.xaxis.set_major_formatter(h_fmt)
        ax_lid.xaxis.set_major_formatter(plt.FuncFormatter(plt_helper.timelab_format_func))
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
            ds_era5_cut = ds_era5.sel(time=slice(ds.start_time_utc,ds.end_time_utc))

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
            
            #### CHANGE to era5_cut
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
        if k==3:
            ax1.set_xlabel("T' / K", color='red')
        else:
            ax1.tick_params(labelbottom=False, colors='red')
        
        if k==0:
            for t in range(0,np.shape(ds['temperature'])[0],4): 
                ax0.plot(ds["temperature"][t],ds['alt_plot'],lw=lw_thin,color='black')
            ax0.plot(np.mean(ds["temperature"],axis=0),ds['alt_plot'],lw=lw_thick,color='black')

            if plot_era5:
                for t in range(0,np.shape(ds_era5_cut['t'])[0]):      
                    ax0.plot(ds_era5_cut["t"][t],ds_era5_cut['level']/1000,lw=lw_thin,color='hotpink')
                    ax1.plot(ds_era5_cut["tprime"][t],ds_era5_cut['level']/1000,lw=lw_thin,color='red')
                ax0.plot(np.mean(ds_era5_cut["t"],axis=0),ds_era5_cut['level']/1000,lw=lw_thick,color='hotpink')
                ax1.plot(np.mean(ds_era5_cut["tprime"],axis=0),ds_era5_cut['level']/1000, lw=lw_thick, color='red')
        else:
            for t in range(0,np.shape(ds['temperature'])[0],4):      
                ax0.plot(ds["temperature"][t],ds['alt_plot'],lw=lw_thin,color='black')
                ax1.plot(plot_vars[k][t],ds['alt_plot'],lw=lw_thin,color='red')
            ax0.plot(np.mean(ds["temperature"],axis=0),ds['alt_plot'],lw=lw_thick,color='black')
            ax1.plot(np.nanmean(plot_vars[k],axis=0),ds['alt_plot'], lw=lw_thick, color='red')
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
    TRES = int(config.get("GENERAL","RESOLUTION").split("Z")[0][2:])
    VRES = int(config.get("GENERAL","RESOLUTION").split("Z")[-1][:-3])
    axes[0,0].text(-0.025, 1.0, "UTC", horizontalalignment='right', verticalalignment='bottom', transform=axes[0,0].transAxes)

    fig.suptitle('          German Aerospace Center (DLR)\n \
    {}, {}\n \
    $\lambda_c$={}$\,$km, T$_c$={}$\,$h \n \
    Resolution: {}$\,$m  x  {}$\,$min'.format(config.get("GENERAL","INSTRUMENT"), config.get("GENERAL","STATION_NAME"), VERTICAL_CUTOFF, TEMPORAL_CUTOFF/60, VRES, TRES))
    ##         ------------------------------\n \

    """Watermark"""
    fig = plt_helper.add_watermark(fig)
    
    """Save figure"""
    fig_name = file_name[:14] + ds.duration_str + '.png'
    fig.savefig(os.path.join(config.get("OUTPUT","FOLDER"),config.get("GENERAL","FILTERTYPE"),fig_name), 
                facecolor='w', edgecolor='w', format='png', dpi=150, bbox_inches='tight') # orientation='portrait'

    """Finish"""
    plt_helper.show_progress(pbar['progress_counter'], pbar['lock'], pbar["stime"], pbar['ntasks'])

    # except:
    #     print("Plot failed for measurement: {}".format(file_name))