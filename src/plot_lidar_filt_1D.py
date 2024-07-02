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

def plot_lidar_filt_1D(config, obs, pbar):
    """Plot lidar data with vertical BW and temporal BW filter (only one filter applied for each plot)."""

    file_name = os.path.split(obs)[-1]

    zrange = eval(config.get("GENERAL","ALTITUDE_RANGE"))
    trange = eval(config.get("GENERAL","TRANGE"))
    ds = lidar_processor.open_and_decode_lidar_measurement(obs)
    if ds is None:
        return
    ds = lidar_processor.process_lidar_measurement(config, ds)
    ds = lidar_processor.calculate_primes(ds, TEMPORAL_CUTOFF, VERTICAL_CUTOFF)
    vars = [None, ds["tprime_tbwf"].values, ds["tprime_vbwf"].values]

    """ERA5 and SAAMER data for plot"""
    era5_file_path = os.path.join(config["INPUT"]["ERA5-FOLDER"], file_name[0:13] + "-ml-int.nc")
    plot_era5 = False
    plot_saamer = False
    if os.path.exists(era5_file_path):
        ds_era5 = xr.open_dataset(era5_file_path)
        if config.getboolean("ERA5","WESTERN_COORDS"):
            lon = config.getfloat("ERA5","LON") + 360
        else:
            lon = config.getfloat("ERA5","LON")
        # ds_era5 = ds_era5.sel(latitude=-53.75,longitude=292.25) # method='nearest'
        ds_era5 = ds_era5.sel(latitude=config.getfloat("ERA5","LAT"),longitude=lon)
        tprime_era5_T21  = ds_era5['tprime'].values
        # tprime_era5_temp = (ds_era5["t"]-ds_era5["t"].rolling(time=10,center=True).mean()).values
        vert_res_era5    = (ds_era5['level'][1]-ds_era5['level'][0]).values / 1000 # km
        temp_res_era5    = 60 # min # (ds_era5['time'][1]-ds_era5['time'][0]).values
        tprime_era5_bwf15, tbg15 = filter.butterworth_filter(ds_era5["t"].values, cutoff=1/VERTICAL_CUTOFF, fs=1/vert_res_era5, order=5, mode='both')
        tprime_era5_bwf_time, tbg_time = filter.butterworth_filter(ds_era5["t"].values.T, cutoff=1/TEMPORAL_CUTOFF, fs=1/temp_res_era5, order=5, mode='both')
        plot_era5 = True

        """SAAMER data"""
        saamer_file_path = "/export/data/SAAMER/SAAMER_Hindley22_version_2018.nc"
        if config["GENERAL"]["INSTRUMENT"] == "CORAL" and os.path.exists(saamer_file_path):
            ds_saamer = xr.open_dataset(saamer_file_path)
            base_date = datetime.datetime(2018,1,1,0,0)
            datetime_index = [base_date + datetime.timedelta(days=int(ts), minutes=(ts-int(ts))*24*60) for ts in ds_saamer['time'].values]
            ds_saamer = ds_saamer.assign_coords(time=datetime_index)
            # ds_saamer = ds_saamer.sel(time=slice(ds.start_time_utc,ds.end_time_utc))
            ds_saamer = ds_saamer.sel(time=slice(ds_era5.time[0],ds_era5.time[-1])) 
            if len(ds_saamer.time) > 2:
                plot_saamer = True

    """Figure"""
    gskw = {'hspace':0.04, 'wspace':0.03, 'width_ratios': [2,6,2], 'height_ratios': [5,5,5,1]} #  , 'width_ratios': [5,5]}
    fig, axes = plt.subplots(4,3, figsize=(9,12), sharey=True, gridspec_kw=gskw)

    axes[3,0].axis('off')
    axes[3,1].axis('off')
    axes[3,2].axis('off')

    h_fmt      = mdates.DateFormatter('%H')
    hlocator   = mdates.HourLocator(byhour=range(0,24,2))
    filter_str =['T-T$_{T21}$ (ERA5)','T < T$_{c}=8\,h$','$\lambda$ < $\lambda_{c}=15\,km$']

    clev = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32] # 32
    clev_contour = [-32,-16,-8,-4,-2,-1,1,2,4,8,16,32] 
    clev_l = [-16,-4,-1,1,4,16]
    cmap = cmaps.get_wave_cmap()
    norm = BoundaryNorm(boundaries=clev, ncolors=cmap.N, clip=True)

    wind_contours = np.arange(-120,135,15)

    for k in range(0,3):
        ax_wind = axes[k,0]
        ax_lid  = axes[k,1]
        ax0     = axes[k,2]
        
        """Temperature contours"""
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
                ax_lid.contour(ds_era5['time'].values, ds_era5['level']/1000, tprime_era5_bwf_time, levels=clev_contour,
                            colors='k', linewidths=0.3)
        else: # k==2
            pcolor0 = ax_lid.pcolormesh(ds.time.values, ds.alt_plot.values, np.matrix.transpose(vars[k]),
                            cmap=cmap, norm=norm)
            if plot_era5:
                ax_lid.contour(ds_era5['time'].values, ds_era5['level']/1000, tprime_era5_bwf15.T, levels=clev_contour,
                            colors='k', linewidths=0.3)
        
        """Saamer contours"""
        if plot_saamer:
                ax_lid.contour(ds_saamer['time'].values, ds_saamer['altitude'], ds_saamer['zonal_wind'], levels=wind_contours,
                            colors='darkslateblue', negative_linestyles='dashed', linewidths=0.5)
                            

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

        """Wind axes"""
        ypp = 0.96
        lw_thin=0.1
        lw_med=0.4
        lw_thick=2
        cwind = 'royalblue'
        cwind2 = 'skyblue' # 'darkturquoise'
        csaameru = 'darkslateblue'
        csaamerv = 'violet'
        ax_wind.set_xlim([-140,140])
        # ax_wind.set_xticks([0,50,100])
        ax_wind.set_ylabel('altitude / km')
        if k==2:
            ax_wind.set_xlabel('(u,v) / m$\,$s$^{-1}$')
            ax_wind.tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=True,labeltop=False)
        else:
            ax_wind.tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=False,labeltop=False)
        ax_wind.xaxis.set_minor_locator(AutoMinorLocator())
        ax_wind.axvline(x=0,c='grey',ls='--')
        if plot_era5:
            if plot_saamer:
                ax_wind.plot(np.mean(ds_saamer['zonal_wind'],axis=1),ds_saamer['altitude'],lw=lw_thick,color=csaameru, label="SAAMER u")
                ax_wind.plot(np.mean(ds_saamer['meridional_wind'],axis=1),ds_saamer['altitude'],lw=lw_thick,color=csaamerv, label="SAAMER v")
                for jj in range(0,np.shape(ds_saamer["zonal_wind"])[1],3):
                    ax_wind.plot(ds_saamer['zonal_wind'][:,jj],ds_saamer['altitude'],lw=lw_med,color=csaameru)
                    ax_wind.plot(ds_saamer['meridional_wind'][:,jj],ds_saamer['altitude'],lw=lw_med,color=csaamerv)
            ds_era5_cut = ds_era5.sel(time=slice(ds.start_time_utc,ds.end_time_utc))            
            ax_wind.plot(np.mean(ds_era5_cut["u"],axis=0),ds_era5_cut['level']/1000,lw=lw_thick,color=cwind, label="ERA5 u")
            ax_wind.plot(np.mean(ds_era5_cut["v"],axis=0),ds_era5_cut['level']/1000,lw=lw_thick,color=cwind2, label="ERA5 v")
            for jj in range(0,np.shape(ds_era5_cut["u"])[0]):
                ax_wind.plot(ds_era5_cut['u'][jj],ds_era5_cut['level']/1000,lw=lw_thin,color=cwind)
                ax_wind.plot(ds_era5_cut['v'][jj],ds_era5_cut['level']/1000,lw=lw_thin,color=cwind2)
            
            ax_wind.text(0.03, ypp, filter_str[k], transform=ax_lid.transAxes, verticalalignment='top', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
            if k==0:
                ax_wind.legend(loc='lower left', fontsize=7)
        ax_lid.text(0.03, ypp, filter_str[k], transform=ax_lid.transAxes, verticalalignment='top', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})            
        ax_lid.grid()

        """T and T' (third axis)"""
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

        numb_str = ['a','b','c','d','e','f','g','h','i','j','k']
        xpp0 = 0.97
        xpp1 = 0.9
        ax_wind.text(xpp1, ypp, numb_str[3*k], verticalalignment='top', horizontalalignment='right', transform=ax_wind.transAxes, 
                            weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ax_lid.text(xpp0, ypp, numb_str[3*k+1], verticalalignment='top', horizontalalignment='right', transform=ax_lid.transAxes, 
                            weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ax1.text(xpp1, ypp, numb_str[3*k+2], verticalalignment='top', horizontalalignment='right', transform=ax1.transAxes, 
                            weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})


    # - COLORBAR - #
    cbar = fig.colorbar(pcolor0, ax=axes[-1,1], location='bottom', ticks=clev_l, fraction=1, shrink=0.9, aspect=25, extend='both') # aspect=30
    cbar.set_label(r"$T'$ / K")

    axes[0,0].set_ylim(zrange[0],zrange[1])

    """Formatting"""
    TRES = int(config.get("GENERAL","RESOLUTION").split("Z")[0][2:])
    VRES = int(config.get("GENERAL","RESOLUTION").split("Z")[-1][:-3])

    # - Use date of first measurement - #
    axes[0,1].text(-0.015, 1.0, "UTC", horizontalalignment='right', verticalalignment='bottom', transform=axes[0,1].transAxes)

    fig.suptitle('          German Aerospace Center (DLR)\n \
    {}, {}\n \
    ------------------------------\n \
    Resolution: {}$\,$m  x  {}$\,$min'.format(config.get("GENERAL","INSTRUMENT"), config.get("GENERAL","STATION_NAME"), VRES, TRES))

    """Watermark"""
    fig = plt_helper.add_watermark(fig)

    """Save figure"""
    fig_name = file_name[:14] + ds.duration_str + '.png'
    fig.savefig(os.path.join(config.get("OUTPUT","FOLDER"),config.get("GENERAL","CONTENT"),fig_name), 
                facecolor='w', edgecolor='w', format='png', dpi=150, bbox_inches='tight') # orientation='portrait'

    """Finish"""
    plt_helper.show_progress(pbar['progress_counter'], pbar['lock'], pbar["stime"], pbar['ntasks'])

    # except:
    #     print("Plot failed for measurement: {}".format(file_name))