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
import cartopy.crs as ccrs 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

## import imageio
import imageio.v2 as imageio

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import filter, cmaps, era5_processor, lidar_processor

plt.style.use('latex_default.mplstyle')

"""Config"""
duration_threshold = 6
content_folder = "era5-tropo"
#content_folder = "era5-jet"
PVU_z_levels = [3,4,5,6,7,8,9,10,11,12,13]

def timelab_format_func(value, tick_number):
    dt = mdates.num2date(value)
    if dt.hour == 0:
        return "{}\n{}".format(dt.strftime("%Y-%b-%d"), dt.strftime("%H"))
    else:
        return dt.strftime("%H")

def major_formatter_lon(x, pos):
    """Using western coordinates"""
    return "%.f°W" % abs(x)
    ##return "%.f°E" % abs(x)

def major_formatter_lat(x, pos):
    return "%.f°S" % abs(x)


def prepare_era5_composition(CONFIG_FILE):
    """Visualize ERA5 composition of virtual lidar measurement,
    zonal & meridional cross sections in stratosphere and at tropopause level"""
    
    """Settings"""
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    config["ERA5"]["g"] = "9.80665"

    if config.get("INPUT","OBS_FILE") == "NONE":
        obs_list = sorted(glob.glob(os.path.join(config.get("INPUT","OBS_FOLDER") , config.get("GENERAL","RESOLUTION"))))
    else:
        obs_list = os.path.join(config.get("INPUT","OBS_FOLDER"), config.get("INPUT","OBS_FILE"))
    
    config["INPUT"]["ERA5-FOLDER"] = os.path.join(config.get("OUTPUT","FOLDER"),"era5-region")

    os.makedirs(os.path.join(config.get("OUTPUT","FOLDER"),content_folder), exist_ok=True)
    ##zrange = eval(config.get("GENERAL","ALTITUDE_RANGE"))
    ##trange = eval(config.get("GENERAL","TRANGE"))
    
    config['GENERAL']['NTASKS'] = str(int(multiprocessing.cpu_count()-2))
    #config['GENERAL']['NTASKS'] = str(2)
    print("[i]  CPUs available: {}".format(multiprocessing.cpu_count()))
    print("[i]  CPUs used: {}".format(config.get("GENERAL","NTASKS")))
    print("[i]  Observations (without duration limit): {}".format(len(obs_list)))
    procs = []
    sema = multiprocessing.Semaphore(config.getint("GENERAL","NTASKS"))
    ii = 0
    # - Start processes - #
    for ii, obs in enumerate(obs_list):
        sema.acquire()
        proc = multiprocessing.Process(target=plot_era5_composition, args=(ii, config, obs, sema))
        procs.append(proc)
        proc.start()   

    # - Complete processes - #
    for proc in procs:
        proc.join()


def plot_era5_composition(ii,config,obs,sema):
    file_name = os.path.split(obs)[-1]
    ds = lidar_processor.open_and_decode_lidar_measurement(obs)
    
    if ds.duration > datetime.timedelta(hours=duration_threshold):
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
        animation_name = file_name[:14] + duration_str + ".mp4"
        animation_path = os.path.join(config.get("OUTPUT","FOLDER"), content_folder, animation_name)

        if os.path.isfile(animation_path):
            print("[i]  ERA5 animation ALREADY EXISTS for measurement {}".format(file_name))
        else:
            """Process lidar data and load ERA5 data for plot"""
            era5_files_name = os.path.join(config["INPUT"]["ERA5-FOLDER"], file_name[0:13])
            if not os.path.exists(era5_files_name + '-ml-int.nc'):
                print("[i]  Missing ERA5 data for measurement {}".format(file_name))
            else:
                ds = lidar_processor.process_lidar_measurement(config, ds)

                ds_ml   = xr.open_dataset(era5_files_name + '-ml-int.nc')
                ds_pv   = xr.open_dataset(era5_files_name + '-pl.nc')
                ds_2pvu = xr.open_dataset(era5_files_name + '-pvu.nc')
                ds_ml,ds_pv,ds_2pvu = era5_processor.processing_data_for_jetexit_comp(config,ds_ml,ds_pv,ds_2pvu)
                preprocessed_vars = vlidar_and_latlon_slices(config,ds_ml,ds_pv)

                if content_folder == "era5-tropo":
                    """Plot ERA5 tropopause dynamics composition"""
                    config["OUTPUT"]["FOLDER_PLOTS"] = os.path.join(config.get("OUTPUT","FOLDER"), content_folder, file_name[0:13])
                    os.makedirs(config.get("OUTPUT","FOLDER_PLOTS"), exist_ok=True)
                    # tstep = 23
                    # era5_tropopause_composition(config, preprocessed_vars, ds, ds_ml, ds_pv, ds_2pvu, tstep)
                    for tstep in range(np.shape(ds_ml['t'])[0]):
                        era5_tropopause_composition(config, preprocessed_vars, ds, ds_ml, ds_pv, ds_2pvu, tstep)
                        print("Plot: {}".format(tstep), end="\r")

                elif content_folder == "era5-jet":
                    """Plot ERA5 jet regions and 2PVU level in mainly horizontal cross sections"""
                    config["OUTPUT"]["FOLDER_PLOTS"] = os.path.join(config.get("OUTPUT","FOLDER"), content_folder, file_name[0:13])
                    os.makedirs(config.get("OUTPUT","FOLDER_PLOTS"), exist_ok=True)
                    # tstep = 23
                    # era5_jet_composition(config, preprocessed_vars, ds, ds_ml, ds_pv, ds_2pvu, tstep)
                    for tstep in range(np.shape(ds_ml['t'])[0]):
                        era5_jet_composition(config, preprocessed_vars, ds, ds_ml, ds_pv, ds_2pvu, tstep)
                        print("Plot: {}".format(tstep), end="\r")

                """Closing files and generating animation"""
                ds.close()
                ds_ml.close()
                ds_pv.close()
                ds_2pvu.close()

                create_animation(config.get("OUTPUT","FOLDER_PLOTS"), animation_path)
                shutil.rmtree(config.get("OUTPUT","FOLDER_PLOTS"), ignore_errors=True)
                print("[i]  ERA5 animation created for measurement {}".format(file_name))
    sema.release()


def era5_jet_composition(config, vars, ds, ds_ml, ds_pv, ds_2pvu, t):
    """Multiple horizontal cross sections (xy) at different altitudes and zonal/vertical cross sections (xz)"""

    lidar_label = config.get("GENERAL","NAME")
    lat         = config.getfloat("ERA5","LAT")
    lon         = config.getfloat("ERA5","LON")
    if config.getboolean("ERA5","WESTERN_COORDS"):
        lon_eastern = 360+lon
    else:
        lon_eastern = lon
    lon_range   = eval(config.get("ERA5","LON_RANGE"))
    lat_range   = eval(config.get("ERA5","LAT_RANGE"))
    g = config.getfloat("ERA5","g")
    vert_range = [14,61]
    vert_range_lid = [14,95]

    """Figure"""
    projection = ccrs.PlateCarree()
    gskw = {'hspace':0.03, 'wspace':0.03, 'height_ratios': [1.12,1,1,1,1.2], 'width_ratios': [7,7,1]} #  , 'width_ratios': [5,5]}
    fig, axes = plt.subplots(5,3, figsize=(9,13), sharex=True, sharey=True, gridspec_kw=gskw, subplot_kw={'projection': ccrs.PlateCarree()}) # 
    # fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95,
    #                 wspace=0.02, hspace=0.02)
    
    # - Remove last column axes with colorbars - #
    axes[0,2].axis('off')
    axes[1,2].axis('off')
    axes[2,2].axis('off')
    axes[3,2].axis('off')
    axes[4,2].axis('off')

    gs = axes[-1,0].get_gridspec()

    # - Remove underlying axes in last row - #
    for ax in axes[-1,0:2]:
        ax.remove()
    axb1 = fig.add_subplot(gs[-1,0])
    axb2 = fig.add_subplot(gs[-1,1])

    # --- Split top row into two axes --- #
    for ax in axes[0,0:2]:
        ax.remove()
    # ax_lid = fig.add_subplot(gs_top[0,0:2])
    gs_top = fig.add_gridspec(5,3, hspace=0.03, wspace=0.03, height_ratios=[1.12,1,1,1,1.2], width_ratios=[11,3,1])
    ax_lid  = fig.add_subplot(gs_top[0])
    ax_lid2 = fig.add_subplot(gs_top[1])

    """Levels and colormaps"""
    # levels = [40000,30000,20000,12000]
    vert_range = [3,31] # [3,27]

    # levels = [24000,18000,12000], [30000,22000,14000]
    levels = [26000,20000,14000]
    h_str = ['z: {}km'.format(int(levels[0]/1000)),'z: {}km'.format(int(levels[1]/1000)),'z: {}km'.format(int(levels[2]/1000))]
    pvlev = [-4,-3,-2,-1]

    CLEV_16        = [-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16]
    CLEV_16_LABELS = [-16,-8,-4,-2,-1,1,2,4,8,16]
    clev           = CLEV_16
    clev_l         = CLEV_16_LABELS
    clev_contour   = [-16,-8,-4,-2,-1,1,2,4,8,16] # for contours in lidar plot
    cmap_st = cmaps.get_wave_cmap()
    norm_st = BoundaryNorm(boundaries=clev, ncolors=cmap_st.N, clip=True)

    # - Linear levels for contour plots - #
    ##clev_lin = [-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8]
    ##clev_lin = [-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8]
    clev_lin = [-10,-9,-8,-7,-6,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10]
    blue = 'mediumblue'
    red  = 'firebrick'
    clev_colors = [blue,blue,blue,blue,blue,blue,blue,blue,blue,blue,blue,blue,blue,blue,red,red,red,red,red,red,red,red,red,red,red,red,red,red]

    # - Theta / U / Pressure levels - #
    thlev = np.arange(220,600,5)
    thlev_labels = np.arange(220,600,40)
    # thlev_st = np.exp(5+0.03*np.arange(1,120,4))
    thlev_st = np.exp(5+0.03*np.arange(1,120,1))
    thlev_st_labels = np.exp(5+0.03*np.arange(1,120,3))

    ulev = np.arange(30,180,10)

    # pressure_levels = np.arange(0,1000,1) * 100
    # pressure_levels = np.exp(np.arange(-2,50,0.05))
    pressure_levels = np.logspace(-1,5,500)

    geop_levels = np.arange(7500,15000,70) # maybe choose small range here
    
    wind_levels = np.arange(35,80,5)
    cmap = plt.get_cmap('Greens') # Greys, Greens
    norm = BoundaryNorm(boundaries=wind_levels , ncolors=cmap.N, clip=True)
    wind_levels_vert = np.arange(30,110,10)
    cmap_vert = plt.get_cmap('Greys') # Greys
    norm_vert = BoundaryNorm(boundaries=wind_levels_vert, ncolors=cmap_vert.N, clip=True)
    nbarbs = 25
    met_level = 300
    lw_cut    = 2
    lw_wind   = 0.5
    lw_thin   = 0.15
    lw_medium = 0.8
    lw_thick  = 1.5
    # h_fmt = mdates.DateFormatter('%m-%d %H:%M')
    
    """(a) Lidar plot (first row)"""
    cont_lid = ax_lid.contour(ds_ml['time'].values, ds_ml['level']/1000, vars["tprime_vlidar"], levels=clev_contour, colors='k', linewidths=0.4) # BW

    # --- MEASUREMENT DATA (CORAL/...) --- #
    contf_lid = ax_lid.contourf(ds.time.values, ds.alt_plot, np.matrix.transpose(ds["tprime_temp"].values),
                                levels=clev, cmap=cmap_st, norm=norm_st, extend='both') # negative_linestyles='dashed'

    hlocator = mdates.HourLocator(byhour=range(0,24,3))
    ax_lid.xaxis.set_major_locator(hlocator)
    ax_lid.xaxis.set_major_formatter(plt.FuncFormatter(timelab_format_func))
    ##h_fmt = mdates.DateFormatter('%b-%d %H:%M')
    ##ax_lid.xaxis.set_major_formatter(h_fmt)
    ax_lid.yaxis.set_minor_locator(AutoMinorLocator()) 
    ax_lid.xaxis.set_minor_locator(AutoMinorLocator())
    
    ax_lid.xaxis.set_label_position('top')
    ax_lid.tick_params(which='both', labelbottom=False,labeltop=True)
    ax_lid.set_ylabel('altitude / km')
    ax_lid.set_ylim(vert_range_lid)
    ax_lid.set_xlim(ds_ml['time'].values[0],ds_ml['time'].values[-1])
    time_ticks = ax_lid.get_xticks()
    ax_lid.set_xticks(time_ticks[1::])
    ## ax_lid.set_aspect(0.05)
    ax_lid.text(-0.011, 1.0, "UTC", horizontalalignment='right', verticalalignment='bottom', transform=ax_lid.transAxes)
    ax_lid.grid()
    
    """(b) Temperature profiles (mean plus timestamp)"""
    ax_lid2.plot(np.mean(vars["t_vlidar"],axis=0),ds_ml['level']/1000, lw=lw_medium, ls="--", color='black')
    ax_lid2.plot(vars["t_vlidar"][t,:],ds_ml['level']/1000,lw=lw_thick,color='black')
    ##for tt in range(0,np.shape(ds_ml['t'])[0],3):      
    ##    ax_lid2.plot(data_temp[tt,:],ds_ml['level']/1000,lw=lw_thin,color='black')
    
    # --- CORAL --- #
    ax_lid2.plot(np.mean(ds["temperature"],axis=0), ds.alt_plot, lw=lw_medium, ls="--", color='firebrick')
    ax_lid2.plot(ds["temperature"].sel(time=ds_ml.time[t],method="nearest").values, ds.alt_plot, lw=lw_thick, color='firebrick')

    ax_lid2.yaxis.set_minor_locator(AutoMinorLocator()) 
    ax_lid2.xaxis.set_minor_locator(AutoMinorLocator())
    ax_lid2.xaxis.set_label_position('top')
    ax_lid2.tick_params(which='both', labelleft=False, labelbottom=False,labeltop=True)
    ax_lid2.set_ylim(vert_range_lid)
    ax_lid2.set_xlim([160,295])
    ax_lid2.xaxis.set_major_formatter("{x:.0f}K")
    ax_lid2.grid()

    timestamp_str = str(ds_ml.time[t].values)[0:16].replace('T',' ')
    ax_lid.text(0.98, 0.955, timestamp_str, transform=ax_lid.transAxes, verticalalignment='top', horizontalalignment='right', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ax_lid.axvline(ds_ml['time'].values[t], color='black', ls='-', lw=lw_cut)

    ypp = 0.91
    ax_lid.text(0.02, ypp, "a", transform=ax_lid.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ax_lid2.text(0.09, ypp, "b", transform=ax_lid2.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    """(c-h) horizontal cross sections"""
    for j in range(0,3):
        ax_tpj = axes[j+1,0]
        ax_pvu = axes[j+1,1]
        tmp_level = levels[j]

        # --- TPJ Jet --- #
        cont_z   = ax_tpj.contour(ds_pv.longitude_plot, ds_pv.latitude, ds_pv['z'].sel(level=met_level)[t,:,:]/g, colors='dimgray', levels=geop_levels, linewidths=lw_wind)
        contf_wind  = ax_tpj.contourf(ds_pv.longitude_plot, ds_pv.latitude, ds_pv['u_horiz'].sel(level=met_level)[t,:,:], cmap=cmap, norm=norm, levels=wind_levels,extend='both')

        # Replace everything below threshold with NAN -> transparent??
        cont_tprime  = ax_tpj.contour(ds_ml.longitude_plot,  ds_ml.latitude, ds_ml['tprime'][t,:,:,:].sel(level=tmp_level), colors=clev_colors, levels=clev_lin, linewidths=lw_medium, extend='both')
        # contf_tprime = ax_tpj.contourf(ds.longitude,  ds.latitude, ds['tprime_m'][t,:,:,:].sel(level=tmp_level), cmap=cmap_st, norm=norm_st, levels=clev, extend='both', alpha=1)


        # barbs_met  = ax_tpj.barbs(ds_pv.longitude[::nbarbs], ds_pv.latitude[::nbarbs], ds_pv['u'].sel(level=met_level)[t,::nbarbs,::nbarbs], ds_pv['v'].sel(level=met_level)[t,::nbarbs,::nbarbs], transform=projection, length=5, linewidth=0.6)
        # ax_pvu.clabel(cont_met, fmt= '%1.0fm', inline=True, fontsize=9)

        ax_tpj.axhline(lat, color='black', ls='--', lw=lw_cut)
        ax_tpj.axvline(lon, color='black', ls='--', lw=lw_cut)

        ax_tpj.coastlines(lw=0.5)
        gls = ax_tpj.gridlines(draw_labels=True, x_inline=False, y_inline=False)
        gls.right_labels=False
        gls.top_labels=False
        if j!=3:
            gls.bottom_labels=False

        # --- 2PVU --- #
        contf_pvu = ax_pvu.contourf(ds_2pvu.longitude_plot, ds_2pvu['latitude'], ds_2pvu['z'][t,:,:]/(1000*g), levels=PVU_z_levels, transform=projection, cmap='turbo', alpha=0.85, extend='both') # Spectral_r
        cont_p    = ax_pvu.contour(ds_ml.longitude_plot, ds_ml.latitude, ds_ml['p'].sel(level=tmp_level)[t,:,:],colors='dimgray', levels=pressure_levels, linewidths=lw_wind)

        cont_tprime  = ax_pvu.contour(ds_ml.longitude_plot,  ds_ml.latitude, ds_ml['tprime'][t,:,:,:].sel(level=tmp_level), colors='k', levels=clev_lin, linewidths=lw_medium, extend='both')

        ax_pvu.axhline(lat-5, color='black', ls='--', lw=lw_cut)

        ax_pvu.coastlines(lw=0.5)
        gls = ax_pvu.gridlines(draw_labels=True, x_inline=False, y_inline=False)
        gls.left_labels=False
        gls.right_labels=False
        gls.top_labels=False
        if j!=3:
            gls.bottom_labels=False
    
        # - Numbering - #
        # - Labels - #
        numb_str = ['c','d','e','f','g','h','i','j']
        ypp = 0.89
        xpp = 0.033
        ax_tpj.text(xpp, ypp, numb_str[2*j], transform=ax_tpj.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ax_pvu.text(xpp, ypp, numb_str[2*j+1], transform=ax_pvu.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ax_tpj.text(xpp, 0.07, h_str[j], transform=ax_tpj.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    # - Draw rectangle around GWs above fold - #
    # if plot_rectangles:
    #     # rect0 = Rectangle([126,-61],15,13,linewidth=2.5, linestyle=(0, (1, 1)), edgecolor='black', facecolor='none')
    #     rect0 = FancyBboxPatch([126,-61],15,13, linewidth=2.5, linestyle=(0, (1, 1)), edgecolor='black', boxstyle='round', facecolor='none')
    #     rect3 = FancyBboxPatch([126,-61],15,13, linewidth=2.5, linestyle=(0, (1, 1)), edgecolor='black', boxstyle='round', facecolor='none')
    #     rect1 = copy.copy(rect0)
    #     rect2 = copy.copy(rect0)
    #     axes[2,0].add_patch(rect0)
    #     axes[1,0].add_patch(rect1)
    #     axes[0,0].add_patch(rect2)
    #     axes[2,1].add_patch(rect3)

    # ax_tpj.set_xlim([ds_2pvu.longitude[0],ds_2pvu.longitude[-1]])
    ax_tpj.set_xlim(lon_range)
    ax_tpj.set_ylim(lat_range)
    # ax_pvu.set_xlim([ds_2pvu.longitude[0],ds_2pvu.longitude[-1]])
    # ax_pvu.set_ylim([-77.5,-32.5])
    fig.subplots_adjust(bottom=0, top=0.95, left=0, right=1,
                    wspace=0.04, hspace=0.04)
    

    """Vertical cross sections"""
    for ii in range(0,2):
        if ii==0: 
            axb = axb1
            lat_temp = lat
        else:
            axb = axb2
            lat_temp = lat-5

        ## Tprime
        # nx_avg = 50 # 50 -> approx. lambdax=750km assuming 1°=60km, 60->900km
        # tprime_lon_z, tprime_lat_z = era_filter.horizontal_temp_filter(ds,t,lat,lon, nx_avg=nx_avg)
        tprime_lon_z_T21 = ds_ml['tprime'][t,:,:,:].sel(latitude=lat_temp,method='nearest').values

        # contf_tprime = axb1.contourf(ds['longitude'], ds['level']/1000, tprime_lon_z_T21, levels=clev, cmap=cmap_st, norm=norm_st, extend='both')
        cont_tprime  = axb.contour(ds_ml.longitude_plot, ds_ml['level']/1000, tprime_lon_z_T21, colors=clev_colors, levels=clev_lin, linewidths=lw_medium, extend='both') # lw=0.8

        ## Theta
        cont_th0  = axb.contour(ds_ml.longitude_plot, ds_ml['level']/1000, ds_ml['th'].sel(latitude=lat_temp)[t,:,:], colors='dimgray', levels=thlev_st, linewidths=0.3)
        # axb1.clabel(cont_th0, thlev_labels, fmt= '%1.0fK', inline=True, fontsize=9, manual=th_label_lon)

        ## PVU lines
        contb0 = axb.contour(ds_pv['longitude_3d'][t,:,0,:], ds_pv['z'].sel(latitude=lat_temp)[t,:,:] / (g*1000), ds_pv['pv'].sel(latitude=lat)[t,:,:] * 10**(6), colors=['k', 'k', 'limegreen', 'k'], linestyles='solid', linewidths=[1.5, 1.5, 2.5, 1.5], levels=pvlev)
        axb.clabel(contb0, [-4,-3,-2,-1], fmt= '%1.0f', inline=True)

        ## Wind
        # cont_v  = axb.contour(ds['longitude'], ds['level']/1000, ds['u_horiz'].sel(latitude=lat)[t,:,:], colors='k', levels=ulev, linewidths=0.9, linestyles='--')
        # axb.clabel(cont_v, ulev, fmt= '%1.0f', inline=True, fontsize=9)
        contf_wind_vert  = axb.contourf(ds_ml.longitude_plot, ds_ml['level']/1000, ds_ml['u_horiz'].sel(latitude=lat_temp)[t,:,:], cmap=cmap_vert, norm=norm_vert, levels=wind_levels_vert, alpha=0.95, extend='both')
        # contf_wind_vert  = axb.contourf(ds.longitude_plot, ds['level']/1000, ds['u_horiz'].sel(latitude=lat_temp)[t,:,:], cmap=cmap, norm=norm, levels=wind_levels, extend='both')

        axb.axhline(levels[0]/1000, color='black', ls='--', lw=lw_cut)
        axb.axhline(levels[1]/1000, color='black', ls='--', lw=lw_cut)
        axb.axhline(levels[2]/1000, color='black', ls='--', lw=lw_cut)

        axb.set_ylim(vert_range)
        axb.set_xlim(lon_range)

        axb.yaxis.set_minor_locator(AutoMinorLocator()) 
        axb.xaxis.set_minor_locator(AutoMinorLocator())

        axb.set_xlabel('longitude / °')

        axb.xaxis.set_major_formatter(major_formatter_lon)
        axb.xaxis.set_label_position('bottom') 
        axb.tick_params(which='both', top=True, labelbottom=True,labeltop=False)

        th_y_pos = np.linspace(13,30,6)
        th_label_lon = []
        for lab in th_y_pos:
            th_label_lon.append((lon-30,lab))
        axb.clabel(cont_th0, fmt= '%1.0fK', inline=True, fontsize=9, manual=th_label_lon)


    # - Draw rectangle around GWs above fold - #
    # if plot_rectangles:
    #     rect0 = FancyBboxPatch([126,12],15,17.5,linewidth=2.5, linestyle=(0, (1, 1)), edgecolor='black', boxstyle='round', facecolor='none')
    #     axb1.add_patch(rect0)

    axb1.text(0.033, 0.05, "y: " + str(lat) + "°", transform=axb1.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    axb2.text(0.033, 0.05, "y: " + str(lat-5) + "°", transform=axb2.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    axb1.axvline(lon, color='black', ls='--', lw=lw_cut)
    axb1.set_ylabel('altitude / km')
    # axb1.tick_params(which='both', labelbottom=False,labeltop=False)
    axb2.tick_params(which='both',labeltop=False,labelleft=False)

    j=3
    ypp=0.915
    xpp=0.033
    axb1.text(xpp, ypp, numb_str[-2], transform=axb1.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    axb2.text(xpp, ypp, numb_str[-1], transform=axb2.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    """Colorbars"""
    # cbar = fig.colorbar(contf_st, ax=axes[0:2,2], location='right', ticks=clev_l, shrink=0.6, fraction=1, aspect=25)
    # cbar.set_label(r"$T'$ / K")

    cbar = fig.colorbar(contf_lid, ax=axes[0,2], location='right', ticks=clev_l, shrink=0.95, fraction=1, aspect=21)
    cbar.set_label(r"$T'$ / K")

    # - 2PVU - #
    cbar = fig.colorbar(contf_pvu, ticks=PVU_z_levels, ax=axes[1:3,2], location='right', fraction=1, shrink=0.75, aspect=27)
    cbar.set_label('height of the dynamical tropopause / km')
    cb_position = cbar.ax.get_position()
    cbar.ax.set_position([cb_position.x0, cb_position.y0+0.04, cb_position.width, cb_position.height])

    # - 300hPa wind - #
    cbar = fig.colorbar(contf_wind, ax=axes[2:4,2], location='right', fraction=1, shrink=0.65, aspect=23)
    cbar.set_label('horizontal wind at 300hPa / ms$^{-1}$')
    cb_position = cbar.ax.get_position()
    cbar.ax.set_position([cb_position.x0+0.003, cb_position.y0-0.055, cb_position.width, cb_position.height])

    # - Absolute wind - #
    cbar = fig.colorbar(contf_wind_vert, ax=axes[4,2], location='right', fraction=1, shrink=0.95, aspect=21)
    cbar.set_label('horizontal wind / ms$^{-1}$')
    
    # timestamp_str = datetime.datetime.strftime(pd.to_datetime(str(ds['time'][t].values)), format='%Y-%m-%d %H:%M')

    fig_name = 'era5_horiz_comp_' + '{:02d}'.format(t) + '.png'
    fig.savefig(os.path.join(config.get("OUTPUT","FOLDER_PLOTS"),fig_name),
                facecolor='w', edgecolor='w', format='png', dpi=150, bbox_inches='tight') # orientation='portrait'


def era5_tropopause_composition(config, vars, ds, ds_ml, ds_pv, ds_2pvu, t):
    """Visualize single timestamp for animation of ERA5 composition"""

    lidar_label = config.get("GENERAL","NAME")
    lat         = config.getfloat("ERA5","LAT")
    lon         = config.getfloat("ERA5","LON")
    if config.getboolean("ERA5","WESTERN_COORDS"):
        lon_eastern = 360+lon
    else:
        lon_eastern = lon
    lon_range   = eval(config.get("ERA5","LON_RANGE"))
    lat_range   = eval(config.get("ERA5","LAT_RANGE"))
    g = config.getfloat("ERA5","g")
    vert_range = [14,61]
    vert_range_lid = [14,95]

    # - Cross sections - #
    #data_temp = ds['t'].sel(latitude=lat)[t,:,:].values.copy()# [::-1,:]
    #tprime_zonal, tbg_zonal = era_filter.butterworth_filter(data_temp.T, highcut=1/15, fs=1/(vert_res/1000), order=5)
    #data_temp = ds['t'].sel(longitude=lon)[t,:,:].values.copy()# [::-1,:]
    #tprime_meridional, tbg_meridional = era_filter.butterworth_filter(data_temp.T, highcut=1/15, fs=1/(vert_res/1000), order=5)
    
    """Temperature filter (Gauss 1D horizontal, T21, ...) for stratosphere spatial cross sections"""
    # - Horizontal FFT filter - #
    nx_avg = 56 # 50 -> approx. lambdax=800km assuming 1°=60km, 56->900km
    tprime_lon_z, tprime_lat_z = filter.horizontal_temp_filter(ds_ml,t,lat,lon_eastern,nx_avg=nx_avg)
    
    # - T21 - #
    tprime_lon_z = ds_ml['tprime'][t,:,:,:].sel(latitude=lat).values
    tprime_lat_z = ds_ml['tprime'][t,:,:,:].sel(longitude=lon_eastern).values

    """Figure specs"""
    gskw  = {'hspace':0.06, 'wspace':0.04, 'height_ratios': [4,4,3,0.001,3.5], 'width_ratios': [5,5,1]} #  , 'width_ratios': [5,5]}
    #gskw2 = {'hspace':0.06, 'wspace':0.06, 'height_ratios': [3.5,4,3,0.001,3.5], 'width_ratios': [7.5,2.5,1]} #  , 'width_ratios': [5,5]}
    fig, axes = plt.subplots(5,3, figsize=(10,12.5), gridspec_kw=gskw)

    axes[0,2].axis('off')
    axes[1,2].axis('off')
    axes[2,2].axis('off')
    axes[3,0].axis('off')
    axes[3,1].axis('off')
    axes[3,2].axis('off')
    axes[4,2].axis('off')

    gs_top = axes[0,0].get_gridspec()
    gs = axes[3,0].get_gridspec()

    # --- Remove the underlying axes --- #
    for ax in axes[0,0:2]:
        ax.remove()

    # --- Split top row into two axes --- #
    # ax_lid = fig.add_subplot(gs_top[0,0:2])
    gs_top2 = fig.add_gridspec(5,3, hspace=0.06, wspace=0.04, height_ratios=[4,4,3,0.001,3.5], width_ratios=[8,2,1])
    ax_lid  = fig.add_subplot(gs_top2[0])
    ax_lid2 = fig.add_subplot(gs_top2[1])

    for ax in axes[-1,0:2]:
        ax.remove()

    projection = ccrs.PlateCarree()
    ax_pvu = fig.add_subplot(gs[-1,0:2], projection=projection)
    lw_cut    = 2
    lw_wind   = 0.5
    lw_thin   = 0.15
    lw_medium = 0.8
    lw_thick  = 1.5
    
    fs_clabel = 8.5

    """Levels and Colormaps"""
    thlev_st = np.exp(5+0.03*np.arange(1,120,4))
    thlev_st_labels = np.exp(5+0.03*np.arange(1,120,8))
    lon_label = lon-25
    lat_label = lat-10

    thlev = np.arange(220,600,5)
    thlev_labels = np.arange(220,600,40)

    #ulev = np.arange(-180,180,20)
    ulev1 = np.arange(20,180,20)
    ulev0 = np.arange(-160,0,20)
    ulev = np.concatenate((ulev0,ulev1)) 

    pvlev = [-4,-3,-2,-1]

    CLEV_11 = [-11,-9,-7,-5,-3,-1,1,3,5,7,9,11]
    CLEV_11_LABELS = [-9,-7,-5,-3,-1,1,3,5,7,9]
    # CLEV_32 = [-32,-16,-8,-4,-2,-1,-0.5,-0.25,0.25,0.5,1,2,4,8,16,32]
    # CLEV_32_LABELS = [-16,-4,-1,-0.25,0.25,1,4,16]
    CLEV_16   = [-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16]
    # CLEV_16   = [-16,-8,-4,-2,-1,1,2,4,8,16]
    # CLEV_16_LABELS = [-16,-4,-1,1,4,16]
    CLEV_16_LABELS = [-16,-8,-4,-2,-1,1,2,4,8,16]

    clev         = CLEV_16
    clev_l       = CLEV_16_LABELS
    clev_contour = [-16,-8,-4,-2,-1,1,2,4,8,16]
    cmap_st = cmaps.get_wave_cmap()
    norm_st = BoundaryNorm(boundaries=clev, ncolors=cmap_st.N, clip=True)

    n2lev = np.array([-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5])
    # clev_l = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5]*10**-4
    cmap = plt.get_cmap('coolwarm')
    norm = BoundaryNorm(boundaries=n2lev , ncolors=cmap.N, clip=True)

    """(a) Lidar plot (first row)"""
    ## pcolor_lid = ax_lid.pcolormesh(ds['time'].values, ds['level']/1000, data_temp,cmap='jet', vmin=180,vmax=300) # absolute temperature
    ## contourf_lid = ax_lid.contourf(ds['time'].values, ds['level']/1000, tprime_lid.T, levels=clev, cmap=cmap_st, norm=norm_st, extend='both')
    ##pcolor_lid = ax_lid.pcolormesh(ds_ml['time'].values, ds_ml['level']/1000, vars["tprime_vlidar"], cmap=cmap_st, norm=norm_st, shading='nearest') # BW
    pcolor_lid = ax_lid.contour(ds_ml['time'].values, ds_ml['level']/1000, vars["tprime_vlidar"], levels=clev_contour, colors='k', linewidths=0.4) # BW

    # --- MEASUREMENT DATA (CORAL/...) --- #
    ##clev_coral = [-12,-6,6,12], [-32,-16,-8,8,16,32]
    ##clev_coral = [-16,-14,-12,-10,-8,-6,6,8,10,12,14,16]
    clev_coral = [-10,-5,5,10]
    clev_coral = [-15,-10,-5,5,10,15]
    red = 'firebrick'
    blue = 'royalblue'
    coral_colors = [blue,blue,blue,red,red,red]
    
    cmap_coral = plt.get_cmap('RdBu_r')
    norm_coral = BoundaryNorm(boundaries=clev_coral, ncolors=cmap_coral.N, clip=True) 
    # pcolor_1 = ax_lid.contour(DS.time.values, DS.alt_plot.altitude/1000, np.matrix.transpose(DS.tmp_pert.values),
    #                        levels=clev_coral, cmap=cmap_coral, norm=norm_coral)
    # pcolor_1 = ax_lid.contour(DS.time.values, DS.alt_plot.altitude/1000, np.matrix.transpose(DS.tmp_pert.values),
    #                         levels=clev_coral, colors='k', linewidths=0.67) # negative_linestyles='dashed'
    ##pcolor_1 = ax_lid.contour(ds.time.values, ds.alt_plot.altitude/1000, np.matrix.transpose(ds["tprime_temp"].values),
    ##                            levels=clev_coral, colors='k', linewidths=0.67) # negative_linestyles='dashed'
    pcolor_1 = ax_lid.contourf(ds.time.values, ds.alt_plot, np.matrix.transpose(ds["tprime_temp"].values),
                                levels=clev, cmap=cmap_st, norm=norm_st) # negative_linestyles='dashed'
    
    # --- MEASUREMENT DATA (CORAL/...) --- #

    hlocator   = mdates.HourLocator(byhour=range(0,24,3))
    ax_lid.xaxis.set_major_locator(hlocator)
    ax_lid.xaxis.set_major_formatter(plt.FuncFormatter(timelab_format_func))
    ##h_fmt = mdates.DateFormatter('%b-%d %H:%M')
    ##ax_lid.xaxis.set_major_formatter(h_fmt)
    ax_lid.yaxis.set_minor_locator(AutoMinorLocator()) 
    ax_lid.xaxis.set_minor_locator(AutoMinorLocator())
    
    ax_lid.xaxis.set_label_position('top')
    ax_lid.tick_params(which='both', labelbottom=False,labeltop=True)
    ax_lid.set_ylabel('altitude / km')
    ax_lid.set_ylim(vert_range_lid)
    ax_lid.set_xlim(ds_ml['time'].values[0],ds_ml['time'].values[-1])
    time_ticks = ax_lid.get_xticks()
    ax_lid.set_xticks(time_ticks[1::])
    ## ax_lid.set_aspect(0.05)
    ax_lid.text(-0.011, 1.0, "UTC", horizontalalignment='right', verticalalignment='bottom', transform=ax_lid.transAxes)
    ax_lid.grid()
    
    """(b) Temperature profiles (mean plus timestamp)"""
    ax_lid2.plot(np.mean(vars["t_vlidar"],axis=0),ds_ml['level']/1000, lw=lw_medium, ls="--", color='black')
    ax_lid2.plot(vars["t_vlidar"][t,:],ds_ml['level']/1000,lw=lw_thick,color='black')
    ##for tt in range(0,np.shape(ds_ml['t'])[0],3):      
    ##    ax_lid2.plot(data_temp[tt,:],ds_ml['level']/1000,lw=lw_thin,color='black')
    
    # --- CORAL --- #
    ax_lid2.plot(np.mean(ds["temperature"],axis=0), ds.alt_plot, lw=lw_medium, ls="--", color='firebrick')
    ax_lid2.plot(ds["temperature"].sel(time=ds_ml.time[t],method="nearest").values, ds.alt_plot, lw=lw_thick, color='firebrick')

    ax_lid2.yaxis.set_minor_locator(AutoMinorLocator()) 
    ax_lid2.xaxis.set_minor_locator(AutoMinorLocator())
    ax_lid2.xaxis.set_label_position('top')
    ax_lid2.tick_params(which='both', labelleft=False, labelbottom=False,labeltop=True)
    ax_lid2.set_ylim(vert_range_lid)
    ax_lid2.set_xlim([160,295])
    ax_lid2.xaxis.set_major_formatter("{x:.0f}K")
    ax_lid2.grid()

    timestamp_str = str(ds_ml.time[t].values)[0:16].replace('T',' ')
    ax_lid.text(0.98, 0.955, timestamp_str, transform=ax_lid.transAxes, verticalalignment='top', horizontalalignment='right', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ax_lid.axvline(ds_ml['time'].values[t], color='black', ls='-', lw=lw_cut)

    """(c,d) Stratosphere"""
    k=1
    # - Running mean - #
    # contf_st = axes[k,0].pcolormesh(ds['longitude2'], ds['geom_height'][t,:,0,0]/1000, ds['tmp_runningM'].sel(latitude=-53.75)[t,:,:], cmap=cmap_st, norm=norm_st, shading='nearest')
    # contf_st = axes[k,1].pcolormesh(ds.latitude,  ds['geom_height'][t,:,0,0]/1000, ds['tmp_runningM'].sel(longitude=360-67.75)[t,:,:], cmap=cmap_st, norm=norm_st, shading='nearest')
    
    ### FFT or T21 ###
    contf_st = axes[k,0].contourf(ds_ml.longitude_plot, ds_ml['level']/1000, tprime_lon_z, levels=clev, cmap=cmap_st, norm=norm_st, extend='both')
    contf_st = axes[k,1].contourf(ds_ml.latitude, ds_ml['level']/1000, tprime_lat_z, levels=clev, cmap=cmap_st, norm=norm_st, extend='both')

    # - Butterworth - #
    # contf_st = axes[k,0].contourf(ds['longitude'], ds['level']/1000, tprime_zonal.T, levels=clev, cmap=cmap_st, norm=norm_st, extend='both')
    # contf_st = axes[k,1].contourf(ds.latitude, ds['level']/1000, tprime_meridional.T, levels=clev, cmap=cmap_st, norm=norm_st, extend='both')

    """Theta (tropo)"""
    # cont  = axes[k,0].contour(ds.longitude, ds['geom_height'][t,:,0,0].values/1000, th_lon_z[t,:,:].values, colors='k') # , levels=thlev)
    # cont  = axes[k,1].contour(ds.latitude,  ds['geom_height'][t,:,0,0].values/1000, th_lat_z[t,:,:].values, colors='k', levels=thlev)
    cont_th0  = axes[k,0].contour(ds_ml.longitude_plot, ds_ml['level']/1000, vars["th_lon_z"][t,:,:], colors='dimgray', levels=thlev_st, linewidths=0.3)
    cont_th1  = axes[k,1].contour(ds_ml.latitude,  ds_ml['level']/1000, vars["th_lat_z"][t,:,:], colors='dimgray', alpha=1, levels=thlev_st, linewidths=0.3)
    th_y_pos = np.linspace(10,56,7)
    th_label_lon = []
    th_label_lat = []
    for lab in th_y_pos:
        th_label_lon.append((lon_label,lab))
        th_label_lat.append((lat_label,lab))
    axes[k,0].clabel(cont_th0, fmt= '%1.0fK', inline=True, fontsize=fs_clabel, manual=th_label_lon)
    #axes[k,1].clabel(cont_th1, thlev_st_labels, fmt= '%1.0fK', inline=True, fontsize=9, manual=th_label_lat)
    # ax.clabel(isentropes, thlev[1::], fontsize=8, fmt='%1.0f K', inline_spacing=1, inline=True, 
    #             manual=[(8,ds.zcr[t,10,0,x]), (8,ds.zcr[t,-15,0,x])]) # ha='left', thlev[1::3]

    """Wind u and v (strato)"""
    cont_v  = axes[k,0].contour(ds_ml.longitude_plot, ds_ml['level']/1000, vars["v_lon_z"][t,:,:], colors='k', levels=ulev, linewidths=lw_wind) # linestyles='-'
    cont_u  = axes[k,1].contour(ds_ml.latitude,  ds_ml['level']/1000, vars["u_lat_z"][t,:,:], colors='k', levels=ulev, linewidths=lw_wind)
    #cont_v  = axes[k,0].contour(ds.longitude_plot, ds['level']/1000, UV_lon_z[t,:,:], colors='k', levels=ulev, linewidths=0.9, linestyles='--')
    #cont_u  = axes[k,1].contour(ds.latitude,  ds['level']/1000, UV_lat_z[t,:,:], colors='k', levels=ulev, linewidths=0.9, linestyles='--')
    axes[k,0].clabel(cont_v, ulev, fmt= '%1.0f', inline=True, fontsize=fs_clabel)
    axes[k,1].clabel(cont_u, ulev, fmt= '%1.0f', inline=True, fontsize=fs_clabel)
    # th_y_pos = np.linspace(10,65,8)
    # v_label_lon = []
    # u_label_lat = []
    # for lab in th_y_pos:
    #     v_label_lon.append((lon_label+60,lab))
    #     u_label_lat.append((lat_label,lab))
    # axes[k,0].clabel(cont_v, ulev, fmt= '%1.0f', inline=True, fontsize=9, manual=v_label_lon)
    # axes[k,0].clabel(cont_u, ulev, fmt= '%1.0f', inline=True, fontsize=9, manual=u_label_lat)

    # - Coral position - #
    # axes[k,0].scatter(-67.75,-53.79, transform=ccrs.PlateCarree(), color="black",s=150, marker='x', lw=3)
    axes[k,0].axvline(lon, color='black', ls='--', lw=lw_cut)
    axes[k,1].axvline(lat, color='black', ls='--', lw=lw_cut)

    # - Draw rectangle around GWs above fold - #
    # if plot_rectangle:
    #     rect_width = 20
    #     rect = Rectangle([-98,13],rect_width,43,linewidth=2, linestyle='dotted', edgecolor='black', facecolor='none')
    #     axes[k,0].add_patch(rect)
    
    axes[k,0].set_xlim(lon_range)
    axes[k,1].set_xlim(lat_range)
    axes[k,0].set_ylim(vert_range)
    axes[k,1].set_ylim(vert_range)
    
    # axes[k,0].text(0.73, 0.03, 'lidar', transform=axes[k,0].transAxes, weight='bold', color='black')
    axes[k,0].text(lon+2, 15, lidar_label, weight='bold', color='black')

    """(e,f) TROPOSPHERE"""
    # - Brunt-Vaisala frequency N^2 - #
    contf = axes[k+1,0].contourf(ds_ml.longitude_plot, ds_ml['level']/1000, vars["n2_lon_z"][t,:,:], cmap=cmap, norm=norm, levels=n2lev, extend='both')
    contf = axes[k+1,1].contourf(ds_ml.latitude,  ds_ml['level']/1000, vars["n2_lat_z"][t,:,:], cmap=cmap, norm=norm, levels=n2lev, extend='both')

    """THETA"""
    # cont  = axes[k,0].contour(ds.longitude, ds['geom_height'][t,:,0,0].values/1000, th_lon_z[t,:,:].values, colors='k') # , levels=thlev)
    # cont  = axes[k,1].contour(ds.latitude,  ds['geom_height'][t,:,0,0].values/1000, th_lat_z[t,:,:].values, colors='k', levels=thlev)
    cont_th0  = axes[k+1,0].contour(ds_ml.longitude_plot, ds_ml['level']/1000, vars["th_lon_z"][t,:,:], colors='dimgray', levels=thlev, linewidths=0.3)
    cont_th1  = axes[k+1,1].contour(ds_ml.latitude,  ds_ml['level']/1000, vars["th_lat_z"][t,:,:], colors='dimgray', alpha=1, levels=thlev, linewidths=0.3)
    th_y_pos = np.linspace(6,13,3)
    th_label_lon = []
    th_label_lat = []
    for lab in th_y_pos:
        th_label_lon.append((lon_label,lab))
        th_label_lat.append((lat_label,lab))
    axes[k+1,0].clabel(cont_th0, fmt= '%1.0fK', inline=True, fontsize=fs_clabel, manual=th_label_lon)
    #axes[k+1,1].clabel(cont_th1, thlev_labels, fmt= '%1.0fK', inline=True, fontsize=fs_clabel, manual=th_label_lat)

    """Wind u and v (tropo)"""
    cont_v  = axes[k+1,0].contour(ds_ml.longitude_plot, ds_ml['level']/1000, vars["v_lon_z"][t,:,:], colors='k', levels=ulev, linewidths=lw_wind)
    cont_u  = axes[k+1,1].contour(ds_ml.latitude,  ds_ml['level']/1000, vars["u_lat_z"][t,:,:], colors='k', levels=ulev, linewidths=lw_wind)
    axes[k+1,0].clabel(cont_v, ulev, fmt= '%1.0f', inline=True, fontsize=fs_clabel)
    axes[k+1,1].clabel(cont_u, ulev, fmt= '%1.0f', inline=True, fontsize=fs_clabel)

    """(g) PV (dynamical tropopause in xz section)"""
    # axes[k,0].contour(ds.longitude, ds['geom_height'][t,:,0,0]/1000, pv_lon_z[t,:,:], colors='k', lw=3, levels=pvlev)
    # axes[k,1].contour(ds.latitude,  ds['geom_height'][t,:,0,0]/1000, pv_lat_z[t,:,:], colors='k', lw=3, levels=pvlev)
    cont0 = axes[k+1,0].contour(ds_pv['longitude_3d'][t,:,0,:], vars["zpv_lon_z"][t,:,:], vars["pv_lon_z"][t,:,:], colors=['k', 'k', 'limegreen', 'k'], linestyles='solid', linewidths=[1.5, 1.5, 2.5, 1.5], levels=pvlev)
    cont1 = axes[k+1,1].contour(ds_pv['latitude_3d'][t,:,:,0],  vars["zpv_lat_z"][t,:,:], vars["pv_lat_z"][t,:,:], colors=['k', 'k', 'limegreen', 'k'], linestyles='solid', linewidths=[1.5, 1.5, 2.5, 1.5], levels=pvlev)
    axes[k+1,0].clabel(cont0, [-4,-3,-2,-1], fmt= '%1.0f', inline=True)
    axes[k+1,1].clabel(cont1, [-4,-3,-2,-1], fmt= '%1.0f', inline=True)

    # - Coral position - #
    # axes[k,0].scatter(-67.75,-53.79, transform=ccrs.PlateCarree(), color="black",s=150, marker='x', lw=3)
    axes[k+1,0].axvline(lon, color='black', ls='--', lw=lw_cut)
    axes[k+1,1].axvline(lat, color='black', ls='--', lw=lw_cut)

    # - Draw rectangle around GWs above fold - #
    # if plot_rectangle:
    #     rect = Rectangle([-98,3.6],rect_width,30,linewidth=2, linestyle='dotted', edgecolor='black', facecolor='none')
    #     axes[k+1,0].add_patch(rect)

    axes[k+1,0].set_xlim(lon_range)
    axes[k+1,1].set_xlim(lat_range)
    axes[k+1,0].set_ylim(3,14)
    axes[k+1,1].set_ylim(3,14)

    axes[k,0].set_ylabel('altitude / km')
    axes[k+1,0].set_ylabel('altitude / km')

    for k in range(1,3):
        axes[k,0].yaxis.set_minor_locator(AutoMinorLocator()) 
        axes[k,0].xaxis.set_minor_locator(AutoMinorLocator())
        axes[k,1].yaxis.set_minor_locator(AutoMinorLocator()) 
        axes[k,1].xaxis.set_minor_locator(AutoMinorLocator())
        axes[k,0].tick_params(which='both', labelbottom=False,labeltop=False)
        axes[k,1].tick_params(which='both', labelbottom=False,labeltop=False,labelleft=False)

    # - Axis formatting - #
    k=2
    # axes[k,0].set_xlabel('longitude / °') # change to longitudes, latitude 10$^3$
    # axes[k,1].set_xlabel('latitude / °') # change to longitudes, latitude

    # axes[k,0].xaxis.set_major_formatter(FormatStrFormatter('%.0f°W'))
    axes[k,0].xaxis.set_major_formatter(major_formatter_lon)
    axes[k,1].xaxis.set_major_formatter(major_formatter_lat)
    axes[k,0].xaxis.set_label_position('bottom') 
    axes[k,1].xaxis.set_label_position('bottom')
    axes[k,0].tick_params(which='both', top=True, labelbottom=True,labeltop=False)
    axes[k,1].tick_params(which='both', top=True, labelbottom=True,labeltop=False)

    """2PVU horizontal section and winds 850hPa"""
    geop_levels = np.arange(1000,4000,50)
    nbarbs = 25
    met_level = 700 # hPa
    contf_pvu  = ax_pvu.contourf(ds_2pvu.longitude_plot, ds_2pvu.latitude, ds_2pvu['z'][t,:,:]/(1000*g), levels=PVU_z_levels, transform=projection,cmap='turbo', extend='both') # Spectral_r
    cont_met   = ax_pvu.contour(ds_pv.longitude_plot, ds_pv.latitude, ds_pv['z'].sel(level=met_level)[t,:,:]/g, transform=projection, colors='k', levels=geop_levels, linewidths=0.4)
    barbs_met  = ax_pvu.barbs(ds_pv.longitude_plot[::nbarbs], ds_pv.latitude[::nbarbs], ds_pv['u'].sel(level=met_level)[t,::nbarbs,::nbarbs], ds_pv['v'].sel(level=met_level)[t,::nbarbs,::nbarbs], transform=projection, length=5, linewidth=0.7)
    ax_pvu.clabel(cont_met, fmt= '%1.0fm', inline=True, fontsize=fs_clabel)

    """Lidar location"""
    #ax_pvu.scatter(lon,lat, transform=ccrs.PlateCarree(), facecolor="black",s=120, marker='x', lw=2.5)
    ax_pvu.axvline(lon, color='black', ls='--', lw=lw_cut)
    ax_pvu.axhline(lat, color='black', ls='--', lw=lw_cut)

    ax_pvu.coastlines()
    gls = ax_pvu.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gls.right_labels=False
    gls.top_labels=False
    
    """Labels / Numbering"""
    numb_str = ['a','b','c','d','e','f','g','h']
    k=0
    ypp = 0.9
    ax_lid.text(0.025, ypp, numb_str[0], transform=ax_lid.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ax_lid2.text(0.12, ypp, numb_str[1], transform=ax_lid2.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    k=1
    ypp = 0.92
    axes[k,0].text(0.04, ypp, numb_str[2], transform=axes[k,0].transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    axes[k,1].text(0.04, ypp, numb_str[3], transform=axes[k,1].transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    k=2
    ypp = 0.9
    axes[k,0].text(0.04, ypp, numb_str[4], transform=axes[k,0].transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    axes[k,1].text(0.04, ypp, numb_str[5], transform=axes[k,1].transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    ax_pvu.text(0.04, 0.9, numb_str[6], transform=ax_pvu.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    ax_pvu.set_xlim(lon_range)
    ax_pvu.set_ylim(lat_range)

    """Colorbars"""
    #cbar = fig.colorbar(contf_st, ax=axes[0:2,2], location='right', ticks=clev_l, shrink=0.6, fraction=1, aspect=25)
    cbar = fig.colorbar(contf_st, ax=axes[0:2,2], location='right', ticks=clev_l, shrink=0.7, fraction=1, aspect=30)
    cbar.set_label(r"$T'$ / K")

    cbar = fig.colorbar(contf, ax=axes[2,2], location='right', shrink=0.9, fraction=1, aspect=15)
    cbar.set_label('$N^2$ / 10$^{-4}$ s$^{-2}$')

    cbar = fig.colorbar(contf_pvu, ticks=PVU_z_levels, ax=axes[4,2], location='right', fraction=1, shrink=0.9, aspect=17)
    cbar.set_label('height of the dynamical tropopause / km')

    fig_name = 'era5_trop_comp_' + '{:02d}'.format(t) + '.png'
    fig.savefig(os.path.join(config.get("OUTPUT","FOLDER_PLOTS"),fig_name),
                facecolor='w', edgecolor='w', format='png', dpi=150, bbox_inches='tight') # orientation='portrait'


# ----------------------------------- SUBROUTINES ----------------------------------- #
def vlidar_and_latlon_slices(config,ds_ml,ds_pv):
    """Filtering virtual lidar data (vlidar) and averaging lat lon bands or using slices of data"""
    
    vars = {}
    lat = config.getfloat("ERA5","LAT")
    lon = config.getfloat("ERA5","LON")
    if config.getboolean("ERA5","WESTERN_COORDS"):
        lon_eastern = 360+lon
    else:
        lon_eastern = lon
    lat_avg = eval(config.get("ERA5","LAT_AVG"))
    lon_avg = eval(config.get("ERA5","LON_AVG"))
    g = config.getfloat("ERA5","g")

    vars["th_lon_z"] = ds_ml['th'].sel(latitude=lat)          ## ds['th'].sel(latitude =slice(lat_range[0],lat_range[1])).mean(axis=2)
    vars["th_lat_z"] = ds_ml['th'].sel(longitude=lon_eastern) ## ds['th'].sel(longitude=slice(lon_range[0],lon_range[1])).mean(axis=3)

    vars["n2_lon_z"] = ds_ml['N2'].sel(latitude=lat)*10**4          ## ds_ml['N2'].sel(latitude =slice(lat_avg[0],lat_avg[1])).mean(axis=2)
    vars["n2_lat_z"] = ds_ml['N2'].sel(longitude=lon_eastern)*10**4 ## ds_ml['N2'].sel(longitude=slice(lon_avg[0],lon_avg[1])).mean(axis=3)

    #v_lon_z = ds_ml['v'].sel(latitude =slice(lat_range[0],lat_range[1])).mean(axis=2).copy()
    #u_lat_z = ds_ml['u'].sel(longitude=slice(lon_range[0],lon_range[1])).mean(axis=3).copy()
    vars["v_lon_z"]  = ds_ml['v'].sel(latitude=lat)
    vars["u_lat_z"]  = ds_ml['u'].sel(longitude=lon_eastern)
    vars["UV_lon_z"] = (ds_ml['u'].sel(latitude=lat)**2 + ds_ml['v'].sel(latitude=lat)**2)**(1/2)
    vars["UV_lat_z"] = (ds_ml['u'].sel(longitude=lon_eastern)**2 + ds_ml['v'].sel(longitude=lon_eastern)**2)**(1/2)

    # - Rolling mean - #
    # tmp_mean = ds['t'].rolling(longitude=90, center = True).mean(dim='longitude')
    # ds['tprime_runningM'] = ds['t'].copy()-tmp_mean

    # mean_temp = ds['t'].mean(axis=3).copy()
    # ds['tprime'] = ds['t'].copy() - mean_temp

    ### - PV dataset (on pressure levels) - ###
    ##print(ds_pv)
    # vars["pv_lon_z"] = ds_pv['pv'].sel(latitude=lat)*10**(6)          ##  .sel(latitude =slice(lat_avg[0],lat_avg[1])).mean(axis=2) * 10**(6)
    # vars["pv_lat_z"] = ds_pv['pv'].sel(longitude=lon_eastern)*10**(6) ## .sel(longitude=slice(lon_avg[0],lon_avg[1])).mean(axis=3) * 10**(6)

    # vars["zpv_lon_z"] = ds_pv['z'].sel(latitude=lat)/(g*1000)          ## .sel(latitude =slice(lat_avg[0],lat_avg[1])).mean(axis=2)
    # vars["zpv_lat_z"] = ds_pv['z'].sel(longitude=lon_eastern)/(g*1000) ## .sel(longitude=slice(lon_avg[0],lon_avg[1])).mean(axis=3)

    vars["pv_lon_z"] = ds_pv['pv'].sel(latitude =slice(lat_avg[0],lat_avg[1])).mean(axis=2) * 10**(6)
    vars["pv_lat_z"] = ds_pv['pv'].sel(longitude=slice(lon_avg[0],lon_avg[1])).mean(axis=3) * 10**(6)

    vars["zpv_lon_z"] = ds_pv['z'].sel(latitude =slice(lat_avg[0],lat_avg[1])).mean(axis=2) / (g*1000) 
    vars["zpv_lat_z"] = ds_pv['z'].sel(longitude=slice(lon_avg[0],lon_avg[1])).mean(axis=3) / (g*1000) 

    """Virtual lidar data"""
    vert_res                   = ds_ml['level'].values[1]-ds_ml['level'].values[0]
    vars["t_vlidar"]           = ds_ml['t'].sel(latitude=lat,longitude=lon_eastern).copy()
    #tprime_time = data_temp - data_temp.mean(axis=0)
    vars["tprime_T21"]         = ds_ml['tprime'].sel(latitude=lat,longitude=lon_eastern).values.T.copy()
    vars["tprime_BW"], tbg_BW  = filter.butterworth_filter(vars["t_vlidar"].values, highcut=1/15, fs=1/(vert_res/1000), order=5)
    vars["tprime_12hmean"]     = vars["t_vlidar"] - vars["t_vlidar"].rolling(time=12,center=True).mean()
    #tprime_12hmean = (data_temp-tprime_BW) - (data_temp-tprime_BW).rolling(time=12,center=True).mean() ## Subtract BW filter first, then running mean
    #tprime_lid = tprime_BW.T
    #tprime_lid = tprime_T21
    vars["tprime_vlidar"]      = vars["tprime_12hmean"].T

    return vars


def create_animation(png_folder, output_path):
    """Create animation (mp4) from pngs"""

    ## pip install imageio[ffmpeg]
    filenames    = sorted(os.listdir(png_folder))
    fps          = 4
    ## fps          = 10
    macro_block_size = 16 # Default is 16 for optimal compatibility

    with imageio.get_writer(output_path, fps=fps, macro_block_size=macro_block_size) as writer: # duration=1000*1/fps
        for filename in filenames:
            if filename.endswith(".png"):
                image = imageio.imread(os.path.join(png_folder, filename))
                writer.append_data(image)
    # imageio.mimsave(image_folder + "/era5_sequence.gif", images, duration=1/fps, palettesize=256/2)  # loop=0, quantizer="nq", palettesize=256


if __name__ == '__main__':
    """provide ini file as argument and pass it to function"""
    """Try changing working directory for Crontab (change in CRON Job with cd ... as first call)"""
    try:
        os.chdir(os.path.dirname(sys.argv[0]))
    except:
        print('[i]  Working directory already set!')
    prepare_era5_composition(sys.argv[1])