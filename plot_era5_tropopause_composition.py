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

import filter, cmaps, era5_processor, lidar_processor, plt_helper

plt.style.use('latex_default.mplstyle')

"""Config"""
PVU_z_levels = [3,4,5,6,7,8,9,10,11,12,13]


def plot_era5_tropopause_composition(config, vars, ds, ds_ml, ds_pv, ds_2pvu, t):
    """Visualize single timestamp for animation of ERA5 composition"""

    lidar_label = config.get("GENERAL","INSTRUMENT")
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
    # nx_avg = 56 # 50 -> approx. lambdax=800km assuming 1°=60km, 56->900km
    # tprime_lon_z, tprime_lat_z = filter.horizontal_temp_filter(ds_ml,t,lat,lon_eastern,nx_avg=nx_avg)
    
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
    pcolor_lid = ax_lid.contour(ds_ml['time'].values, ds_ml['level']/1000, vars["tprime_vlidar_tbwf"], levels=clev_contour, colors='k', linewidths=0.4) # BW

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
    pcolor_1 = ax_lid.contourf(ds.time.values, ds.alt_plot, ds["tprime_tbwf"].T.values, 
                                levels=clev, cmap=cmap_st, norm=norm_st, extend='both') # negative_linestyles='dashed'
    
    # --- MEASUREMENT DATA (CORAL/...) --- #
    hlocator   = mdates.HourLocator(byhour=range(0,24,3))
    ax_lid.xaxis.set_major_locator(hlocator)
    ax_lid.xaxis.set_major_formatter(plt.FuncFormatter(plt_helper.timelab_format_func))
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
    ax_lid2.plot(np.mean(vars["t_vlidar"],axis=1),ds_ml['level']/1000, lw=lw_medium, ls="--", color='black')
    ax_lid2.plot(vars["t_vlidar"][:,t],ds_ml['level']/1000,lw=lw_thick,color='black')
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
    axes[k,0].xaxis.set_major_formatter(plt_helper.major_formatter_lon)
    axes[k,1].xaxis.set_major_formatter(plt_helper.major_formatter_lat)
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

    """Watermark"""
    fig = plt_helper.add_watermark(fig)

    fig_name = 'era5_trop_comp_' + '{:02d}'.format(t) + '.png'
    fig.savefig(os.path.join(config.get("OUTPUT","FOLDER_PLOTS"),fig_name),
                facecolor='w', edgecolor='w', format='png', dpi=150, bbox_inches='tight') # orientation='portrait'