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

def plot_era5_jet_composition(config, vars, ds, ds_ml, ds_pv, ds_2pvu, ds_saamer, t):
    """Multiple horizontal cross sections (xy) at different altitudes and zonal/vertical cross sections (xz)"""

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
    vert_range = [6,61]
    vert_range_lid = [6,95]

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
    gs_top  = fig.add_gridspec(5,4, hspace=0.03, wspace=0.03, height_ratios=[1.12,1,1,1,1.2], width_ratios=[2.5,9,2.5,1])
    ax_wind = fig.add_subplot(gs_top[0])
    ax_lid  = fig.add_subplot(gs_top[1])
    ax_lid2 = fig.add_subplot(gs_top[2])

    """Levels and colormaps"""
    levels1 = [32000,24000,16000]
    levels2 = [56000,48000,40000]
    h_str1 = ['z: {}km'.format(int(levels1[0]/1000)),'z: {}km'.format(int(levels1[1]/1000)),'z: {}km'.format(int(levels1[2]/1000))]
    h_str2 = ['z: {}km'.format(int(levels2[0]/1000)),'z: {}km'.format(int(levels2[1]/1000)),'z: {}km'.format(int(levels2[2]/1000))]
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
    ##clev_lin = [-10,-9,-8,-7,-6,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10]
    clev_lin = [-16,-14,-12,-10,-8,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,8,10,12,14,16]
    blue = 'mediumblue'
    red  = 'firebrick'
    clev_colors = [blue,blue,blue,blue,blue,blue,blue,blue,blue,blue,blue,red,red,red,red,red,red,red,red,red,red,red]

    # - Theta / U / Pressure levels - #
    thlev = np.arange(220,600,5)
    thlev_labels = np.arange(220,600,40)
    # thlev_st = np.exp(5+0.03*np.arange(1,120,4))
    thlev_st = np.exp(5+0.03*np.arange(1,120,2))
    thlev_st_labels = np.exp(5+0.03*np.arange(1,120,4))

    ulev = np.arange(30,180,10)

    # pressure_levels = np.arange(0,1000,1) * 100
    # pressure_levels = np.exp(np.arange(-2,50,0.05))
    # pressure_levels = np.logspace(-1,5,500)
    pressure_levels = np.logspace(-1,6,1000)

    # geop_levels = np.arange(7500,15000,70) # maybe choose small range here
    geop_levels = np.arange(7500,60000,60) # maybe choose small range here
    
    wind_levels = np.arange(35,80,5)
    cmap = plt.get_cmap('Greens') # Greys, Greens
    norm = BoundaryNorm(boundaries=wind_levels , ncolors=cmap.N, clip=True)
    wind_levels_vert = np.arange(50,130,10)
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
    
    """Wind (top left panel) ERA5 and SAAMER"""
    cwindu = 'royalblue'
    cwindv = 'skyblue' # 'darkturquoise'
    csaameru = 'darkslateblue'
    csaamerv = 'violet'

    ax_wind.plot(np.mean(ds_ml['u'].sel(latitude=lat,longitude=lon_eastern),axis=0),ds_ml['level']/1000,lw=lw_medium,ls='--', color=cwindu)
    ax_wind.plot(np.mean(ds_ml['v'].sel(latitude=lat,longitude=lon_eastern),axis=0),ds_ml['level']/1000,lw=lw_medium,ls='--', color=cwindv)
    era5u = ax_wind.plot(ds_ml['u'].sel(latitude=lat,longitude=lon_eastern)[t],ds_ml['level']/1000,lw=lw_thick,color=cwindu, label="ERA5 u")
    era5v = ax_wind.plot(ds_ml['v'].sel(latitude=lat,longitude=lon_eastern)[t],ds_ml['level']/1000,lw=lw_thick,color=cwindv, label="ERA5 v")
    if ds_saamer is not None:
        saameru = ax_wind.plot(ds_saamer['zonal_wind'][:,t],ds_saamer['altitude'],lw=lw_thick,color=csaameru, label="SAAMER u")
        saamerv = ax_wind.plot(ds_saamer['meridional_wind'][:,t],ds_saamer['altitude'],lw=lw_thick,color=csaamerv, label="SAAMER v")

    ax_wind.yaxis.set_minor_locator(AutoMinorLocator()) 
    ax_wind.xaxis.set_minor_locator(AutoMinorLocator())
    ax_wind.tick_params(which='both', labelbottom=False,labeltop=True)
    ax_wind.xaxis.set_label_position('top')
    ax_wind.set_xlabel('(u,v) / m$\,$s$^{-1}$')
    ax_wind.set_ylabel('altitude / km')
    ax_wind.set_xlim([-140,140])
    ax_wind.set_ylim(vert_range_lid)
    ax_wind.legend(loc='lower left', fontsize=7, handlelength=1, framealpha=0.5) # [era5u,era5v,saameru,saamerv], ['ERA5 u', 'ERA5 v', 'SAAMER u', 'SAAMER v']
    ax_wind.grid()

    """(a) Lidar plot (first row)"""
    cont_lid = ax_lid.contour(ds_ml['time'].values, ds_ml['level']/1000, vars["tprime_vlidar_tbwf"], levels=clev_contour, colors='k', linewidths=0.4) # BW

    # --- MEASUREMENT DATA (CORAL/...) --- #
    contf_lid = ax_lid.contourf(ds.time.values, ds.alt_plot, ds["tprime_tbwf"].T.values,
                                levels=clev, cmap=cmap_st, norm=norm_st, extend='both') # negative_linestyles='dashed'

    hlocator = mdates.HourLocator(byhour=range(0,24,3))
    ax_lid.xaxis.set_major_locator(hlocator)
    ax_lid.xaxis.set_major_formatter(plt.FuncFormatter(plt_helper.timelab_format_func))
    ##h_fmt = mdates.DateFormatter('%b-%d %H:%M')
    ##ax_lid.xaxis.set_major_formatter(h_fmt)
    ax_lid.yaxis.set_minor_locator(AutoMinorLocator()) 
    ax_lid.xaxis.set_minor_locator(AutoMinorLocator())
    
    ax_lid.xaxis.set_label_position('top')
    ax_lid.tick_params(which='both', labelbottom=False,labeltop=True, labelleft=False)
    ax_lid.set_ylim(vert_range_lid)
    ax_lid.set_xlim(ds_ml['time'].values[0],ds_ml['time'].values[-1])
    time_ticks = ax_lid.get_xticks()
    ax_lid.set_xticks(time_ticks[1::])
    ## ax_lid.set_aspect(0.05)
    ax_lid.text(0.04, 1.0, "UTC", horizontalalignment='right', verticalalignment='bottom', transform=ax_lid.transAxes)
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

    ypp = 0.91
    ax_wind.text(0.09, ypp, "a", transform=ax_wind.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ax_lid.text(0.02, ypp, "b", transform=ax_lid.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ax_lid2.text(0.09, ypp, "c", transform=ax_lid2.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    """(c-h) horizontal cross sections"""
    for j in range(0,3):
        ax_tpj = axes[j+1,0]
        ax_jet = axes[j+1,1]
        tmp_level1 = levels1[j]
        tmp_level2 = levels2[j]

        # --- TPJ Jet --- #
        if j==2:
            ax_tpj.contour(ds_pv.longitude_plot, ds_pv.latitude, ds_pv['z'].sel(level=met_level)[t,:,:]/g, colors='dimgray', levels=geop_levels, linewidths=lw_wind)
            contf_wind = ax_tpj.contourf(ds_pv.longitude_plot, ds_pv.latitude, ds_pv['u_horiz'].sel(level=met_level)[t,:,:], cmap=cmap, norm=norm, levels=wind_levels,extend='both')
        else:
            ax_tpj.contour(ds_ml.longitude_plot, ds_ml.latitude, ds_ml['p'].sel(level=tmp_level1)[t,:,:],colors='dimgray', levels=pressure_levels, linewidths=lw_wind)
            ax_tpj.contourf(ds_ml.longitude_plot, ds_ml.latitude, ds_ml['u_horiz'].sel(level=tmp_level1)[t,:,:], cmap=cmap_vert, norm=norm_vert, levels=wind_levels_vert,extend='both')
        ax_tpj.contour(ds_ml.longitude_plot,  ds_ml.latitude, ds_ml['tprime'][t,:,:,:].sel(level=tmp_level1), colors=clev_colors, levels=clev_lin, linewidths=lw_medium, extend='both')

        ax_tpj.axhline(lat, color='black', ls='--', lw=lw_cut)
        ax_tpj.axvline(lon, color='black', ls='--', lw=lw_cut)

        ax_tpj.coastlines(lw=0.5)
        gls = ax_tpj.gridlines(draw_labels=True, x_inline=False, y_inline=False)
        gls.right_labels=False
        gls.top_labels=False
        if j!=3:
            gls.bottom_labels=False

        # --- JET (right side) --- #
        cont_p   = ax_jet.contour(ds_ml.longitude_plot, ds_ml.latitude, ds_ml['p'].sel(level=tmp_level2)[t,:,:],colors='dimgray', levels=pressure_levels, linewidths=lw_wind)
        ax_jet.contourf(ds_ml.longitude_plot, ds_ml.latitude, ds_ml['u_horiz'].sel(level=tmp_level2)[t,:,:], cmap=cmap_vert, norm=norm_vert, levels=wind_levels_vert,extend='both')
        ax_jet.contour(ds_ml.longitude_plot,  ds_ml.latitude, ds_ml['tprime'][t,:,:,:].sel(level=tmp_level2), colors=clev_colors, levels=clev_lin, linewidths=lw_medium, extend='both')

        ax_jet.axhline(lat-5, color='black', ls='--', lw=lw_cut)

        ax_jet.coastlines(lw=0.5)
        gls = ax_jet.gridlines(draw_labels=True, x_inline=False, y_inline=False)
        gls.left_labels=False
        gls.right_labels=False
        gls.top_labels=False
        if j!=3:
            gls.bottom_labels=False
    
        # - Numbering - #
        # - Labels - #
        numb_str = ['d','e','f','g','h','i','j','k']
        ypp = 0.89
        xpp = 0.033
        ax_tpj.text(xpp, ypp, numb_str[2*j], transform=ax_tpj.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ax_jet.text(xpp, ypp, numb_str[2*j+1], transform=ax_jet.transAxes, weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ax_tpj.text(xpp, 0.07, h_str1[j], transform=ax_tpj.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ax_jet.text(xpp, 0.07, h_str2[j], transform=ax_jet.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})


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
    # ax_jet.set_xlim([ds_2pvu.longitude[0],ds_2pvu.longitude[-1]])
    # ax_jet.set_ylim([-77.5,-32.5])
    fig.subplots_adjust(bottom=0, top=0.95, left=0, right=1,
                    wspace=0.04, hspace=0.04)
    

    """Vertical cross sections"""
    for ii in range(0,2):
        if ii==0: 
            axb = axb1
            lat_temp = lat
            levels = levels1
        else:
            axb = axb2
            lat_temp = lat-5
            levels = levels2

        ## Tprime
        # nx_avg = 50 # 50 -> approx. lambdax=750km assuming 1°=60km, 60->900km
        # tprime_lon_z, tprime_lat_z = era_filter.horizontal_temp_filter(ds,t,lat,lon, nx_avg=nx_avg)
        tprime_lon_z_T21 = ds_ml['tprime'][t,:,:,:].sel(latitude=lat_temp,method='nearest').values

        # contf_tprime = axb1.contourf(ds['longitude'], ds['level']/1000, tprime_lon_z_T21, levels=clev, cmap=cmap_st, norm=norm_st, extend='both')
        axb.contour(ds_ml.longitude_plot, ds_ml['level']/1000, tprime_lon_z_T21, colors=clev_colors, levels=clev_lin, linewidths=lw_medium, extend='both') # lw=0.8

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

        axb.xaxis.set_major_formatter(plt_helper.major_formatter_lon)
        axb.xaxis.set_label_position('bottom') 
        axb.tick_params(which='both', top=True, labelbottom=True,labeltop=False)

        # - clabels theta - #
        # th_y_pos = np.linspace(13,30,6)
        # th_label_lon = []
        # for lab in th_y_pos:
        #     th_label_lon.append((lon-30,lab))
        # axb.clabel(cont_th0, fmt= '%1.0fK', inline=True, fontsize=9, manual=th_label_lon)


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

    # - 300hPa wind - #
    cbar = fig.colorbar(contf_wind, ax=axes[1:3,2], location='right', fraction=1, shrink=0.8, aspect=27)
    cbar.set_label('horizontal wind at 300hPa / ms$^{-1}$')
    cb_position = cbar.ax.get_position()
    # cbar.ax.set_position([cb_position.x0+0.003, cb_position.y0-0.055, cb_position.width, cb_position.height])

    # - Absolute wind - #
    cbar = fig.colorbar(contf_wind_vert, ax=axes[3:5,2], location='right', fraction=1, shrink=0.8, aspect=29)
    cbar.set_label('horizontal wind / ms$^{-1}$')
    
    # timestamp_str = datetime.datetime.strftime(pd.to_datetime(str(ds['time'][t].values)), format='%Y-%m-%d %H:%M')

    """Watermark"""
    fig = plt_helper.add_watermark(fig)
    
    fig_name = 'era5_jet_' + '{:02d}'.format(t) + '.png'
    fig.savefig(os.path.join(config.get("OUTPUT","FOLDER_PLOTS"),fig_name),
                facecolor='w', edgecolor='w', format='png', dpi=150, bbox_inches='tight') # orientation='portrait'