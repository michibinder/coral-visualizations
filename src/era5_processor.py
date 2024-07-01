################################################################################
# Copyright 2023 German Aerospace Center                                       #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

import numpy as np
import pandas as pd
import xarray as xr

"""Constants"""
# omega = 7.292*10**(-5)
g = 9.80665
Rd = 287.06
Re = 6371229 # m (Radius of Earth for GRIB2 format - applies to ERA5 on ML)
p0 = 101325


def processing_data_for_jetexit_comp(config,ds,ds_pv,ds_2pvu):
    """Further processing of ERA5 data"""

    ds['th'] = ds['t'] * (p0/ds['p'])**(2/7)
    
    # - N^2 - #
    ds['N2'] = (['time','level','latitude','longitude'], g/ds['th'].values * np.gradient(ds['th'].values, ds['level'].values[1]-ds['level'].values[0], axis=1))
    # i=np.arange(1,137)
    # ds['N2'] = ds['th'].copy()
    # ds['N2'][:,0,:,:] = np.nan
    # ds['N2'][:,i,:,:] = g/ds['th'][:,i,:,:] * (ds['th'][:,i,:,:].values-ds['th'][:,i-1,:,:].values)/(ds['level'][:,i,:,:].values-ds['level'][:,i-1,:,:].values)

    ##############################
    # - Absolute vorticity - #
    # PV = (rel_vo_isentropic_surf + f) * (-g dth / dp)
    # ds['f'] = 2* omega * np.sin(ds.latitude.expand_dims(dim={'time':dims[0],'level':137,'longitude':dims[3]},axis=[0,1,3])) # planetary vorticity
    # ds['abs_vo'] = (ds['f'] + ds['vo']) #  * 10**(6)

    # - PV (Ertel) and dth/dp - #
    # i=np.arange(1,137)
    # ds['dthdp'] = ds['th'].copy()
    # ds['dthdp'][:,0,:,:] = np.nan
    # ds['dthdp'][:,i,:,:] = (ds['th'][:,i,:,:].values-ds['th'][:,i-1,:,:].values)/(ds['p'][:,i,:,:].values-ds['p'][:,i-1,:,:].values)
    # ds['pv'] = ds['abs_vo'] * -g * ds['dthdp'] * 10**(6)
    ##############################
    
    # ds['tprime_m'] = ds['tprime'].where(abs(ds['tprime'])>1)
    ds['u_horiz'] = (ds['u']**2 + ds['v']**2)**(1/2)

    ds_pv['u_horiz'] = (ds_pv['u']**2 + ds_pv['v']**2)**(1/2)
    # ds_pv['u_horiz'] = (ds_pv['u']**2 + ds_pv['v']**2)**(1/2)
    # ds_pv['div_u'] = ds_pv['u'].differentiate(coord='longitude') + ds_pv['v'].differentiate(coord='latitude')

    if config.getboolean("ERA5","WESTERN_COORDS"):
        ds['longitude_plot']      = ds['longitude'] - 360
        ds_pv['longitude_plot']   = ds_pv['longitude'] - 360
        ds_2pvu['longitude_plot'] = ds_2pvu['longitude'] - 360
        #ds_pv['longitude_plot']   = (["longitude"], ds['longitude'].values - 360)
        #ds_2pvu['longitude_plot'] = (["longitude"], ds['longitude'].values - 360)
        
    else:
        ds['longitude_plot']      = ds['longitude']
        ds_pv['longitude_plot']   = ds_pv['longitude']
        ds_2pvu['longitude_plot'] = ds_2pvu['longitude']

    dims_pv = np.shape(ds_pv['pv'])
    ds_pv['longitude_3d'] = ds_pv.longitude_plot.expand_dims(dim={'time':dims_pv[0],'level':dims_pv[1],'latitude':dims_pv[2]},axis=[0,1,2])
    ds_pv['latitude_3d']  = ds_pv.latitude.expand_dims(dim={'time':dims_pv[0],'level':dims_pv[1],'longitude':dims_pv[3]},axis=[0,1,3])

    return ds,ds_pv,ds_2pvu


def prepare_interpolated_ml_ds(file_ml,file_ml_T21,file_ml_coeff,file_ml_int):
    """"Open files"""
    ml_coeff = pd.read_csv(file_ml_coeff)
    # engine="netcdf4"
    with xr.open_dataset(file_ml) as ds:
        with xr.open_dataset(file_ml_T21) as ds_T21:
            """Interpolate the model level dataset to a regular grid and combine with T21 (filtered) dataset"""
            lnsp = ds['lnsp'][:,0,:,:].drop_vars('level').expand_dims(dim={'level':138}, axis=1)
            dims = np.shape(lnsp)

            a = xr.DataArray(ml_coeff['a [Pa]'])
            a = a.rename({'dim_0':'level'})
            a = a.expand_dims(dim={'time':dims[0],'latitude':dims[2],'longitude':dims[3]},axis=[0,2,3])

            b = xr.DataArray(ml_coeff['b'])
            b = b.rename({'dim_0':'level'})
            b = b.expand_dims(dim={'time':dims[0],'latitude':dims[2],'longitude':dims[3]},axis=[0,2,3])

            # - Pressure at half levels - # 
            p_half = a + b * np.exp(lnsp.values)

            """Compute geopotential and geometric height"""
            ds = compute_z_level(ds, p_half)
            ds['geom_height'] = Re * ds['geop_height'] / (Re-ds['geop_height'])

            # - Calculate pressure on model levels - #
            i=np.arange(0,137)
            ds['p'] = ds['t'].copy()
            ds['p'][:,i,:,:] = (p_half[:,i+1,:,:].values + p_half[:,i,:,:].values) / 2
            
            # - Calculate T' - #
            ds['tprime'] = ds['t']-ds_T21['t']

            """Interpolate data to aequidistant vertical grid"""
            z_new = np.linspace(0,70,176) * 1000
            # z_new   = np.linspace(0,80,161) * 1000
            vars    = ['t','p','u','v','tprime']
            alt_var = 'geop_height' # 'geom_height'
            ds = interp_ds_vertically(ds,z_new,alt_var,vars)

            # - Compression - #
            for var_name in vars: # ds.variables
                ds[var_name] = ds[var_name].astype('float32') #'int16'

            ds.to_netcdf(file_ml_int)


def compute_z_level(ds, p_half):
    """Compute z at half & full level for the given level, based on t/q/sp"""
    # https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height

    # - Get surface geopotential - #
    z_h = ds['z'][:,0,:,:].copy()
    
    # compute moist temperature
    ds['t_moist'] = ds['t'] * (1. + 0.609133 * ds['q'])
 
    # compute the pressures (on half-levels)
    i=np.arange(0,137)

    for i in range(136,0,-1):
        ph_levplusone  = p_half[:,i+1,:,:].values
        ph_lev         = p_half[:,i,:,:].values

        dlog_p = np.log(ph_levplusone / ph_lev)
        alpha = 1. - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)

        t_level = ds['t_moist'][:,i,:,:] * Rd
 
        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the full level
        # z_f = z_h + (t_level * alpha)
        ds['z'][:,i,:,:] = z_h + (t_level * alpha)
    
        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h = z_h + (t_level * dlog_p)
 
    i = 0
    ph_levplusone   = p_half[:,i+1,:,:].values
    ph_lev          = p_half[:,i,:,:].values
    
    dlog_p = np.log(ph_levplusone / 0.1)
    alpha = np.log(2)

    t_level = ds['t_moist'][:,i,:,:] * Rd
    ds['z'][:,i,:,:] = z_h + (t_level * alpha)

    # calculate geopotential height!!
    ds['geop_height'] = ds['z'] / g
    
    return ds


def interp_ds_vertically(ds,z_new,alt_var,vars):
    """Interpolate model levels to aequdistant grid for applying filters"""
    for var in vars:
        shape = np.shape(ds[vars[0]].values)
        data  = np.zeros((shape[0],len(z_new),shape[2],shape[3]))
        for t in range(0,shape[0]):
            ## print(var, ': ',t, end='\r')
            for lat in range(0,shape[2]):
                for lon in range(0,shape[3]):
                    data[t,:,lat,lon] = np.interp(z_new,ds[alt_var].values[t,::-1,lat,lon],ds[var].values[t,::-1,lat,lon])
        if var==vars[0]:
            ds_new = xr.Dataset({var: (['time','level','latitude','longitude'],data,ds[var].attrs)},
                                coords={'time'     : ds['time'],
                                        'level'    : z_new,
                                        'latitude' : ds['latitude'],
                                        'longitude': ds['longitude']},
                                attrs=ds.attrs)
        else:
            ds_new[var] = (['time','level','latitude','longitude'],data,ds[var].attrs)
    return ds_new