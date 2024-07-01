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
import multiprocessing as mp
import time
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
from plot_era5_tropopause_composition import plot_era5_tropopause_composition
from plot_era5_jet_pvu_composition import plot_era5_jet_pvu_composition
from plot_era5_jet_composition import plot_era5_jet_composition

plt.style.use('latex_default.mplstyle')

"""Config"""
VERTICAL_CUTOFF = 15 # km (LAMBDA_CUT)
TEMPORAL_CUTOFF = 8*60 # min (TAU_CUT)
PVU_z_levels = [3,4,5,6,7,8,9,10,11,12,13]
saamer_file_path = "/export/data/SAAMER/SAAMER_Hindley22_version_2018.nc"

def prepare_era5_composition(config_file, content, reset):
    """Visualize ERA5 composition of virtual lidar measurement,
    zonal & meridional cross sections in stratosphere and at tropopause level"""
    
    """Settings"""
    config = configparser.ConfigParser()
    config.read(config_file)
    config["ERA5"]["g"] = "9.80665"

    if config.get("INPUT","OBS_FILE") == "NONE":
        obs_list = sorted(glob.glob(os.path.join(config.get("INPUT","OBS_FOLDER") , config.get("GENERAL","RESOLUTION"))))
    else:
        obs_list = os.path.join(config.get("INPUT","OBS_FOLDER"), config.get("INPUT","OBS_FILE"))
    
    config["INPUT"]["ERA5-FOLDER"] = os.path.join(config.get("OUTPUT","FOLDER"),"era5-region")
    os.makedirs(config.get("INPUT","ERA5-FOLDER"), exist_ok=True)
    era5_region_list = sorted(glob.glob(os.path.join(config.get("INPUT","ERA5-FOLDER"),"*pvu.nc")))
    era5_region_list = [path.split("/")[-1] for path in era5_region_list]

    config["GENERAL"]["CONTENT"] = content
    if reset:
        shutil.rmtree(os.path.join(config.get("OUTPUT","FOLDER"),config.get("GENERAL","CONTENT")), ignore_errors=True)
    os.makedirs(os.path.join(config.get("OUTPUT","FOLDER"),config.get("GENERAL","CONTENT")), exist_ok=True)
    out_list = sorted(glob.glob(os.path.join(config.get("OUTPUT","FOLDER"),config.get("GENERAL","CONTENT"),"*.mp4")))
    out_list = [fig_path.split("/")[-1] for fig_path in out_list]
    
    progress_counter = mp.Manager().Value('i', 0)
    lock = mp.Manager().Lock()
    stime = time.time()
    pbar = {"progress_counter": progress_counter, "lock": lock, "stime": stime}

    print(f"[i]  Collecting observations with available ERA5 data...")
    args_list = []
    for obs in obs_list:
        filename = os.path.split(obs)[-1][0:13]
        """Check if ERA5 data exists"""
        if filename + "-pvu.nc" not in era5_region_list:
            continue

        """Check if animation already exists"""
        animation_exists = False
        for animation in out_list:
            if filename in animation:
                animation_exists = True
                break
        
        """Check year (manually)"""
        if filename[0:4] != "2018":
            animation_exists = True

        if reset or not animation_exists:
            args = (config, obs, pbar)
            args_list.append(args)

    pbar['ntasks'] = len(args_list)
    config['GENERAL']['NCPUS'] = str(int(mp.cpu_count()-2))
    # config['GENERAL']['NCPUS'] = "1"
    print(f"[i]  CPUs available: {mp.cpu_count()}")
    print(f"[i]  CPUs used: {config.get('GENERAL','NCPUS')}")
    print(f"[i]  Distributed tasks (observations with regional ERA5 data): {pbar['ntasks']}")

    with mp.Pool(processes=config.getint("GENERAL","NCPUS")) as pool:
        pool.starmap(plot_era5_composition, args_list)


def plot_era5_composition(config, obs, pbar):
    file_name = os.path.split(obs)[-1]
    ds = lidar_processor.open_and_decode_lidar_measurement(obs)
    if ds is None:
        """Finish"""
        plt_helper.show_progress(pbar['progress_counter'], pbar['lock'], pbar["stime"], pbar['ntasks'])
        return

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
    animation_path = os.path.join(config.get("OUTPUT","FOLDER"), config.get("GENERAL","CONTENT"), animation_name)

    """Process lidar data and load ERA5 data for plot"""
    era5_files_name = os.path.join(config["INPUT"]["ERA5-FOLDER"], file_name[0:13])
    if not os.path.exists(era5_files_name + '-ml-int.nc'):
        print(f"[i]  Missing ERA5 data for measurement {file_name}")
    elif not os.path.exists(era5_files_name + '-pl.nc'):
        print(f"[i]  Missing ERA5 data for measurement {file_name}")
    elif not os.path.exists(era5_files_name + '-pvu.nc'):
        print(f"[i]  Missing ERA5 PVU data for measurement {file_name}")
    else:
        ds      = lidar_processor.process_lidar_measurement(config, ds)
        ds      = lidar_processor.calculate_primes(ds, TEMPORAL_CUTOFF, VERTICAL_CUTOFF)
        ds_ml   = xr.open_dataset(era5_files_name + '-ml-int.nc')
        ds_pv   = xr.open_dataset(era5_files_name + '-pl.nc')
        ds_2pvu = xr.open_dataset(era5_files_name + '-pvu.nc')
        ds_ml,ds_pv,ds_2pvu = era5_processor.processing_data_for_jetexit_comp(config,ds_ml,ds_pv,ds_2pvu)
        preprocessed_vars = vlidar_and_latlon_slices(config,ds_ml,ds_pv)

        if config.get("GENERAL","CONTENT") == "era5-tropo":
            print("plotting")
            """Plot ERA5 tropopause dynamics composition"""
            config["OUTPUT"]["FOLDER_PLOTS"] = os.path.join(config.get("OUTPUT","FOLDER"), config.get("GENERAL","CONTENT"), file_name[0:13])
            os.makedirs(config.get("OUTPUT","FOLDER_PLOTS"), exist_ok=True)
            tstep = 23
            plot_era5_tropopause_composition(config, preprocessed_vars, ds, ds_ml, ds_pv, ds_2pvu, tstep)
            # for tstep in range(np.shape(ds_ml['t'])[0]):
            #     plot_era5_tropopause_composition(config, preprocessed_vars, ds, ds_ml, ds_pv, ds_2pvu, tstep)
            #     print("Plot: {}".format(tstep), end="\r")

        elif config.get("GENERAL","CONTENT") == "era5-jet-pvu":
            """Plot ERA5 jet regions and 2PVU level in mainly horizontal cross sections"""
            config["OUTPUT"]["FOLDER_PLOTS"] = os.path.join(config.get("OUTPUT","FOLDER"), config.get("GENERAL","CONTENT"), file_name[0:13])
            os.makedirs(config.get("OUTPUT","FOLDER_PLOTS"), exist_ok=True)
            tstep = 23
            plot_era5_jet_pvu_composition(config, preprocessed_vars, ds, ds_ml, ds_pv, ds_2pvu, tstep)
            # for tstep in range(np.shape(ds_ml['t'])[0]):
            #     plot_era5_jet_pvu_composition(config, preprocessed_vars, ds, ds_ml, ds_pv, ds_2pvu, tstep)
            #     print("Plot: {}".format(tstep), end="\r")
        
        elif config.get("GENERAL","CONTENT") == "era5-jet":
            """Plot ERA5 jet regions and 2PVU level in mainly horizontal cross sections"""
            config["OUTPUT"]["FOLDER_PLOTS"] = os.path.join(config.get("OUTPUT","FOLDER"), config.get("GENERAL","CONTENT"), file_name[0:13])
            os.makedirs(config.get("OUTPUT","FOLDER_PLOTS"), exist_ok=True)

            """SAAMER data"""
            ds_saamer = None
            if config["GENERAL"]["INSTRUMENT"] == "CORAL" and os.path.exists(saamer_file_path):
                ds_saamer = xr.open_dataset(saamer_file_path)
                base_date = datetime.datetime(2018,1,1,0,0)
                datetime_index = [base_date + datetime.timedelta(days=int(ts), minutes=(ts-int(ts))*24*60) for ts in ds_saamer['time'].values]
                ds_saamer = ds_saamer.assign_coords(time=datetime_index)
                ds_saamer = ds_saamer.sel(time=slice(ds_ml.time[0],ds_ml.time[-1]))
                if len(ds_saamer.time) != len(ds_ml.time):
                    ds_saamer = None
            # tstep = 23
            # plot_era5_jet_composition(config, preprocessed_vars, ds, ds_ml, ds_pv, ds_2pvu, ds_saamer, tstep)
            for tstep in range(np.shape(ds_ml['t'])[0]):
                plot_era5_jet_composition(config, preprocessed_vars, ds, ds_ml, ds_pv, ds_2pvu, ds_saamer, tstep)
                print("Plot: {}".format(tstep), end="\r")

        """Closing files and generating animation"""
        ds.close()
        ds_ml.close()
        ds_pv.close()
        ds_2pvu.close()

        create_animation(config.get("OUTPUT","FOLDER_PLOTS"), animation_path)
        shutil.rmtree(config.get("OUTPUT","FOLDER_PLOTS"), ignore_errors=True)

    """Finish"""
    plt_helper.show_progress(pbar['progress_counter'], pbar['lock'], pbar["stime"], pbar['ntasks'])


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
    vres = (ds_ml['level'].values[1]-ds_ml['level'].values[0]) / 1000 # km
    tres = 60 # min 
    vars["t_vlidar"]           = ds_ml['t'].sel(latitude=lat,longitude=lon_eastern).values.T.copy()
    vars["tprime_T21"]         = ds_ml['tprime'].sel(latitude=lat,longitude=lon_eastern).values.T.copy()
    vars["tprime_vlidar_tbwf"], tbg_BW = filter.butterworth_filter(vars["t_vlidar"], cutoff=1/TEMPORAL_CUTOFF, fs=1/tres, order=5, mode='both')
    tprime_vlidar_vbwf, tbg_BW = filter.butterworth_filter(vars["t_vlidar"].T, cutoff=1/VERTICAL_CUTOFF, fs=1/vres, order=5, mode='both')
    vars["tprime_vlidar_vbwf"] = tprime_vlidar_vbwf.T
    # vars["tprime_12hmean"]     = vars["t_vlidar"] - vars["t_vlidar"].rolling(time=12,center=True).mean()
    return vars


def create_animation(png_folder, output_path):
    """Create animation (mp4) from pngs"""

    ## pip install imageio[ffmpeg]
    filenames    = sorted(os.listdir(png_folder))
    fps          = 4
    ## fps          = 10
    macro_block_size = 16 # Default is 16 for optimal compatibility

    # Increase the probesize to give FFmpeg more data to estimate the rate
    # writer_options = {'ffmpeg_params': ['-probesize', '10000000']}  # Increase probesize to 5MB
    writer_options = {'ffmpeg_params': ['-probesize', '5000000', '-analyzeduration', '5000000']}

    with imageio.get_writer(output_path, fps=fps, macro_block_size=macro_block_size, **writer_options) as writer:
        for filename in filenames:
            if filename.endswith(".png"):
                image = imageio.imread(os.path.join(png_folder, filename))
                resized_image = resize_to_macro_block(image, macro_block_size)
                writer.append_data(resized_image)
    # imageio.mimsave(image_folder + "/era5_sequence.gif", images, duration=1/fps, palettesize=256/2)  # loop=0, quantizer="nq", palettesize=256


def resize_to_macro_block(image, macro_block_size):
    """Function to make image dimensions divisible by macro block size"""
    height, width = image.shape[:2]
    new_height = (height + macro_block_size - 1) // macro_block_size * macro_block_size
    new_width = (width + macro_block_size - 1) // macro_block_size * macro_block_size
    if (new_height != height) or (new_width != width):
        image = np.pad(image, ((0, new_height - height), (0, new_width - width), (0, 0)), 'constant')
    return image


if __name__ == '__main__':
    """provide ini file as argument and pass it to function"""
    """Try changing working directory for Crontab (change in CRON Job with cd ... as first call)"""
    try:
        os.chdir(os.path.dirname(sys.argv[0]))
    except:
        print('[i]  Working directory already set!')

    config_file = sys.argv[1]
    content = sys.argv[2]
    reset = False
    if len(sys.argv) > 3:
        if sys.argv[3].lower().capitalize() == "True":
            reset = True
    prepare_era5_composition(config_file, content, reset)
