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
from tqdm import tqdm as tqdm
import time

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
from plot_lidar_1Dfilter import plot_lidar_1Dfilter
from plot_lidar_stacked_filter import plot_lidar_stacked_filter
from plot_lidar_tmp import plot_lidar_tmp

plt.style.use('latex_default.mplstyle')

"""Config"""
VERTICAL_CUTOFF = 15 # km (LAMBDA_CUT)

# def fwrapper(arg):
#     config = arg[0]
#     if config.get("GENERAL","FILTERTYPE") == "filter-stacked":
#         return plot_lidar_stacked_filter(*arg)
#     elif config.get("GENERAL","FILTERTYPE") == "filter1D":
#         return plot_lidar_1D_filter(*arg)
#     else:
#         return plot_lidar_tmp(*arg)

# def initpool(progress_counter_):
#     global progress_counter
#     progress_counter = progress_counter_

def plot_lidar_data(CONFIG_FILE, filtertype, reset):
    """Visualize lidar measurements (time-height diagrams + absolute temperature measurements)"""
    
    """Settings"""
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    if config.get("INPUT","OBS_FILE") == "NONE":
        obs_list = sorted(glob.glob(os.path.join(config.get("INPUT","OBS_FOLDER") , config.get("GENERAL","RESOLUTION"))))
    else:
        obs_list = os.path.join(config.get("INPUT","OBS_FOLDER"), config.get("INPUT","OBS_FILE"))
    
    config["INPUT"]["ERA5-FOLDER"] = os.path.join(config.get("OUTPUT","FOLDER"),"era5-profiles")

    #FILTERTYPE      = "stacked_filter"
    #FILTERTYPE      = "1Dfilter"
    #FILTERTYPE      = "tmp"
    config["GENERAL"]["FILTERTYPE"] = filtertype
    os.makedirs(os.path.join(config.get("OUTPUT","FOLDER"),config.get("GENERAL","FILTERTYPE")), exist_ok=True)
    fig_list = sorted(glob.glob(os.path.join(config.get("OUTPUT","FOLDER"),config.get("GENERAL","FILTERTYPE"),"*.png")))
    fig_list = [fig_path.split("/")[-1] for fig_path in fig_list]

    progress_counter = mp.Manager().Value('i', 0)
    lock = mp.Manager().Lock()
    stime = time.time()
    pbar = {"progress_counter": progress_counter, "lock": lock, "stime": stime}

    args_list = []
    for obs in obs_list:
        """Check if figure already exists"""
        fig_exists = False
        filename   = obs.split("/")[-1][0:13]
        for figname in fig_list:
            if filename in figname:
                fig_exists = True
                break
        if reset or not fig_exists:
            args = (config, obs, pbar)
            args_list.append(args)

    pbar['ntasks'] = len(args_list)
    config['GENERAL']['NCPUS']  = str(int(mp.cpu_count()-2))
    print(f"[i]  CPUs available: {mp.cpu_count()}")
    print(f"[i]  CPUs used: {config.get('GENERAL','NCPUS')}")
    print(f"[i]  Observations (without duration limit): {pbar['ntasks']}")

    with mp.Pool(processes=config.getint("GENERAL","NCPUS")) as pool:

        if config.get("GENERAL","FILTERTYPE") == "stacked_filter":
            results = pool.starmap(plot_lidar_stacked_filter, args_list)
        elif config.get("GENERAL","FILTERTYPE") == "1Dfilter":
            pool.starmap(plot_lidar_1Dfilter, args_list)
        else:
            pool.starmap(plot_lidar_tmp, args_list)
    
    ### - alternative parallelization with tqdm --- #
    # with mp.Pool(processes=config.getint("GENERAL","NCPUS"), initializer=initpool, initargs=(progress_counter,)) as pool:

    # with mp.Pool(processes=config.getint("GENERAL","NCPUS")) as pool:
    #     results = list(tqdm(pool.imap(fwrapper, args_list), total=len(args_list), leave=False, colour='green')) # chunksize=int(config.getint("GENERAL","NTASKS") / config.getint("GENERAL","NCPUS"))

    etime    = time.time()
    hours    = int((etime - stime) / 3600)
    minutes  = int(((etime - stime) % 3600) / 60)
    seconds  = int(((etime - stime) % 60)) # seconds = round(((etime - stime) % 60), 3)
    time_str = str(hours).zfill(2) + ":" + str(minutes).zfill(2) + ":" + str(seconds).zfill(2)
    print("")
    print(f"[i]  Visualizations completed in {time_str} hours.")


if __name__ == '__main__':
    """provide ini file as argument and pass it to function"""

    """Example: 
        >> python3 plot_lidar_data.py coral.ini tmp true
    """
    
    """Try changing working directory for Crontab"""
    try:
        os.chdir(os.path.dirname(sys.argv[0]))
    except:
        print('[i]  Working directory already set!')
    
    filtertype = sys.argv[2]
    # tmp, 1Dfilter, stacked_filter
    
    reset = False
    if len(sys.argv) > 3:
        if sys.argv[3].lower().capitalize() == "True":
            reset = True
    plot_lidar_data(sys.argv[1], filtertype, reset)