import sys
import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import warnings

from LidarT_visualization import plot_LidarT, plot_nightlyMeans

DEBUG_MODE = 0
if DEBUG_MODE == 0:
    warnings.filterwarnings("ignore")

SETTINGS_FILE = 'settings.txt'
# SETTINGS_FILE_2 = 'settings_default.txt'

def setup_file_list():
        """
        generates three lists:
            - list of available files
            - list of dates of type string
            - list of dates of type datetime
        """
        file_list = []
        date_list = []
        date_type_list =[]
        
        date_plotted = SETTINGS['DATE_PLOTTED'].replace("-","")
        print('Only plot temperature profiles for dates after {}'.format(SETTINGS['DATE_PLOTTED']))
        
        file_paths = sorted(glob.glob(os.path.join(SETTINGS['FILELOCATION'], '*.nc')))
        
        for file in file_paths:
            file = file.rpartition('/')[-1]
            file_list.append(file)
            
            # generate list of available dates (plus time)
            # file_date = file.rpartition('_')[0]
            file_date = file.partition('_')[0]
            if file_date in date_list:
                continue
            elif file_date <= date_plotted: # check if date is before DATE_PLOTTED in settings file
                continue
            else:
                date_list.append(file_date)
            
        date_type_list = [datetime.datetime.strptime(i, '%Y%m%d-%H%M') for i in date_list]
        
        # for i, date in enumerate(date_list):
        
        return file_list, date_list, date_type_list, date_plotted
    


def setup_file_list_of_plot_folders():
    file_paths_tmp = sorted(glob.glob(os.path.join(SETTINGS['PLOT_FOLDER'], '*.png')))
    file_paths_bwf = sorted(glob.glob(os.path.join(SETTINGS['PLOT_FOLDER_BWF'], '*.png')))
    
    file_list_tmp = []
    for file in file_paths_tmp:
        file = file.rpartition('/')[-1]
        file_list_tmp.append(file)
        
    file_list_bwf = []
    for file in file_paths_bwf:
        file = file.rpartition('/')[-1]
        file_list_bwf.append(file)
    
    return file_list_tmp, file_list_bwf
        
        
def load_settings():
    global SETTINGS
    SETTINGS = {}
    with open(SETTINGS_FILE, 'r') as file:
        for line in file:
            try:
                line = line.strip()
                (key, val) = line.split(": ")
                SETTINGS[key] = val
            except:
                print('The following line could not be executed: ' + line)
                print('Variable might be missing!')


def update_settings_file(last_date_plotted):
        # update file location and file name if required
        
        # update last date plotted
        SETTINGS['DATE_PLOTTED'] = last_date_plotted
        
        # save dict to txt file
        with open(SETTINGS_FILE, 'w') as file:
            for key in SETTINGS:
                file.write(key + ': ' + SETTINGS[key] + '\n')

    
if __name__ == '__main__':
    # Check provided system arguments
    args = sys.argv[1:]
    load_settings()
    n = 10000
    delete_current_plots = 0
    
    if delete_current_plots == 1:
        for f in os.listdir(SETTINGS['PLOT_FOLDER']):
            os.remove(os.path.join(SETTINGS['PLOT_FOLDER'], f))
        for f in os.listdir(SETTINGS['PLOT_FOLDER_BWF']):
            os.remove(os.path.join(SETTINGS['PLOT_FOLDER_BWF'], f))
                               
    if len(args) >= 1:
        if os.path.exists(args[0]):
            SETTINGS['FILELOCATION']=args[0]
            print('Data foulder found!')
            SETTINGS['STARTFILE'] = sorted(glob.glob(os.path.join(SETTINGS['FILELOCATION'], '*.nc')))[0].rpartition('/')[-1]
        else:
            print('Data foulder NOT found!')
            sys.exit()
        
        if len(args) >= 2:
            print('Too many system arguments!')
    
    # .nc files used for plotting
    file_list, date_list, date_type_list, date_plotted = setup_file_list()
    
    # List of files/plots that already exist
    file_list_tmp, file_list_bwf = setup_file_list_of_plot_folders()
    
    # stats_path = SETTINGS['STATSLOCATION'] + '/' + date + '_' + TRES + ZRES + '_stats.nc'        
    # date = '20200103-0210' # '%Y%m%d-%H%M'
    TRES = 'T20' # {'T15', 'T20', 'T30', 'T60', ...}
    ZRES = 'Z900' # Z1900
    
    # Plot temperature for all dates
    i = 0
    last_date_plotted = date_plotted
    print('Plotting temperature profiles for {} dates...'.format(len(date_list)))
    print('(Limit set to {})'.format(n))
    for date in date_list:
        # check if plots for this date already exist -> delete
        for file in file_list_tmp:
            if date == file.partition('_')[0]:
                try:
                    os.remove(os.path.join(SETTINGS['PLOT_FOLDER'],file))
                except:
                    print('{} most likely already deleted!'.format(file))
        for file in file_list_bwf:
            if date == file.partition('_')[0]:
                try:
                    os.remove(os.path.join(SETTINGS['PLOT_FOLDER_BWF'],file))
                except:
                    print('{} most likely already deleted!'.format(file))
        
        if date > last_date_plotted:
            last_date_plotted = date
        file_name = date + '_' + TRES + ZRES + '.nc'
        file_name_BWF = date + '_' + TRES + ZRES + '_BWF.nc'
        # print('Date: {}'.format(date))
        
        try:
            fig = plot_LidarT(file_location = SETTINGS['FILELOCATION'], file_name=file_name, SETTINGS=SETTINGS, plot_content='tmp', fixed_timeframe=True, stats_content='nightly_mean', stats_path='', save_fig=True)
            fig.clear()
            plt.close(fig)
            fig = plot_LidarT(file_location = SETTINGS['FILELOCATION'], file_name=file_name, SETTINGS=SETTINGS, plot_content='butterworthF', fixed_timeframe=True, stats_content='nightly_mean', stats_path='', save_fig=True)
            fig.clear()
            plt.close(fig)
        except:
            print('Measurements {} could not be plotted!'.format(date))
            
        i = i+1
        print('.', end ="")
        if i % 10 == 0:
            print(i)
        if i == n:
            break
    
    last_date_plotted_datetype = datetime.datetime.strptime(last_date_plotted, '%Y%m%d-%H%M')
    
    last_date_plotted = datetime.datetime.strftime(last_date_plotted_datetype, '%Y-%m-%d')
    
    # Plots the last day again... to make sure all plots of the respective date are plotted (simple way)
    update_settings_file(last_date_plotted)
    
    #### PLOT NIGHTLY MEAN ####
    fig = plot_nightlyMeans(file_location = SETTINGS['FILELOCATION_NIGHTLY_MEANS'], SETTINGS=SETTINGS, save_fig=True)
    try:
        # fig = plot_nightlyMeans(file_location = SETTINGS['FILELOCATION_NIGHTLY_MEANS'], SETTINGS=SETTINGS, save_fig=True)
        fig.clear()
        plt.close(fig)
    except:
        print('Nightly Means could not be plotted!')
