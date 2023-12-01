import sys
import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt

from LidarT_visualization import plot_LidarT

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
        
        file_paths = sorted(glob.glob(os.path.join(SETTINGS['FILELOCATION'], '*.nc')))
        for file in file_paths:
            file = file.rpartition('/')[-1]
            file_list.append(file)
            
            # generate list of available dates (plus time)
            # file_date = file.rpartition('_')[0]
            file_date = file.partition('_')[0]
            if file_date in date_list:
                continue
            else:
                date_list.append(file_date)
            
        date_type_list = [datetime.datetime.strptime(i, '%Y%m%d-%H%M') for i in date_list]
        
        return file_list, date_list, date_type_list
    

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

    
if __name__ == '__main__':
    # Check provided system arguments
    args = sys.argv[1:]
    
    load_settings()
    
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
    
    file_list, date_list, date_type_list = setup_file_list()
    
    
    # date = '20200103-0210' # '%Y%m%d-%H%M'
    TRES = 'T15' # {'T15', 'T20', 'T30', 'T60', ...}
    ZRES = 'Z900' # Z1900
    
    # stats_path = SETTINGS['STATSLOCATION'] + '/' + date + '_' + TRES + ZRES + '_stats.nc'
    
    # file_name =  '20180118-0253_T10Z900.nc' # '20200103-0210_T10Z900.nc'
    
    # Plot temperature for all dates
    n = 250
    i = 0
    print('Plotting temperature profiles for {} dates...'.format(len(date_list)))
    print('(Limit set to {})'.format(n))
    for date in date_list:
        file_name = date + '_' + TRES + ZRES + '.nc'
        # print('Date: {}'.format(date))
        try:
            fig = plot_LidarT(file_location = SETTINGS['FILELOCATION'], file_name=file_name, SETTINGS=SETTINGS, plot_content='tmp', fixed_timeframe=True, stats_content='nightly_mean', stats_path='', save_fig=True)
            fig.clear()
            plt.close(fig)
        except:
            print('Date {} could not be plotted!'.format(date))
        i = i+1
        print('.', end ="")
        if i % 10 == 0:
            print(i)
        if i == n:
            break
        
    # Plot butterworth filtersfor all dates??
        
