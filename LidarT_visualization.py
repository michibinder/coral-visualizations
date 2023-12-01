import os
import datetime
import numpy as np
import xarray as xr
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

# import matplotlib
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = ['Tahoma']


def plot_LidarT(file_location='../Data', file_name="20191015-0014_T15Z900.nc", SETTINGS=None, plot_content='tmp', fixed_timeframe=False, stats_content='nightly_mean', stats_path='', save_fig=False):
    """
    Input:
        file_location
        file_name
        plot_content
        fixed_timeframe.. boolean value
        plots_stats.. boolean value (show mean temp or statistics of measurement)
        timeframe... vector of [startTime,endTime] without date !as string!
        save_fig.. boolean value
    """
    
    global FIGSIZE
    FIGSIZE = eval(SETTINGS['FIGURE_SIZE'])
    
    path = os.path.join(file_location, file_name)
    
    DS = xr.open_dataset(path, decode_times=False)
    
    DS.assign_coords({'time':DS.time.values / 1000})
    DS.integration_start_time.values = DS.integration_start_time.values / 1000
    DS.integration_end_time.values = DS.integration_end_time.values / 1000
    
    # Decode times with time offset
    unit_str = DS.time_offset.attrs['units']
    DS.attrs['reference_date'] = unit_str[14:-6]
    # Reference date is first reference
    # 'Time offset' is 'seconds' after reference date
    # Time is 'seconds' after time offset
    
    time_reference = datetime.datetime.strptime(DS.reference_date, '%Y-%m-%d %H:%M:%S.%f')
    time_offset = datetime.timedelta(seconds=float(DS.time_offset.values[0]))
    new_time_reference = time_reference + time_offset
    time_reference_str = datetime.datetime.strftime(new_time_reference, '%Y-%m-%d %H:%M:%S')
    
    DS.time.attrs['units'] = 'seconds since ' + time_reference_str
    DS.integration_start_time.attrs['units'] = 'seconds since ' + time_reference_str
    DS.integration_end_time.attrs['units'] = 'seconds since ' + time_reference_str
    
    DS = xr.decode_cf(DS, decode_coords = True, decode_times = True) 
    
    # Date for plotting should always refer to the center of the plot (04:00 UTC)
    timeframe = eval(SETTINGS['FIXED_TIMEFRAME'])
    if fixed_timeframe:
        fixed_start = timeframe[0]
        fixed_end = timeframe[1]
        if fixed_end < fixed_start:
            fixed_intervall = fixed_end + 24 - fixed_start
        else: 
            fixed_intervall = fixed_end - fixed_start
            
        start_date = datetime.datetime.utcfromtimestamp(DS.time.values[0].astype('O')/1e9)
        fixed_start_date = datetime.datetime(start_date.year, start_date.month, start_date.day, fixed_start, 0,0)
        duration = datetime.datetime.utcfromtimestamp(DS.integration_end_time.values[-1].astype('O')/1e9) -  datetime.datetime.utcfromtimestamp(DS.integration_start_time.values[0].astype('O')/1e9)# for calendar
        
        reference_hour = 15
        if (start_date.hour > reference_hour) and (fixed_start_date.hour > reference_hour):
            DS['date_startp'] = fixed_start_date
            DS['date_endp'] = fixed_start_date + datetime.timedelta(hours=fixed_intervall)
        elif (start_date.hour > reference_hour) and (fixed_start_date.hour < reference_hour): # prob in range of 0 to 10
            DS['date_startp'] = fixed_start_date + datetime.timedelta(hours=24)
            DS['date_endp'] = fixed_start_date + datetime.timedelta(hours=24+fixed_intervall)
        elif (start_date.hour < reference_hour) and (fixed_start_date.hour > reference_hour):
            DS['date_startp'] = fixed_start_date - datetime.timedelta(hours=24)
            DS['date_endp'] = fixed_start_date - datetime.timedelta(hours=24-fixed_intervall)
        else: # (start_date.hour < 18) and (fixed_start_date.hour < 18):
            DS['date_startp'] = fixed_start_date
            DS['date_endp'] = fixed_start_date - datetime.timedelta(hours=fixed_intervall)
            
        DS['fixed_timeframe'] = 1
    else:
        DS['fixed_timeframe'] = 0
        
    # Temperature (Change 0 to NaN)
    DS.temperature.values = np.where(DS.temperature == 0, np.nan, DS.temperature)
    DS.temperature_err.values = np.where(DS.temperature_err == 0, np.nan, DS.temperature_err)
    
    #da_tmp = DS.temperature.dropna(dim='time', how="all")
    #da_tmp_err = DS.temperature_err.dropna(dim='time', how="all")
    # da_tmp = DS.temperature
    # da_tmp_err = DS.temperature_err
    # ds_tmp = da_tmp.to_dataset()
    # ds_tmp['temperature_err'] = da_tmp_err

    # Altitude
    DS['alt_plot'] = (DS.altitude + DS.altitude_offset + DS.station_height) / 1000 #km

    if int(SETTINGS['USE_LATEX_FORMAT']):
        set_latex()
        
    if plot_content == 'runningM':
        fig, ax0, ax1 = plot_vertical_runningM_profile(DS, SETTINGS)
    
    elif plot_content == 'butterworthF':
        fig, ax0, ax1 = plot_vertical_butterworthF_profile(DS, SETTINGS)
        
    elif plot_content == 'tmp_err':
        fig, ax0, ax1 = plot_vertical_tmp_err_profile(DS, SETTINGS)
    else:  # fig_content == 'tmp'
        fig, ax0, ax1 = plot_vertical_T_profile(DS, SETTINGS)
    
    fig, ax0, ax1 = second_plot(fig, ax0, ax1, DS, SETTINGS, stats_content, stats_path)
        
    fig = format_plot(fig, ax0, ax1, DS, SETTINGS)
    
    if save_fig:
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        duration_str = ''
        if hours <= 9:
            duration_str = duration_str + '0' + str(int(hours))
        else:
            duration_str = duration_str + str(int(hours))
        if minutes <= 9:
            duration_str = duration_str + '0' + str(int(minutes))
        else:
            duration_str = duration_str + str(int(minutes))
        
        if plot_content=='butterworthF':
            fig_name = file_name[:14] + duration_str + 'h_' + 'bwf' + '.png'
            fig.savefig(SETTINGS['PLOT_FOLDER_BWF'] + '/' + fig_name, facecolor='w', edgecolor='w',
                        format='png') # orientation='portrait'
        else:
            # fig_name = file_name[:-3] + '_' + plot_content + '.png'
            fig_name = file_name[:14] + duration_str + 'h_' + plot_content + '.png'
            fig.savefig(SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                    format='png') # orientation='portrait'
    
    return fig

    
def plot_vertical_T_profile(DS, SETTINGS):
    """ Visualizes vertical temperature profile of provided dataset."""  
    v_range = eval(SETTINGS['TEMP_RANGE'])
    
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=FIGSIZE, gridspec_kw={'height_ratios': [2, 1]});
    pcolor_plot = ax0.pcolormesh(DS.time, DS.alt_plot, np.matrix.transpose(DS.temperature.values),
                             cmap='jet', vmin=v_range[0], vmax=v_range[1]);
    cbar = fig.colorbar(pcolor_plot, ax=ax0)
    cbar.set_label('Temperature (K)')
    
    return fig, ax0, ax1


def plot_vertical_runningM_profile(DS, SETTINGS):
    """ Visualizes vertical temperature profile of provided dataset."""  
    v_range = eval(SETTINGS['TEMP_RANGE_RUNNINGM'])
    
    tmp_mean = DS.temperature.rolling(time=30, center = True).mean(dim='time')
    DS['tmp_runningM'] = DS.temperature-tmp_mean
        
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=FIGSIZE, gridspec_kw={'height_ratios': [2, 1]})
    pcolor_plot = ax0.pcolormesh(DS.time, DS.alt_plot, np.matrix.transpose(DS.tmp_runningM.values),
                             cmap='jet', vmin=v_range[0], vmax=v_range[1])
    cbar = fig.colorbar(pcolor_plot, ax=ax0)
    cbar.set_label('Perturbation (K)')
    
    return fig, ax0, ax1


def plot_vertical_butterworthF_profile(DS, SETTINGS):
    """ Visualizes vertical temperature profile of provided dataset."""
    v_range = eval(SETTINGS['TEMP_RANGE_BUTTERWORTHF'])
    
    DS = butterworthf(DS)
    
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=FIGSIZE, gridspec_kw={'height_ratios': [2, 1]})
    pcolor_plot = ax0.pcolormesh(DS.time.values, DS.alt_plot.values, np.matrix.transpose(DS.tmp_pert.values),
                         cmap='jet', vmin=v_range[0], vmax=v_range[1])
    cbar = fig.colorbar(pcolor_plot, ax=ax0)
    cbar.set_label('Perturbation (K)')
    
    return fig, ax0, ax1


def plot_vertical_tmp_err_profile(DS, SETTINGS):
    """ Visualizes temperature error of provided dataset."""    
    v_range = eval(SETTINGS['TEMP_RANGE_ERROR'])
    
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=FIGSIZE, gridspec_kw={'height_ratios': [2, 1]})
    # im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    pcolor_plot = ax0.pcolormesh(DS.time, DS.alt_plot, np.matrix.transpose(DS.temperature_err.values),
                             cmap='jet', vmin=v_range[0], vmax=v_range[1])
    cbar = fig.colorbar(pcolor_plot, ax=ax0)
    cbar.set_label('Uncertainty (K)')
    
    return fig, ax0, ax1


def butterworthf(DS, highcut=1/20, fs=1/0.1, order=3, single_column_filter=True):
    """butterworth filter applied to matrix or each column seperately
        - uses the signal.butter and signal.lfilter functions of the SCIPY library
        - applies a low pass filter based on the given order and highcut frequency
    Input:
        - DS
        - highcut frequency (1/wavelength) 5 work good 20km?
        - fs (sampling frequency) -> 100m 
        - order of filter = 3
    Output:
        - ds_tmp, which now includes the filtered temperature background and the perturbation
    """
    
    b, a = butter_lowpass(highcut, fs, order=order) 
    # print(b,a)
    # print("filter stable!", np.all(np.abs(np.roots(a))<1))

    if single_column_filter:
        # filter each column (returns matrix with Nans at bottom and top)
        columns_bg = np.full(DS.temperature.values.shape, np.NaN)
        for col, column in enumerate(DS.temperature):
            mask = np.isnan(column) # dataarray
            c_masked = column[~mask] # dataarray
            if len(c_masked) >= 10:
                c_mirrored = np.append(np.flip(c_masked, axis=0), c_masked, axis=0) # numpy array
                # c_filtered = signal.lfilter(b,a,c_mirrored, axis=0)
                c_filtered = signal.filtfilt(b,a,c_mirrored, axis=0, padtype='even') # 'even'
                c_filtered = c_filtered[len(c_masked):]
                column_bg = column.copy()
                column_bg[~mask] = c_filtered
                columns_bg[col,:] = column_bg
            else: # column of NANs is just passed through
                columns_bg[col,:] = column
        DS['tmp_bg'] = (['time', 'altitude'], columns_bg)
    else: # not available @ mom
        da_tmp = DS.temperature
        ds_tmp = da_tmp.to_dataset()
        # filter as matrix (some data is lost at the upper boundary)
        da_tmp = ds_tmp.temperature.dropna(dim='altitude', how="any")
        ds_tmp_2 = da_tmp.to_dataset() # required for original window
        #ds_tmp['tmp_bg'] = ds_tmp.temperature
        tmp_mirrored = np.append(np.flip(ds_tmp_2.temperature, axis=1), ds_tmp_2.temperature, axis=1)
        # tmp_filtered = signal.lfilter(b,a,tmp_mirrored, axis=1)
        tmp_filtered = signal.filtfilt(b,a,tmp_mirrored, axis=1)
        tmp_filtered = tmp_filtered[:,len(da_tmp[0]):]
        ds_tmp_2['tmp_bg'] = (['time', 'altitude'], tmp_filtered)
        ds_tmp['tmp_bg'] = ds_tmp_2['tmp_bg']

    DS['tmp_pert'] = DS.temperature - DS.tmp_bg
    return DS


def butter_lowpass(highcut, fs, order=5):
    """
    defines the butterworth filter coefficient based on 
    sample frequency, cut_off frequency and filter order
    """
    nyq = 0.5 * fs # Nyquist frequency
    # highcut = 1/20
    # lowcut = 1/2000000
    # low_crit = lowcut / nyq
   
    high_crit = highcut / nyq # critical frequency ratio
    # Wn = [low_crit, high_crit] # bandpass
    
    b, a = signal.butter(order, high_crit, btype='low', analog=False)
    return b, a


def second_plot(fig, ax0, ax1, DS, SETTINGS, stats_content, stats_path):
    
    if stats_path == '': # default stats_content == 'nightly_mean'
        ## Mean Plot (altitude vs. temperature)
        x_range = eval(SETTINGS['MEAN_TEMP_RANGE'])
        y_range = eval(SETTINGS['ALTITUDE_RANGE'])
    
        ax1.plot(DS.temperature.mean(axis=0), DS.alt_plot)
        ax1.set_xlim(x_range[0],x_range[1])
        ax1.set_ylim(y_range[0],y_range[1])
        ax1.set_aspect('1.4')
        
        # Labels 
        ax1.set_title('Nightly mean')
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Altitude (km)')
    
    else:
        # Time scale is based on DS_STATS
        DS_STATS = xr.open_dataset(stats_path, decode_times=False)
        
        DS_STATS.time.values = DS_STATS.time.values / 1000
    
        time_reference = datetime.datetime.strptime(DS.reference_date, '%Y-%m-%d %H:%M:%S.%f')
        time_offset = datetime.timedelta(seconds=float(DS_STATS.time_offset.values[0]))
        new_time_reference = time_reference + time_offset
        time_reference_str = datetime.datetime.strftime(new_time_reference, '%Y-%m-%d %H:%M:%S')
        
        DS_STATS.time.attrs['units'] = 'seconds since ' + time_reference_str
        
        DS_STATS = xr.decode_cf(DS_STATS, decode_coords = True, decode_times = True) 
        
        if stats_content=='signal':
            y_range = eval(SETTINGS['RAW_SIGNAL_RANGE'])
    
            ax1.plot('time', 'signal_ch0', data=DS_STATS, label='CH0', color='darkblue')
            ax1.plot('time', 'signal_ch1', data=DS_STATS, label='CH1', color='darkred')
            ax1.plot('time', 'signal_ch2', data=DS_STATS, label='CH2', color='darkgreen')
            ax1.plot('time', 'signal_ch3', data=DS_STATS, label='CH3', color='darkgrey')
            ax1.set_ylabel('Signal')
        if stats_content=='background':
            y_range = eval(SETTINGS['BACKGROUND_RANGE'])
            ax1.plot('time', 'background_ch0', data=DS_STATS, label='CH0_b', color='blue')
            ax1.plot('time', 'background_ch1', data=DS_STATS, label='CH1_b', color='red')
            ax1.plot('time', 'background_ch2', data=DS_STATS, label='CH2_b', color='green')
            ax1.plot('time', 'background_ch3', data=DS_STATS, label='CH3_b', color='grey')
            ax1.set_ylabel('Signal background')
        
        ax1.set_ylim(y_range[0],y_range[1])
        
        # X-Ticks
        h_fmt = mdates.DateFormatter('%H:%M')
        h_interv = mdates.HourLocator(interval = 2)
        ax1.xaxis.set_major_locator(h_interv)
        ax1.xaxis.set_major_formatter(h_fmt)
        if DS.fixed_timeframe.values:
            ax1.set_xlim(DS.date_startp.values,DS.date_endp.values)
            
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, shadow=True) # ncol=2
        
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="10%", pad=0.5)
        cax1.axis('off')
        
        # Labels
        if DS.fixed_timeframe.values:
            date = datetime.datetime.utcfromtimestamp(DS.date_endp.values.astype('O')/1e9)
        else: 
            date = datetime.datetime.utcfromtimestamp(DS.time.values[-1].astype('O')/1e9)
        ax1.set_xlabel('Time ({})'.format(datetime.datetime.strftime(date, '%b %d, %Y')))
        ax1.set_title('Channel statistics')
        
    return fig, ax0, ax1


def format_plot(fig, ax0, ax1, DS, SETTINGS):      
    # Labels
    if DS.fixed_timeframe.values:
        date = datetime.datetime.utcfromtimestamp(DS.date_endp.values.astype('O')/1e9)
    else: 
        date = datetime.datetime.utcfromtimestamp(DS.time.values[-1].astype('O')/1e9)
    # use date of last measurement (date of morning day)
    ax0.set_xlabel('Time ({})'.format(datetime.datetime.strftime(date, '%b %d, %Y')))
    ax0.set_ylabel('Altitude (km)')
    
    # X-Ticks
    h_fmt = mdates.DateFormatter('%H:%M')
    h_interv = mdates.HourLocator(interval = 2)
    ax0.xaxis.set_major_locator(h_interv)
    ax0.xaxis.set_major_formatter(h_fmt)
    if DS.fixed_timeframe.values:
        ax0.set_xlim(DS.date_startp.values,DS.date_endp.values)
    
    y_range = eval(SETTINGS['ALTITUDE_RANGE'])
    
    # Y-Ticks
    ax0.set_ylim(y_range[0],y_range[1])
    
    ax0.grid() # not sure why this is needed again for first plot

    if DS.instrument_name == '':
        fig.suptitle('          German Aerospace Center (DLR)\n \
        ------------------------------\n \
        ------------------------------\n \
        Vertical resolution: {} km\n \
        Temporal resolution: {} min'.format(DS.altitude.resolution / 1000, DS.time.resolution / (1000*60)), fontsize=12)
    else: 
        fig.suptitle('          German Aerospace Center (DLR)\n \
        {}, {}\n \
        ------------------------------\n \
        Vertical resolution: {} km\n \
        Temporal resolution: {} min'.format(DS.instrument_name, DS.station_name, DS.altitude.resolution / 1000, DS.time.resolution / (1000*60)), fontsize=12)
    # Adjust font size to 12 (originally 14)
    fig.tight_layout(rect=[0, 0, 1, 0.88]) # tuple (left, bottom, right, top),
    # fig.subplots_adjust(top=0.87)
    # fig.tight_layout()
    # annotate() for fixed ratios
    
    return fig


def set_latex():
    # plt.style.use('/Users/tennismichel/.matplotlib/stylelib/new_default.mplstyle')
    plt.style.use('latex_default.mplstyle')
    
    # Set tex font
    # plt.rc('font',**{'family':'serif','serif':['CMU Serif']})
    # plt.rc('text', usetex=True) don't use it
    # font.family : serif
    # font.serif : CMU Serif
    # mathtext.fontset : stix

    # Set font size
    # FONT_SIZE = 11
    # SMALL_SIZE = 10.95
    # MEDIUM_SIZE = 11
    # plt.rc('font', size=FONT_SIZE)         # controls default text sizes
    # plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title
    ###
    
    # Axes things
    # plt.rc('axes', )
    # axes.spines.bottom : False
    # axes.spines.top : False
    # axes.spines.left : False
    # axes.spines.right : False
    # axes.axisbelow : True
    # axes.facecolor : dcdcdc
    # xtick.bottom : False
    # ytick.left : False
    
    # # Lines
    # lines.linewidth : 1.2
    
    # # Legend
    # legend.facecolor : white
    
    # # Grid
    # axes.grid : True
    
    # # Figure
    # figure.dpi : 100


    # matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
    # plt.rcParams['font.family'] = "sans-serif"


def plot_nightlyMeans(file_location='../..', file_name="v12means.nc", SETTINGS=None, save_fig=False):
    """
    """
    
    path = os.path.join(file_location, file_name)
    
    DS = xr.open_dataset(path, decode_times=True)
    DS['alt_plot'] = (DS.altitude + DS.altitude_offset + DS.station_height) / 1000 #km
    DS.temperature.values = np.where(DS.temperature == 0, np.nan, DS.temperature) # Change 0 to NaN

    if int(SETTINGS['USE_LATEX_FORMAT']):
        set_latex()
        
    fig, ax0 = plt.subplots(figsize=(10,4))
    im_temp = ax0.pcolormesh(DS.time, DS.alt_plot, np.matrix.transpose(DS.temperature.values),
                             cmap='jet', vmin=140, vmax=280)
    ax0.set_ylim(20,100)
    
    # Labels
    cbar = fig.colorbar(im_temp, ax=ax0)
    cbar.set_label('Temperature (K)')
    ax0.set_xlabel('Time')
    ax0.set_ylabel('Altitude (km)')
    
    ax0.xaxis.tick_bottom()
    plt.grid()
    
    
    # X-Ticks
    # interv = mdates.MonthLocator(interval = 3)
    # ax0.xaxis.set_major_locator(interv)
    fmt = mdates.DateFormatter('%b-%y')
    # fmt = mdates.ConciseDateFormatter(mdates.AutoDateLocator())
    ax0.xaxis.set_major_formatter(fmt)
    # ax0.xaxis.set_tick_params(rotation=30)
    
    # X-limits
    if (SETTINGS['NIGHTLY_MEAN_FIXED'] == '1'):
        timeframe = eval(SETTINGS['FIXED_TIMEFRAME_NM'])
        timeframe[0] = datetime.datetime.strptime(timeframe[0], '%Y-%m')
        timeframe[1] = datetime.datetime.strptime(timeframe[1], '%Y-%m')
        
        ax0.set_xlim(timeframe[0],timeframe[1])
        
    # ax.set_title()
    fig.suptitle('German Aerospace Center (DLR) - Nightly mean temperature profiles           \n\
                 {}, {}           '.format(DS.instrument_name, DS.station_name), fontsize=12)
    fig.subplots_adjust(top=0.85)
    # fig.tight_layout(rect=[1, 1, 1, 0.6])
    # fig.tight_layout()
    
    if save_fig:
        fig_name = 'nightly_means.png'
        fig.savefig(SETTINGS['PLOT_FOLDER_NM'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png') # orientation='portrait'
    
    return fig


if __name__ == '__main__':
    "generation of SETTINGS dict necessary!!!"
    
    # plot_LidarT()
    plot_nightlyMeans(save_fig=False)