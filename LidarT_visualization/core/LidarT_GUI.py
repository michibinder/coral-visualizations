#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:41:20 2020

@author: tennismichel
"""

import sys
import os
import glob
import tkinter as tk
import datetime

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    
    # Request from terminal to import the following...
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # , NavigationToolbar2TkAgg
from LidarT_visualization import plot_LidarT

SETTINGS_FILE = 'settings.txt'
# SETTINGS_FILE_2 = 'settings_default.txt'

class LidarT_GUI(tk.Canvas):
    """inherits from tk canvas"""
    # HEIGHT = 800 set via settings.txt
    # WIDTH  = 700
    TRES = ['T10', 'T15', 'T20', 'T30', 'T60', 'T120', 'T1140']
    ZRES = {'Z900', 'Z1900'}
    
    
    def __init__(self, master):
        self.master = master
        self.file_location = SETTINGS['FILELOCATION']
        self.stats_location = SETTINGS['STATSLOCATION']
        self.current_file = SETTINGS['STARTFILE']
        
        sep_file_name = self.current_file.rpartition('_')
        self.current_date = sep_file_name[0]
        T_res = sep_file_name[2].rpartition('Z')[0]
        Z_res = 'Z' + sep_file_name[2].rpartition('Z')[2][0:-3]
        
        self.date_index = None
        self.file_list = []
        self.date_list = []
        self.date_type_list =[]
        self.setup_file_list()
        self.stats_path = ''
        
        # Variables to trace
        self.var_radioB = tk.StringVar()
        self.var_radioB.set('tmp') # ['tmp', 'runningM', 'butterworthF', 'tmp_err']
        self.var_radioB.trace("w", self.update)
        
        self.var_zres = tk.StringVar()
        self.var_zres.set(Z_res) # {'Z900', 'Z1900'}
        self.var_zres.trace("w", self.update)
        
        self.var_tres = tk.StringVar()
        self.var_tres.set(T_res) # {'T15', 'T20', 'T30', 'T60', ...}
        self.var_tres.trace("w", self.update)
        
        self.var_fixedtime = tk.BooleanVar()
        self.var_fixedtime.set(False)
        self.var_fixedtime.trace("w", self.update)
        
        self.var_stats = tk.StringVar()
        self.var_stats.set('nightly_mean') # ['nightly_mean', 'signal', 'background']
        self.var_stats.trace("w", self.update)
        
        #self.var_stats = tk.BooleanVar()
        #self.var_stats.set(False)
        #self.var_stats.trace("w", self.update)
        
        # Constructor of tk.canvas
        super(LidarT_GUI, self).__init__(master, height=int(SETTINGS['WINDOW_HEIGHT']), width=int(SETTINGS['WINDOW_WIDTH']))
        super(LidarT_GUI, self).pack()
        
        master.title("LIDAR temperature profiles")
        
        # Close button (for windows?)
        self.close_button = tk.Button(self, text="CLOSE", command=self.close_window, font=FONT, bg='#F3F3F3')
        self.close_button.place(relx=0.03, rely=0.01, relwidth=0.2, relheight=0.03)
        
        
        # Forward/Backward button frame
        self.button_frame = tk.Frame(self,  bg='#BCBBBB')
        self.button_frame.place(relx=0.25, rely=0.01, relwidth=0.7, relheight=0.03)

        self.backward_button = tk.Button(self.button_frame, text='<<', command=self.b_button_callback, bg='#F3F3F3', font=FONT2)
        self.backward_button.place(relx=0, rely=0, relwidth=0.2, relheight=1)
        self.forward_button = tk.Button(self.button_frame, text='>>', command=self.f_button_callback, bg='#F3F3F3', font=FONT2)
        self.forward_button.place(relx=0.8, rely=0, relwidth=0.2, relheight=1)
        
        self.file_label = tk.Label(self.button_frame, text=self.current_file, font=FONT, bg='#F3F3F3')
        self.file_label.place(relx=0.2, rely=0, relwidth=0.6, relheight=1)
  
       
        # Side button frame (radio_button_frame)
        self.radio_button_frame = tk.Frame(self, bg='#BCBBBB',bd=2) 
        self.radio_button_frame.place(relx=0.03, rely=0.1, relwidth=0.2, relheight=0.85)
        
        self.tmp_button = tk.Radiobutton(self.radio_button_frame, text='TEMPERATURE', variable=self.var_radioB, value='tmp', bg='#F3F3F3', font=FONT)
        self.tmp_button.place(relx=0, rely=0, relwidth=1, relheight=0.05)
        
        self.runningM_button = tk.Radiobutton(self.radio_button_frame, text='RUNNING MEAN', variable=self.var_radioB, value='runningM', bg='#F3F3F3', font=FONT)
        self.runningM_button.place(relx=0, rely=0.06, relwidth=1, relheight=0.05)
        
        self.topm_button = tk.Radiobutton(self.radio_button_frame, text='BUTTERWORTH F', variable=self.var_radioB, value='butterworthF', bg='#F3F3F3', font=FONT)
        self.topm_button.place(relx=0, rely=0.12, relwidth=1, relheight=0.05)
        
        self.error_button = tk.Radiobutton(self.radio_button_frame, text='ERROR', variable=self.var_radioB, value='tmp_err', bg='#F3F3F3', font=FONT)
        self.error_button.place(relx=0, rely=0.18, relwidth=1, relheight=0.05)
        
        
        self.range_label = tk.Label(self.radio_button_frame, text='RANGE_T:', font=FONT, bg='#F3F3F3')
        self.range_label.place(relx=0, rely=0.25, relwidth=0.45, relheight=0.05)
        
        self.range_entry = tk.Entry(self.radio_button_frame, font=FONT) #  self.current_timeframe)
        self.range_entry.place(relx=0.45, rely=0.25, relwidth=0.55, relheight=0.05)
        self.range_entry.insert(0, SETTINGS['TEMP_RANGE'])
        
        self.range_button = tk.Button(self.radio_button_frame, text='SET RANGE', command=self.range_button_callback, bg='#F3F3F3', font=FONT)
        self.range_button.place(relx=0.2, rely=0.305, relwidth=0.6, relheight=0.05)
        
        
        ####
        self.t_menu = tk.OptionMenu(self.radio_button_frame, self.var_tres, *LidarT_GUI.TRES)
        self.t_menu.config(font=FONT)
        t_menu_list = self.nametowidget(self.t_menu.menuname)
        t_menu_list.config(font=FONT) # set the drop down menu font
        self.t_menu.place(relx=0, rely=0.37, relwidth=1, relheight=0.05)
        
        self.z_menu = tk.OptionMenu(self.radio_button_frame, self.var_zres, *LidarT_GUI.ZRES)
        self.z_menu.config(font=FONT)
        z_menu_list = self.nametowidget(self.z_menu.menuname)
        z_menu_list.config(font=FONT)
        self.z_menu.place(relx=0, rely=0.43, relwidth=1, relheight=0.05)
        
        
        ####        
        self.nightly_mean_button = tk.Radiobutton(self.radio_button_frame, text='NIGHTLY MEAN', variable=self.var_stats, value='nightly_mean', bg='#F3F3F3', font=FONT)
        self.nightly_mean_button.place(relx=0, rely=0.495, relwidth=1, relheight=0.05)
        
        self.signal_button = tk.Radiobutton(self.radio_button_frame, text='RAW SIGNAL', variable=self.var_stats, value='signal', bg='#F3F3F3', font=FONT)
        self.signal_button.place(relx=0, rely=0.545, relwidth=1, relheight=0.05)
        
        self.background_button = tk.Radiobutton(self.radio_button_frame, text='BACKGROUND', variable=self.var_stats, value='background', bg='#F3F3F3', font=FONT)
        self.background_button.place(relx=0, rely=0.595, relwidth=1, relheight=0.05)
        
        
        ####
        self.timeframe_button = tk.Checkbutton(self.radio_button_frame, text='FIXED TIME FRAME', variable=self.var_fixedtime, bg='#F3F3F3', font=FONT)
        self.timeframe_button.place(relx=0, rely=0.665, relwidth=1, relheight=0.05)
        
        self.timeframe_entry = tk.Entry(self.radio_button_frame, justify='center', font=FONT)
        self.timeframe_entry.place(relx=0.25, rely=0.72, relwidth=0.5, relheight=0.05)
        self.timeframe_entry.insert(0, SETTINGS['FIXED_TIMEFRAME'])
        
        
        ####
        self.date_entry = tk.Entry(self.radio_button_frame, text=self.current_date, justify='center', font=FONT)
        self.date_entry.place(relx=0, rely=0.78, relwidth=1, relheight=0.05)
        self.date_entry.insert(0, self.current_date)
        
        self.entry_button = tk.Button(self.radio_button_frame, text='SET DATE', command=self.entry_button_callback, bg='#F3F3F3', font=FONT)
        self.entry_button.place(relx=0.25, rely=0.835, relwidth=0.5, relheight=0.05)
        
        
        ####
        self.save_settings_button = tk.Button(self.radio_button_frame, text='SAVE SETTINGS', command=self.save_settings_button_callback, bg='#F3F3F3', font=FONT)
        self.save_settings_button.place(relx=0, rely=0.895, relwidth=1, relheight=0.05)
        
        self.save_button = tk.Button(self.radio_button_frame, text='SAVE PLOT', command=self.save_button_callback, bg='#F3F3F3', font=FONT)
        self.save_button.place(relx=0, rely=0.95, relwidth=1, relheight=0.05)
        
        self.update()
        
        
    def b_button_callback(self):
        if self.date_index == 0:
            self.date_index = len(self.date_list)-1
        else:
            self.date_index -= 1
            
        self.current_date = self.date_list[self.date_index]
        self.update()
        
        
    def f_button_callback(self):
        if self.date_index == len(self.date_list)-1:
            self.date_index = 0
        else:
            self.date_index += 1
            
        self.current_date = self.date_list[self.date_index]
        self.update()

    
    def entry_button_callback(self):    
        date_requested = self.date_entry.get()
        date_requested = datetime.datetime.strptime(date_requested, '%Y%m%d-%H%M')
        self.date_index = nearest_ind(self.date_type_list, date_requested)
        self.current_date = self.date_list[self.date_index]
        self.update()
    
    
    def range_button_callback(self):    
        range_str = self.range_entry.get()
        plot_content = self.var_radioB.get()
        if plot_content == 'runningM':
            SETTINGS['TEMP_RANGE_RUNNINGM'] = range_str        
        elif plot_content == 'butterworthF':
            SETTINGS['TEMP_RANGE_BUTTERWORTHF'] = range_str            
        elif plot_content == 'tmp_err':
            SETTINGS['TEMP_RANGE_ERROR'] = range_str
        else:  # fig_content == 'tmp'
            SETTINGS['TEMP_RANGE'] = range_str
            
        self.update()
        
    
    def save_button_callback(self):
        plot_LidarT(file_location = self.file_location, file_name=self.current_file, SETTINGS=SETTINGS, plot_content=self.var_radioB.get(), fixed_timeframe=self.var_fixedtime.get(), stats_content=self.var_stats.get(), stats_path=self.stats_path, save_fig=True)
        
        
    def save_settings_button_callback(self):
        # update file location and file name if required
        
        # save dict to txt file
        with open(SETTINGS_FILE, 'w') as file:
            for key in SETTINGS:
                file.write(key + ': ' + SETTINGS[key] + '\n')
        
    def update(self, *args):
        SETTINGS['FIXED_TIMEFRAME'] = self.timeframe_entry.get()
        
        plot_content = self.var_radioB.get()
        if plot_content == 'runningM':
            self.range_label.config(text='RANGE_RM:')
            self.range_entry.delete(0, 'end')
            self.range_entry.insert(0, SETTINGS['TEMP_RANGE_RUNNINGM'])
        elif plot_content == 'butterworthF':
            self.range_label.config(text='RANGE_BF:')  
            self.range_entry.delete(0, 'end')
            self.range_entry.insert(0, SETTINGS['TEMP_RANGE_BUTTERWORTHF'])
        elif plot_content == 'tmp_err':
            self.range_label.config(text='RANGE_E:')
            self.range_entry.delete(0, 'end')
            self.range_entry.insert(0, SETTINGS['TEMP_RANGE_ERROR'])
        else:  # fig_content == 'tmp'
            self.range_label.config(text='RANGE_T:')
            self.range_entry.delete(0, 'end')
            self.range_entry.insert(0, SETTINGS['TEMP_RANGE'])
        
        stats_file_path = self.stats_location + '/' + self.current_date + '_' + self.var_tres.get() + self.var_zres.get() + '_stats.nc'
        if os.path.isfile(stats_file_path):
            self.signal_button.config(state=tk.NORMAL)
            self.background_button.config(state=tk.NORMAL)
            if (self.var_stats.get() == 'signal') or (self.var_stats.get() == 'background'):
                self.stats_path = stats_file_path
            else:
                self.stats_path = ''
        else:
            self.signal_button.config(state=tk.DISABLED)
            self.background_button.config(state=tk.DISABLED)
            self.stats_path = ''
                
            
        self.current_file = self.current_date + '_' + self.var_tres.get() + self.var_zres.get() + '.nc'
        if self.current_file in self.file_list:    
            self.file_label.config(text=self.current_file)
            self.date_entry.delete(0, 'end')
            self.date_entry.insert(0, self.current_date)
            self.draw_figure()
        else:
            self.file_label.config(text='This file is not available')
        
        
            
        
    def draw_figure(self):
        fig = plot_LidarT(file_location = self.file_location, file_name=self.current_file, SETTINGS=SETTINGS, plot_content=self.var_radioB.get(), fixed_timeframe=self.var_fixedtime.get(), stats_content=self.var_stats.get(), stats_path=self.stats_path)
        self.fig_canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
        self.fig_canvas.get_tk_widget().place(relx=0.25, rely=0.05, relwidth=0.7, relheight=0.93)


    def close_window(self):
        """for OSX with Spyder closing the window with the button provided by the OS is best"""
        self.master.quit() # works well for command line
        # self.master.destroy()
        # sys.exit()
        
    
    def setup_file_list(self):
        """
        generates three lists:
            - list of available files
            - list of dates of type string
            - list of dates of type datetime
        """
        file_paths = sorted(glob.glob(os.path.join(self.file_location, '*.nc')))
        for file in file_paths:
            file = file.rpartition('/')[-1]
            self.file_list.append(file)
            
            # generate list of available dates (plus time)
            # file_date = file.rpartition('_')[0]
            file_date = file.partition('_')[0]
            if file_date in self.date_list:
                continue
            else:
                self.date_list.append(file_date)
        
        for i, date in enumerate(self.date_list):
            if date == self.current_date:
                self.date_index = i
                break
            
        self.date_type_list = [datetime.datetime.strptime(i, '%Y%m%d-%H%M') for i in self.date_list]
        

def nearest(items, pivot):
    """returns nearest item in items, if items supports comparison operands"""
    # int_array = [int(i) for i in array]
    # array = np.asarray(array)
    # idx = (np.abs(array - value)).argmin()
    return min(items, key=lambda x: abs(x - pivot))


def nearest_ind(items, pivot):
    """returns index of nearest item in items"""
    time_diff = np.abs([date - pivot for date in items])
    return time_diff.argmin(0)


def load_settings():
    global SETTINGS
    global FONT
    global FONT2
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
    
    
    FONT  = ('Courier', int(SETTINGS['FONTSIZE'])) # ('Arial', 12), ('TkDefaultFont', 12)
    FONT2 = ('Courier', int(SETTINGS['FONTSIZE'])+2)

    
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
            if os.path.exists(args[1]):
                SETTINGS['STARTFILE'] = args[1]
            else:
                print('Startfile NOT found. First one selected')
                SETTINGS['STARTFILE'] = sorted(glob.glob(os.path.join(SETTINGS['FILELOCATION'], '*.nc')))[0].rpartition('/')[-1]
        
    root = tk.Tk()
    my_gui = LidarT_GUI(root)
    root.mainloop()