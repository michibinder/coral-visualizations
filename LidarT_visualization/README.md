# LIDAR temperature measurements visualization

This tool visualizes LIDAR temperature measurements (of the stratosphere)
and provides filters for the observation of gravity waves. 


## Start-up:

Run the LidarT_GUI.py script, which opens a GUI to analyse and compare 
different measurements. The user can provide a first argument with a
relative path to the campaign folder and a second argument with
the respective file name to start with. Without specifing a campaign folder or file, the folder and file within the 'settings.txt' file is loaded.

Therefore, folder and start file can be set through the 'settings.txt' as 
well as other parameters. Some of these parameters can be changed while using
the GUI and can then be overwritten/saved to the 'settings.txt' file with the 'SAVE SETTINGS'-button. 


## Usage

- Files can be changed through the entry field or the forward and backward 
button at the top of the GUI
- The format of the entry field must not be changed
- The visualized time frame can be set below its respective check box. It is updated 
automatically (when something else changes), so either uncheck and check the fixed time frame box afterwards or change the plotted file.
- The temperature range for all four plotting modes (running mean, butterworts..) 
can be set through the range entry field, which adapts to the current plot.
- The second plot in the figure can either be the nightly mean temperature or
(if a stats.nc file is available in the respective raw campaign folder) the different channel signals and channel backgrounds. If no stats.nc file is available for the current settings, the option for channel signals and backgrounds is greyed out.
- Altitude and mean temperature range can only be set before opening the GUI 
in the 'settings.txt'-file. 
- Channel signal and background range can only be set before opening the GUI 
in the 'settings.txt'-file, too.
- The 'settings_default.txt'-file is a backup file, in case the main one is 
corrupted. The name of the settings file can be changed at the beginning of 
'LidarT_GUI.py', if different settings are required for different use cases 
or campaigns.

## Dependencies

Make sure you have all dependencies installed. These are:

- Tkinter

- numpy

- xarray

- scipy

- matplotlib

- mpl_toolkits