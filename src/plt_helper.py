################################################################################
# Copyright 2023 German Aerospace Center                                       #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

import matplotlib.dates as mdates
import time
from datetime import datetime
import numpy as np

"""Config"""
pbar_interval = 5 # %

def timelab_format_func(value, tick_number):
    dt = mdates.num2date(value)
    if dt.hour == 0:
        return "{}\n{}".format(dt.strftime("%Y-%b-%d"), dt.strftime("%H"))
    else:
        return dt.strftime("%H")


def major_formatter_lon(x, pos):
    """Using western coordinates"""
    return "%.f°W" % abs(x)
    ##return "%.f°E" % abs(x)


def major_formatter_lat(x, pos):
    return "%.f°S" % abs(x)


def show_progress(progress_counter, lock, stime, total_tasks):
    with lock:
        progress_counter.value += 1
        if total_tasks <= 100/pbar_interval:
            print(f"[p]  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Number of tasks below progress bar limit.")
        else:
            if (progress_counter.value % (total_tasks // (100/pbar_interval))) == 0 or progress_counter.value == total_tasks or progress_counter.value == 1:
                progress = progress_counter.value / total_tasks
                elapsed = time.time() - stime
                eta = (elapsed / progress) * (1 - progress)

                # Convert elapsed and ETA to hours, minutes, and seconds
                elapsed_hrs, elapsed_rem = divmod(elapsed, 3600)
                elapsed_min, elapsed_sec = divmod(elapsed_rem, 60)
                eta_hrs, eta_rem = divmod(eta, 3600)
                eta_min, eta_sec = divmod(eta_rem, 60)

                # Progress bar
                total_hashtags = int(100/pbar_interval)
                hashtag_str = "#" * int(np.ceil(progress * total_hashtags))
                minus_str = "-" * int((1 - progress) * total_hashtags)

                print(f"[p]  |{hashtag_str}{minus_str}| Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Progress: {progress*100:05.2f}% - Elapsed: {int(elapsed_hrs):02d}:{int(elapsed_min):02d}:{int(elapsed_sec):02d} - ETA: {int(eta_hrs):02d}:{int(eta_min):02d}:{int(eta_sec):02d} (hh:mm:ss)", flush=True)

def add_watermark(fig):
    fig.text(0.25, 0.75, 'German Aerospace Center', style = 'italic', fontsize = 18, color = "grey", alpha=0.15, ha='center', va='center', rotation=30) 
    fig.text(0.75, 0.25, 'German Aerospace Center', style = 'italic', fontsize = 18, color = "grey", alpha=0.15, ha='center', va='center', rotation=30) 
    return fig