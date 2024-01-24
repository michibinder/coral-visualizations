################################################################################
# Copyright 2023 German Aerospace Center                                       #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

from matplotlib.colors import LinearSegmentedColormap

def get_wave_cmap():
    """Create a custom colormap with smooth transitions between the given colors."""
    c0 = 'darkslateblue'
    c1 = 'royalblue'
    c2 = 'cornflowerblue'
    #c2 = 'lightblue'
    c3 = 'lavender'
    c4 = 'white' # 'whitesmoke'
    c5 = 'palegoldenrod'
    c55 = '#EEE600'
    c6 = 'goldenrod'
    c7 = 'indianred' # firebrick
    c8 = 'darkred'

    # colors = [(0.0, c0), (0.1, c1), (0.2, c2), (0.4, c3), (0.48, c4), (0.52, c4), (0.55, c5), (0.65, c55), (0.8, c6), (0.9, c7), (1.0, c8)]
    colors = [(0.0, c0), (0.1, c1), (0.2, c2), (0.4, c3), (0.48, c4), (0.52, c4), (0.6, c5), (0.75, c6), (0.85, c7), (1.0, c8)]
    # colors = [(0.0, c0), (0.32, c0), (0.4, c1), (0.6, c1), (0.68, c2), (0.73, c2), (0.78, c3), (1.0, c3)]
    
    cmap = LinearSegmentedColormap.from_list('wave', colors=colors, N=256)
    return cmap