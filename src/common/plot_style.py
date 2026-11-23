"""Figure style shared by the figure scripts of both tasks (src/direction/plots.py and the scripts that reuse its header,
src/volatility/plot_vol.py): palette, surfaces and matplotlib settings for static slide PNGs, and the figures folder.
Palette validated; two slots are below 3:1 contrast, so every series is direct-labelled.
Taken out of src/direction/plots.py without changes.
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as C

FIG = os.path.join(C.ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
