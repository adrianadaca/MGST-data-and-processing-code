from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import csv
import glob
import re
import datetime as dt
from scipy.signal import savgol_filter
from bokeh.plotting import figure, output_file, show
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.models import LinearAxis, Range1d, HoverTool, Span, Label
import matplotlib.pyplot as plt
from bokeh.io import export_png
import math
from matplotlib.lines import Line2D
import functools
from sklearn.metrics import mean_squared_error

# import data
files = glob.glob("scaled data/*.csv")

# set up some lists
slip20 = []
slip20_indexlist = []
slip70 = []
slip70_indexlist = []
GSL_20 = []
lunar_g_20 = []
GSL_70 = []
lunar_g_70 = []

for i, filepath in enumerate(files):
    df = pd.read_csv(filepath)  # import file as pandas dataframe
    df.set_index(df['Timestamp'], inplace=True)  # set up dataframe index

    slip = re.findall(".*_(\d{2}).*", filepath)  # read slip from filename
    expt_type = re.findall(".*/(.*)_.*", filepath)  # read experiment type from filename (GSL or lunar-g)

    # put file in the appropriate list
    if int(slip[0]) == 20:
        slip20.append(df)
        slip20_indexlist.append(df.index)
        if expt_type[0] == 'lunar-g':
            lunar_g_20.append(df)
        if expt_type[0] == 'gsl':
            length = np.where(df.index >= 20)[0][0]
            GSL_20.append(df[0:length])

    if int(slip[0]) == 70:
        slip70.append(df)
        slip70_indexlist.append(df.index)
        if expt_type[0] == 'lunar-g':
            lunar_g_70.append(df)
        if expt_type[0] == 'gsl':
            length = np.where(df.index >= 20)[0][0]
            GSL_70.append(df[0:length])

# combine all indices
slip20_index = functools.reduce((lambda a, b: a.union(b)), slip20_indexlist)
slip70_index = functools.reduce((lambda a, b: a.union(b)), slip70_indexlist)

# set up more lists
new_indices = [slip20_index, slip70_index]
slip20_dfs = [GSL_20, lunar_g_20]
slip70_dfs = [GSL_70, lunar_g_70]
list_of_df_lists = [slip20_dfs, slip70_dfs]
avg_dfs = []  # order: GSL_20, lunar_g_20, GSL_70, lunar_g_70
stds = []

for j, slip in enumerate(list_of_df_lists):
    for k, expt_type in enumerate(slip):
        for l, df in enumerate(expt_type):
            # reindex with the union
            new_df = df.reindex(new_indices[j])
            # interpolation
            list_of_df_lists[j][k][l] = new_df.interpolate(method='linear', limit_direction='forward', axis=0)
        # concatenate
        concatenation = functools.reduce((lambda a, b: pd.concat([a, b])), expt_type)
        # average
        avg = concatenation.groupby(level=0).mean()
        avg.to_csv('avg-%s_%s.csv' % (j, k), index=True, index_label='Timestamp')
        avg_dfs.append(avg)
        # standard deviation
        std = concatenation.groupby(level=0).std()
        stds.append(std)

# compute MSPE (mean squared percent error)

# sort by slip [slip 20, slip 70]
gsl_avg_data = [avg_dfs[0], avg_dfs[2]]
lunar_g_avg_data = [avg_dfs[1], avg_dfs[3]]
gsl_std = [stds[0], stds[2]]
lunar_g_std = [stds[1], stds[3]]

# compute mspe.
list_20 = [lunar_g_avg_data[0], gsl_avg_data[0]]
list_70 = [lunar_g_avg_data[1], gsl_avg_data[1]]
index_list_20 = [lunar_g_avg_data[0].index, gsl_avg_data[0].index]
index_list_70 = [lunar_g_avg_data[1].index, gsl_avg_data[1].index]
std_list_20 = [lunar_g_std[0], gsl_std[0]]
std_list_70 = [lunar_g_std[1], gsl_std[1]]

# 20% slip:
length_20 = np.where(list_20[0].index >= 20)[0][0]
# 70% slip
length_70 = np.where(list_70[0].index >= 20)[0][0]

# MSPE - mean squared percent error
lunar_dp_20 = list_20[0].dp.iloc[10369:length_20]
lunar_p_20 = list_20[0].p.iloc[10369:length_20]
lunar_z_20 = list_20[0].z.iloc[10369:length_20]
lunar_dp_70 = list_70[0].dp.iloc[9132:length_70]
lunar_p_70 = list_70[0].p.iloc[9132:length_70]
lunar_z_70 = list_70[0].z.iloc[9132:length_70]

# MSPE = np.mean(np.square(((y_true - y_pred) / y_true)), axis=0)
GSL_MSPE_dp_20 = np.mean(np.square(((lunar_dp_20 - list_20[1].dp.iloc[10369:length_20]) / lunar_dp_20)), axis=0) * 100
GSL_MSPE_p_20 = np.mean(np.square(((lunar_p_20 - list_20[1].p.iloc[10369:length_20]) / lunar_p_20)), axis=0) * 100
GSL_MSPE_z_20 = np.mean(np.square(((lunar_z_20 - list_20[1].z.iloc[10369:length_20]) / lunar_z_20)), axis=0) * 100

GSL_MSPE_dp_70 = np.mean(np.square(((lunar_dp_70 - list_70[1].dp.iloc[9132:length_70]) / lunar_dp_70)), axis=0) * 100
GSL_MSPE_p_70 = np.mean(np.square(((lunar_p_70 - list_70[1].p.iloc[9132:length_70]) / lunar_p_70)), axis=0) * 100
GSL_MSPE_z_70 = np.mean(np.square(((lunar_z_70 - list_70[1].z.iloc[9132:length_70]) / lunar_z_70)), axis=0) * 100

print('GSL_MSPE_dp_20', GSL_MSPE_dp_20)
print('GSL_MSPE_z_20', GSL_MSPE_z_20)
print('GSL_MSPE_p_20', GSL_MSPE_p_20)
print('GSL_MSPE_dp_70', GSL_MSPE_dp_70)
print('GSL_MSPE_z_70', GSL_MSPE_z_70)
print('GSL_MSPE_p_70', GSL_MSPE_p_70)

# plot
dp_plot_20, axdp20 = plt.subplots()
dp_plot_70, axdp70 = plt.subplots()
sinkage_plot_20, axz20 = plt.subplots()
sinkage_plot_70, axz70 = plt.subplots()
power_plot_20, axp20 = plt.subplots()
power_plot_70, axp70 = plt.subplots()

axz20.fill_between(gsl_avg_data[0].index, gsl_avg_data[0].z - 1.96 * gsl_std[0].z/np.sqrt(3),
                   gsl_avg_data[0].z + 1.96 * gsl_std[0].z/np.sqrt(3), color='g', alpha=.1)
g20_1 = axz20.plot(gsl_avg_data[0].index, gsl_avg_data[0].z, "g--", linewidth=0.8)
g20_2 = axz20.fill(np.NaN, np.NaN, 'g', alpha=.1)
axz20.fill_between(lunar_g_avg_data[0].index, lunar_g_avg_data[0].z - 1.96 * lunar_g_std[0].z/np.sqrt(4),
                   lunar_g_avg_data[0].z + 1.96 * lunar_g_std[0].z/np.sqrt(4), color='#1F77B4', alpha=.1)
l20_1 = axz20.plot(lunar_g_avg_data[0].index, lunar_g_avg_data[0].z, "#1F77B4", linewidth=0.8)
l20_2 = axz20.fill(np.NaN, np.NaN, '#1F77B4', alpha=.1)
axz20.legend([(l20_1[0], l20_2[0]), (g20_2[0], g20_1[0]), ],
             ['Lunar-g results', 'GSL prediction'])

axdp20.fill_between(gsl_avg_data[0].index, gsl_avg_data[0].dp - 1.96 * gsl_std[0].dp/np.sqrt(3),
                   gsl_avg_data[0].dp + 1.96 * gsl_std[0].dp/np.sqrt(3), color='g', alpha=.1)
g20_1 = axdp20.plot(gsl_avg_data[0].index, gsl_avg_data[0].dp, "g--", linewidth=0.8)
g20_2 = axdp20.fill(np.NaN, np.NaN, 'g', alpha=.1)
axdp20.fill_between(lunar_g_avg_data[0].index, lunar_g_avg_data[0].dp - 1.96 * lunar_g_std[0].dp/np.sqrt(4),
                   lunar_g_avg_data[0].dp + 1.96 * lunar_g_std[0].dp/np.sqrt(4), color='#1F77B4', alpha=.1)
l20_1 = axdp20.plot(lunar_g_avg_data[0].index, lunar_g_avg_data[0].dp, "#1F77B4", linewidth=0.8)
l20_2 = axdp20.fill(np.NaN, np.NaN, '#1F77B4', alpha=.1)
axdp20.legend([(l20_1[0], l20_2[0]), (g20_2[0], g20_1[0]), ],
             ['Lunar-g results', 'GSL prediction'])

axp20.fill_between(gsl_avg_data[0].index, gsl_avg_data[0].p - 1.96 * gsl_std[0].p/np.sqrt(3),
                   gsl_avg_data[0].p + 1.96 * gsl_std[0].p/np.sqrt(3), color='g', alpha=.1)
g20_1 = axp20.plot(gsl_avg_data[0].index, gsl_avg_data[0].p, "g--", linewidth=0.8)
axp20.fill_between(lunar_g_avg_data[0].index, lunar_g_avg_data[0].p - 1.96 * lunar_g_std[0].p/np.sqrt(4),
                   lunar_g_avg_data[0].p + 1.96 * lunar_g_std[0].p/np.sqrt(4), color='#1F77B4', alpha=.1)
l20_1 = axp20.plot(lunar_g_avg_data[0].index, lunar_g_avg_data[0].p, "#1F77B4", linewidth=0.8)
g20_2 = axp20.fill(np.NaN, np.NaN, 'g', alpha=.1)
l20_2 = axp20.fill(np.NaN, np.NaN, '#1F77B4', alpha=.1)
axp20.legend([(l20_1[0], l20_2[0]), (g20_2[0], g20_1[0]), ],
             ['Lunar-g results', 'GSL prediction'])

axz70.fill_between(gsl_avg_data[1].index, gsl_avg_data[1].z - 1.96 * gsl_std[1].z/np.sqrt(3),
                   gsl_avg_data[1].z + 1.96 * gsl_std[1].z/np.sqrt(3), color='g', alpha=.1)
g20_1 = axz70.plot(gsl_avg_data[1].index, gsl_avg_data[1].z, "g--", linewidth=0.8)
axz70.fill_between(lunar_g_avg_data[1].index, lunar_g_avg_data[1].z - 1.96 * lunar_g_std[1].z/np.sqrt(4),
                   lunar_g_avg_data[1].z + 1.96 * lunar_g_std[1].z/np.sqrt(4), color='#1F77B4', alpha=.1)
l20_1 = axz70.plot(lunar_g_avg_data[1].index, lunar_g_avg_data[1].z, "#1F77B4", linewidth=0.8)
g20_2 = axz70.fill(np.NaN, np.NaN, 'g', alpha=.1)
l20_2 = axz70.fill(np.NaN, np.NaN, '#1F77B4', alpha=.1)
axz70.legend([(l20_1[0], l20_2[0]), (g20_2[0], g20_1[0]), ],
             ['Lunar-g results', 'GSL prediction'])

axdp70.fill_between(gsl_avg_data[1].index, gsl_avg_data[1].dp - 1.96 * gsl_std[1].dp/np.sqrt(3),
                   gsl_avg_data[1].dp + 1.96 * gsl_std[1].dp/np.sqrt(3), color='g', alpha=.1)
g20_1 = axdp70.plot(gsl_avg_data[1].index, gsl_avg_data[1].dp, "g--", linewidth=0.8)
axdp70.fill_between(lunar_g_avg_data[1].index, lunar_g_avg_data[1].dp - 1.96 * lunar_g_std[1].dp/np.sqrt(4),
                   lunar_g_avg_data[1].dp + 1.96 * lunar_g_std[1].dp/np.sqrt(4), color='#1F77B4', alpha=.1)
l20_1 = axdp70.plot(lunar_g_avg_data[1].index, lunar_g_avg_data[1].dp, "#1F77B4", linewidth=0.8)
g20_2 = axdp70.fill(np.NaN, np.NaN, 'g', alpha=.1)
l20_2 = axdp70.fill(np.NaN, np.NaN, '#1F77B4', alpha=.1)
axdp70.legend([(l20_1[0], l20_2[0]), (g20_2[0], g20_1[0]), ],
             ['Lunar-g results', 'GSL prediction'])

axp70.fill_between(gsl_avg_data[1].index, gsl_avg_data[1].p - 1.96 * gsl_std[1].p/np.sqrt(3),
                   gsl_avg_data[1].p + 1.96 * gsl_std[1].p/np.sqrt(3), color='g', alpha=.1)
g20_1 = axp70.plot(gsl_avg_data[1].index, gsl_avg_data[1].p, "g--", linewidth=0.8)
axp70.fill_between(lunar_g_avg_data[1].index, lunar_g_avg_data[1].p - 1.96 * lunar_g_std[1].p/np.sqrt(4),
                   lunar_g_avg_data[1].p + 1.96 * lunar_g_std[1].p/np.sqrt(4), color='#1F77B4', alpha=.1)
l20_1 = axp70.plot(lunar_g_avg_data[1].index, lunar_g_avg_data[1].p, "#1F77B4", linewidth=0.8)
g20_2 = axp70.fill(np.NaN, np.NaN, 'g', alpha=.1)
l20_2 = axp70.fill(np.NaN, np.NaN, '#1F77B4', alpha=.1)
axp70.legend([(l20_1[0], l20_2[0]), (g20_2[0], g20_1[0]), ],
             ['Lunar-g results', 'GSL prediction'])

axz20.set_ylim(0, 25)
axz70.set_ylim(0, 25)
axdp20.set_ylim(0, 0.8)
axdp70.set_ylim(0, 0.8)
axp20.set_ylim(0, 4)
axp70.set_ylim(0, 4)

axz20.set_xlim(0, 20)
axz70.set_xlim(0, 20)
axdp20.set_xlim(0, 20)
axdp70.set_xlim(0, 20)
axp20.set_xlim(0, 20)
axp70.set_xlim(0, 20)

axz20.set_xlabel("Elapsed Time (s)", fontsize=14)
axdp20.set_xlabel("Elapsed Time (s)", fontsize=14)
axp20.set_xlabel("Elapsed Time (s)", fontsize=14)
axz70.set_xlabel("Elapsed Time (s)", fontsize=14)
axdp70.set_xlabel("Elapsed Time (s)", fontsize=14)
axp70.set_xlabel("Elapsed Time (s)", fontsize=14)

axz20.set_ylabel("Sinkage (mm)", fontsize=14)
axdp20.set_ylabel("DP/W", fontsize=14)
axp20.set_ylabel("Power (W)", fontsize=14)
axz70.set_ylabel("Sinkage (mm)", fontsize=14)
axdp70.set_ylabel("DP/W", fontsize=14)
axp70.set_ylabel("Power (W)", fontsize=14)

for item in (axdp20.get_xticklabels() + axdp20.get_yticklabels() + axdp70.get_xticklabels() + axdp70.get_yticklabels() +
             axp20.get_xticklabels() + axp20.get_yticklabels() + axp70.get_xticklabels() + axp70.get_yticklabels() +
             axz20.get_xticklabels() + axz20.get_yticklabels() + axz70.get_xticklabels() + axz70.get_yticklabels()):
    item.set_fontsize(12)

dp_plot_20.savefig('gsl-dp-20.png')
dp_plot_70.savefig('gsl-dp-70.png')
sinkage_plot_20.savefig('gsl-sinkage-20.png')
sinkage_plot_70.savefig('gsl-sinkage-70.png')
power_plot_20.savefig('gsl-power-20.png')
power_plot_70.savefig('gsl-power-70.png')

plt.show(dp_plot_20)
plt.show(dp_plot_70)
plt.show(sinkage_plot_20)
plt.show(sinkage_plot_70)
plt.show(power_plot_20)
plt.show(power_plot_70)