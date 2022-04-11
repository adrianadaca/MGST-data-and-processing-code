import pandas as pd
import glob
import re

import numpy as np
import pandas as pd
from bokeh.io import export_png
from bokeh.models import HoverTool
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.plotting import figure, show
from scipy.signal import savgol_filter

from average import average
from interpolate import interpolate

# specify directory where the FTS files, trio files, FTS offsets, and slope data are
directory = 'GSL data'

# get FTS and trio files
files = sorted(glob.glob("%s/*combined.CSV" % directory))
FTS_files = sorted(glob.glob("%s/*.csv" % directory))

# get force/torque sensor offsets - offsets file lists x, y, then z offset
offsets = pd.read_csv('offsets.csv', header=None)
x_offset = float(offsets.iloc[0])
y_offset = float(offsets.iloc[1])
z_offset = float(offsets.iloc[2])

# set up lists for average/max values
slips = []
sinkage = []
current = []
timestamps = []
repeats = []
dates = []
date_times = []
dp_w = []

# set up plots
sinkage_plot = figure(toolbar_location=None, x_axis_label='Elapsed Time (s)', y_axis_label='Sinkage (mm)',
                      x_range=(1, 27000), y_range=(-2, 19.5))  # 31
dp_plot = figure(toolbar_location=None, x_axis_label='Elapsed Time (s)', y_axis_label='DP/W', x_range=(1, 27000),
                 y_range=(0, 0.65))
fn_plot = figure(toolbar_location=None, x_axis_label='Elapsed Time (s)', y_axis_label='Normal Force (N)',
                 x_range=(1, 27000), y_range=(0,205))


for i, file in enumerate(files):
    # read files and get slip value from filename
    slip = re.findall(".*s(\d{2}).*", file)
    slips.append(int(slip[0]))
    file = pd.read_csv('%s' % file)
    fts_file = pd.read_csv('%s' % FTS_files[i])

    # set up dataframe index
    file.set_index(pd.DatetimeIndex(file['timestamp']), inplace=True)

    # smooth the sinkage data
    file.sinkage = savgol_filter(file.sinkage, 201, 2)

    # calculate sinkage in mm from potentiometer counts
    file.sinkage = (file.sinkage - 110) / 132.24
    file.sinkage = file.sinkage - file.sinkage[0]

    # correct for FTS offsets
    fts_file.force_x = -fts_file.force_x + x_offset
    fts_file.force_y = -fts_file.force_y + y_offset
    fts_file.force_z = -fts_file.force_z + z_offset

    # friction correction
    fts_file.force_x = fts_file.force_x + 0.25 * fts_file.force_z

    # automatically detect beginning of test from FTS file:
    slope = np.diff(fts_file.force_x)/np.diff(fts_file.index)
    start = np.where(slope >= 1)[0][0]

    # t1 and t2 indices used for averaging (according to averaging procedure outlined in Niksirat, P., Daca, A., & Skonieczny, K. (2020). The effects of reduced-gravity on planetary rover mobility. The International Journal of Robotics Research, 39(7), 797-811.
    t1 = 500
    t2 = 4000

    [DP_avg, FN_avg, FZ_avg, DP_W_avg, i1, i2] = average(fts_file, start, t1, t2)

    fts_file.set_index(pd.DatetimeIndex(fts_file['timestamp']), inplace=True)

    # smooth
    fts_file.force_x[start:i2] = savgol_filter(fts_file.force_x[start:i2], 41, 2)  # window 41, order 2
    fts_file.force_y[start:i2] = savgol_filter(fts_file.force_y[start:i2], 41, 2)  # window 41, order 2

    # select color and linestyle based on slip
    if int(slip[0]) == 20:
        legend = "20% slip"
        color = "#1F77B4"
        line_dash = "solid"
    if int(slip[0]) == 70:
        legend = "70% slip"
        color = "red"
        line_dash = "dashed"

    # set up data to export avg/max values
    sinkage.append(np.max(file.sinkage))
    current.append(np.mean(file.maxon_current/1000))  # convert mA to A

    timestamp = re.findall(".*-(\d{6})-.*", files[i])
    timestamps.append(int(timestamp[0]))

    date = re.findall(".*(\d{8})-.*", files[i])
    date_times.append(int(date[0] + timestamp[0]))
    date = int(date[0])
    dates.append(date)
    dp_w.append(DP_W_avg)

    # plot sinkage, dp, and normal force
    sinkage_plot.line(file.index - file.index[0], file.sinkage, legend=legend, line_color=color,
                      line_dash=line_dash)
    dp_plot.line(fts_file.index[i1:i2] - fts_file.index[start],
                 fts_file.force_x[i1:i2] / fts_file.force_y[i1:i2],
                 legend=legend, line_color=color, line_dash=line_dash)
    fn_plot.line(fts_file.index[i1:i2] - fts_file.index[start],
                 fts_file.force_y[i1:i2], legend=legend, line_color=color, line_dash=line_dash)

    # set up data to export timeseries to csv
    fn = pd.DataFrame(fts_file.force_y[start:i2])
    fn.set_index(fts_file.index[start:i2] - fts_file.index[start], inplace=True)
    dp = pd.DataFrame(fts_file.force_x[start:i2])
    dp.set_index(fts_file.index[start:i2] - fts_file.index[start], inplace=True)
    dp_raw = pd.DataFrame(fts_file.force_x[start:i2] - 0.25 * fts_file.force_z[start:i2])
    dp_raw.set_index(fts_file.index[start:i2] - fts_file.index[start], inplace=True)
    torque = pd.DataFrame(fts_file.torque_z[start:i2])
    torque.set_index(fts_file.index[start:i2] - fts_file.index[start], inplace=True)

    # interpolate sinkage and current data to timestamps of the force/torque sensor data
    sink = interpolate(file.index - file.index[0], fn.index, file.sinkage)
    sink = pd.DataFrame(sink)
    sink.set_index(fn.index, inplace=True)
    crnt = interpolate(file.index - file.index[0], fn.index, file.maxon_current)
    crnt = pd.DataFrame(crnt / 1000)
    crnt.set_index(fn.index, inplace=True)
    size = np.amin(np.array([fn.size, dp.size, sink.size]))

    # save data to .csv (timestamp, dp, fn, sinkage, and torque)
    timeseries_data = pd.concat([dp[0:size], fn[0:size], sink[0:size], torque[0:size], dp_raw[0:size], crnt[0:size]],
                                axis=1)
    timeseries_data.columns = ['dp', 'fn', 'z', 'torque', 'dp_raw', 'current']
    timeseries_data.to_csv('processed data/1-g_%s-%s.csv' % (int(slip[0]), int(timestamp[0])), index=True, index_label='Timestamp')

# format and export plots
plots = [sinkage_plot, dp_plot, fn_plot]
names = ['sinkage', 'dp', 'fn']

for i, plot in enumerate(plots):
    plot.xaxis.formatter = DatetimeTickFormatter(minsec=['%M:%S'])
    hover_tool = HoverTool()
    plot.tools.append(hover_tool)
    plot.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
        ("Time (ms)", "$x"),
        ("Value", "$y")
    ], formatters=dict(x='datetime')))
    plot.xaxis.axis_label_text_font_size = "14pt"
    plot.xaxis.major_label_text_font_size = "14pt"
    plot.yaxis.axis_label_text_font_size = "14pt"
    plot.yaxis.major_label_text_font_size = "14pt"
    plot.legend.label_text_font_size = "14pt"
    show(plot)
    export_png(plot, filename="%s-%s.png" % (names[i], directory))

# save average/max data
all_data = pd.concat([pd.DataFrame(date_times), pd.DataFrame(slips), pd.DataFrame(sinkage), pd.DataFrame(current),
                      pd.DataFrame(dp_w)], axis=1)
all_data.columns = ['Date_Time', 'Slip', 'Max_Sinkage', 'Average_Current', 'Average DP/W']
print(all_data)

all_data['Repeat'] = all_data.groupby(['Slip'])['Date_Time'].rank(method='first')
all_data['Repeat'] = all_data['Repeat'].map(lambda x: chr(ord('`') + int(x)))

all_data.to_csv('%s_averages.csv' % directory, index=False)