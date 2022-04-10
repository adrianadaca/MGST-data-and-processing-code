import math

import numpy as np
import pandas as pd
from bokeh.io import export_png
from bokeh.models import HoverTool
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.plotting import figure, show
from scipy.signal import savgol_filter

# local imports
from interpolate import interpolate

# lunar results
# Force/torque sensor data
L20_d4_a = pd.read_csv('Flight data/20201002 131424 sliptest s20-p1500-d4.csv')
L20_d4_b = pd.read_csv('Flight data/20201002 132114 sliptest s20-p1500-d4.csv')
L20_d4_c = pd.read_csv('Flight data/20201002 135545 sliptest s20-p1500-d4.csv')
L20_d4_d = pd.read_csv('Flight data/20201002 140110 sliptest s20-p1500-d4.csv')
L70_d4_a = pd.read_csv('Flight data/20201002 134029 sliptest s70-p1500-d4.csv')
L70_d4_b = pd.read_csv('Flight data/20201002 134618 sliptest s70-p1500-d4.csv')
L70_d4_c = pd.read_csv('Flight data/20201002 135133 sliptest s70-p1500-d4.csv')

# trio data (contains motor current and sinkage)
L20_d4_a_s = pd.read_csv('Flight data/20201002-131424-trio-s20-p1500-d4_combined.CSV')
L20_d4_b_s = pd.read_csv('Flight data/20201002-132114-trio-s20-p1500-d4_combined.CSV')
L20_d4_c_s = pd.read_csv('Flight data/20201002-135545-trio-s20-p1500-d4_combined.CSV')
L20_d4_d_s = pd.read_csv('Flight data/20201002-140110-trio-s20-p1500-d4_combined.CSV')
L70_d4_a_s = pd.read_csv('Flight data/20201002-134029-trio-s70-p1500-d4_combined.CSV')
L70_d4_b_s = pd.read_csv('Flight data/20201002-134618-trio-s70-p1500-d4_combined.CSV')
L70_d4_c_s = pd.read_csv('Flight data/20201002-135133-trio-s70-p1500-d4_combined.CSV')

# make some lists
slip20_files = [L20_d4_a, L20_d4_b, L20_d4_c, L20_d4_d]
slip70_files = [L70_d4_a, L70_d4_b, L70_d4_c]
all_files = [slip20_files, slip70_files]

slip20_trio_files = [L20_d4_a_s, L20_d4_b_s, L20_d4_c_s, L20_d4_d_s]
slip70_trio_files = [L70_d4_a_s, L70_d4_b_s, L70_d4_c_s]
all_trio_files = [slip20_trio_files, slip70_trio_files]

# Force/torque sensor offsets
x_offset = 13
y_offset = 9.8
z_offset = -12

# Indices where motion starts
starttimes_slip20_index = [23850, 19802, 19098, 19741]
starttimes_slip70_index = [34498, 27681, 24328]
endtimes_slip20_index = np.array(starttimes_slip20_index) + 7000
endtimes_slip70_index = np.array(starttimes_slip70_index) + 7000
starttimes_slip20 = [25172, 21160, 20406, 21068]
starttimes_slip70 = [35850, 29074, 25694]
all_starttimes = [starttimes_slip20_index, starttimes_slip70_index]
endtimes_slip20 = [29167, 25151, 24334, 25049]
endtimes_slip70 = [39788, 33232, 29836]
all_endtimes = [endtimes_slip20, endtimes_slip70]

# measured soil slopes (for slope correction)
slopes_slip20 = np.array([[6.48, 8.79, 6.81, 8.28],  # measured soil angle at beginning,
          [4.68, 5.91, 7.97, 6.39],  # middle, and
          [4.1, 4.2, 5.78, 3.88]])  # end of each test

slopes_slip70 = np.array([[6.04, 7.2, 6.24],  # measured soil angle at beginning,
          [7.54, 8.1, 6.05],  # middle, and
          [6.26, 6.82, 4.87]])  # end of each test

# measured soil heights (for slope correction)
delta_y_slip20 = np.array([21.47, 38.65, 44.48, 33.82])
delta_y_slip70 = np.array([13.04, 13.04, 10.14])

slip20_soilheight = np.array([[7.43, 4.25, 2.48, 4.25, 5.66],
                  [6.73, 9.56, 7.79, 7.08, 3.54],
                  [8.5, 9.2, 7.79, 9.91, 8.5],
                  [6.02, 9.56, 7.08, 5.31, 14.16]])

slip70_soilheight = np.array([[3.54, 4.25, 4.6, 2.83, 8.85],
                   [4.25, 4.25, 3.19, 1.77, 3.54],
                   [3.89, 1.42, 2.48, 1.06, 1.06]])

all_slopes = [slopes_slip20, slopes_slip70]

# process data
for i in range(0, len(all_files)):
    files = all_files[i]
    trio_files = all_trio_files[i]
    starttimes = all_starttimes[i]
    endtimes = all_endtimes[i]
    slopes = all_slopes[i]
    for ii in range(0, len(files)):
        # set dataframe indices
        files[ii].set_index(pd.DatetimeIndex(files[ii]['timestamp']), inplace=True)
        trio_files[ii].set_index(pd.DatetimeIndex(trio_files[ii]['timestamp']), inplace=True)

        # convert indices to elapsed seconds
        files[ii].index = files[ii].index - files[ii].index[0]
        trio_files[ii].index = trio_files[ii].index - trio_files[ii].index[0]

        # correct for FTS offsets
        files[ii].force_x = -files[ii].force_x + x_offset
        files[ii].force_y = -files[ii].force_y + y_offset
        files[ii].force_z = -files[ii].force_z + z_offset

        # friction correction
        files[ii].force_x = files[ii].force_x + 0.25 * files[ii].force_z
        
        # slope correction for the drawbar pull data
        length = len(files[ii].force_x[starttimes[ii]:endtimes[ii]])
        midpoint = math.floor(length / 2)
        first_half = np.linspace(slopes[0, ii], slopes[1, ii], midpoint)

        if length % 2 == 0:
            second_half = np.linspace(slopes[1, ii], slopes[2, ii], midpoint)
        else:
            second_half = np.linspace(slopes[1, ii], slopes[2, ii], midpoint + 1)

        slope_angles = np.zeros(length)
        slope_angles[0:midpoint] = first_half
        slope_angles[midpoint:length] = second_half

        files[ii].force_x[starttimes[ii]:endtimes[ii]] = \
            files[ii].force_x[starttimes[ii]:endtimes[ii]] - \
            files[ii].force_y[starttimes[ii]:endtimes[ii]] * \
            np.cos(np.radians(slope_angles)) * np.sin(np.radians(slope_angles))

        # smooth
        files[ii].force_x = savgol_filter(files[ii].force_x, 41, 2)  # window 41, order 2
        files[ii].force_y = savgol_filter(files[ii].force_y, 41, 2)  # window 41, order 2

        # convert potentiometer counts to mm of sinkage and smooth
        trio_files[ii].sinkage = savgol_filter(trio_files[ii].sinkage, 201, 2)  # window size 41, polynomial order 2
        trio_files[ii].sinkage = (trio_files[ii].sinkage - 110) / 132.24
        trio_files[ii].sinkage = trio_files[ii].sinkage - trio_files[ii].sinkage[0]

        # smooth the current data
        trio_files[ii].maxon_current = savgol_filter(trio_files[ii].maxon_current, 41, 2)  # window 41, order 2


# plot DP/W
# 20% slip in blue, 70% slip in red.
ylabel = 'DP/W'
plot3 = figure(toolbar_location=None, x_axis_label='Elapsed Time (s)', y_axis_label=ylabel, x_range=(1, 27000), y_range=(0, 0.65))
#
for j in range(0, len(slip20_files)):
    plot3.line(slip20_files[j].index[starttimes_slip20[j]:endtimes_slip20[j]] -
               slip20_files[j].index[starttimes_slip20_index[j]],
               slip20_files[j].force_x[starttimes_slip20[j]:endtimes_slip20[j]] /
               slip20_files[j].force_y[starttimes_slip20[j]:endtimes_slip20[j]],
               legend="20% slip")
#
for k in range(0, len(slip70_files)):
    plot3.line(slip70_files[k].index[starttimes_slip70[k]:endtimes_slip70[k]] -
              slip70_files[k].index[starttimes_slip70_index[k]],
              slip70_files[k].force_x[starttimes_slip70[k]:endtimes_slip70[k]] /
              slip70_files[k].force_y[starttimes_slip70[k]:endtimes_slip70[k]],
              line_color="red", line_dash='dashed', legend="70% slip")

# format plot and export to png
plot3.xaxis.formatter = DatetimeTickFormatter(minsec=['%M:%S'])
hover_tool = HoverTool()
plot3.tools.append(hover_tool)
plot3.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
                            ("Time (ms)", "$x"),
                            ("Value", "$y")
                        ], formatters=dict(x='datetime')))
plot3.xaxis.axis_label_text_font_size = "14pt"
plot3.xaxis.major_label_text_font_size = "14pt"
plot3.yaxis.axis_label_text_font_size = "14pt"
plot3.yaxis.major_label_text_font_size = "14pt"
plot3.legend.label_text_font_size = "14pt"
show(plot3)
export_png(plot3, filename="drawbar-pull-lunar.png")

# plot sinkage
ylabel = 'Sinkage (mm)'
plot = figure(toolbar_location=None, x_axis_label='Elapsed Time (s)', y_axis_label=ylabel, x_range=(1, 27000), y_range=(-2, 31))

for j in range(0, len(slip20_trio_files)):
    # apply slope correction to sinkage data
    # get angle at each point (same procedure used for FTS data, applied to trio time stamps)
    length = len(slip20_trio_files[j].sinkage)

    # make vector with soil height values from 0-5s, 5-10s, 10-15s, 15-20s, and 20s-end:
    delta_y = np.zeros(length)
    delta_y[0:500] = np.linspace(0, slip20_soilheight[j, 0], 500)
    delta_y[500:1000] = np.linspace(0, slip20_soilheight[j, 1], 500) + slip20_soilheight[j, 0]
    delta_y[1000:1500] = np.linspace(0, slip20_soilheight[j, 2], 500) + slip20_soilheight[j, 1]\
                         + slip20_soilheight[j, 0]
    delta_y[1500:2000] = np.linspace(0, slip20_soilheight[j, 3], 500) + slip20_soilheight[j, 2]\
                         + slip20_soilheight[j, 1] + slip20_soilheight[j, 0]
    delta_y[2000:length] = np.linspace(0, slip20_soilheight[j, 4], length-2000) + slip20_soilheight[j, 3] \
                         + slip20_soilheight[j, 2] + slip20_soilheight[j, 1] + slip20_soilheight[j, 0]

    # plot sinkage data with slope correction
    plot.line(slip20_trio_files[j].index[0:length-500], slip20_trio_files[j].sinkage[0:length-500] -
             delta_y[0:length-500], legend="20% slip")

    # set up dataframes to export to .csv
    fn = pd.DataFrame(slip20_files[j].force_y[starttimes_slip20_index[j]:endtimes_slip20[j]])
    fn.set_index(slip20_files[j].index[starttimes_slip20_index[j]:endtimes_slip20[j]] -
                 slip20_files[j].index[starttimes_slip20_index[j]], inplace=True)
    dp = pd.DataFrame(slip20_files[j].force_x[starttimes_slip20_index[j]:endtimes_slip20[j]])
    dp.set_index(slip20_files[j].index[starttimes_slip20_index[j]:endtimes_slip20[j]] -
                 slip20_files[j].index[starttimes_slip20_index[j]], inplace=True)
    dp_raw = pd.DataFrame(slip20_files[j].force_x[starttimes_slip20_index[j]:endtimes_slip20[j]] - 0.25 * \
             slip20_files[j].force_x[starttimes_slip20_index[j]:endtimes_slip20[j]])
    dp_raw.set_index(slip20_files[j].index[starttimes_slip20_index[j]:endtimes_slip20[j]] -
                 slip20_files[j].index[starttimes_slip20_index[j]], inplace=True)
    torque = pd.DataFrame(slip20_files[j].torque_z[starttimes_slip20_index[j]:endtimes_slip20[j]])
    torque.set_index(slip20_files[j].index[starttimes_slip20_index[j]:endtimes_slip20[j]] -
                 slip20_files[j].index[starttimes_slip20_index[j]], inplace=True)

    # interpolate sinkage and current data to the timestamps of the force/torque sensor data
    sink = interpolate(slip20_trio_files[j].index[0:length - 500] - slip20_trio_files[j].index[0], fn.index,
                       slip20_trio_files[j].sinkage[0:length - 500] - delta_y[0:length - 500])
    sink = pd.DataFrame(sink)
    sink.set_index(fn.index, inplace=True)

    current = interpolate(slip20_trio_files[j].index[0:length - 500] - slip20_trio_files[j].index[0], fn.index,
                       slip20_trio_files[j].maxon_current[0:length - 500])
    current = pd.DataFrame(current / 1000)
    current.set_index(fn.index, inplace=True)

    # export processed timeseries data to csv
    size = np.amin(np.array([fn.size, dp.size, sink.size]))
    timeseries_data = pd.concat([dp[0:size], fn[0:size], sink[0:size], torque[0:size], dp_raw[0:size], current[0:size]]
                                , axis=1)
    timeseries_data.columns = ['dp', 'fn', 'z', 'torque', 'dp_raw', 'current']
    timeseries_data.to_csv('processed data/lunar-g_20-%s.csv' % j, index=True, index_label='Timestamp')

for k in range(0, len(slip70_trio_files)):
    # apply slope correction to sinkage data
    # get angle at each point (same procedure used for FTS data, applied to trio time stamps)
    length = len(slip70_trio_files[k].sinkage)

    # make vector with soil height values from 0-5s, 5-10s, 10-15s, 15-20s, and 20s-end:
    delta_y = np.zeros(length)
    delta_y[0:500] = np.linspace(0, slip70_soilheight[k, 0], 500)
    delta_y[500:1000] = np.linspace(0, slip70_soilheight[k, 1], 500) + slip70_soilheight[k, 0]
    delta_y[1000:1500] = np.linspace(0, slip70_soilheight[k, 2], 500) + slip70_soilheight[k, 1] \
                         + slip70_soilheight[k, 0]
    delta_y[1500:2000] = np.linspace(0, slip70_soilheight[k, 3], 500) + slip70_soilheight[k, 2] \
                         + slip70_soilheight[k, 1] + slip70_soilheight[k, 0]
    delta_y[2000:length] = np.linspace(0, slip70_soilheight[k, 3], length - 2000) + slip70_soilheight[k, 3] \
                       + slip70_soilheight[k, 2] + slip70_soilheight[k, 1] + slip70_soilheight[k, 0]

    # plot sinkage data with slope correction
    plot.line(slip70_trio_files[k].index[0:length-500], slip70_trio_files[k].sinkage[0:length-500] -
              delta_y[0:length-500], line_color="red",
              line_dash='dashed', legend="70% slip")

    # set up dataframes to export to .csv
    fn = pd.DataFrame(slip70_files[k].force_y[starttimes_slip70_index[k]:endtimes_slip70[k]])
    fn.set_index(slip70_files[k].index[starttimes_slip70_index[k]:endtimes_slip70[k]] -
                 slip70_files[k].index[starttimes_slip70_index[k]], inplace=True)
    dp = pd.DataFrame(slip70_files[k].force_x[starttimes_slip70_index[k]:endtimes_slip70[k]])
    dp.set_index(slip70_files[k].index[starttimes_slip70_index[k]:endtimes_slip70[k]] -
                 slip70_files[k].index[starttimes_slip70_index[k]], inplace=True)
    dp_raw = pd.DataFrame(slip70_files[k].force_x[starttimes_slip70_index[k]:endtimes_slip70[k]] - 0.25 * \
             slip70_files[k].force_x[starttimes_slip70_index[k]:endtimes_slip70[k]])
    dp_raw.set_index(slip70_files[k].index[starttimes_slip70_index[k]:endtimes_slip70[k]] -
                     slip70_files[k].index[starttimes_slip70_index[k]], inplace=True)
    torque = pd.DataFrame(slip70_files[k].torque_z[starttimes_slip70_index[k]:endtimes_slip70[k]])
    torque.set_index(slip70_files[k].index[starttimes_slip70_index[k]:endtimes_slip70[k]] -
                 slip70_files[k].index[starttimes_slip70_index[k]], inplace=True)

    # interpolate sinkage and current data to the timestamps of the force/torque sensor data
    sink = interpolate(slip70_trio_files[k].index[0:length - 500] - slip70_trio_files[k].index[0], fn.index,
                       slip70_trio_files[k].sinkage[0:length - 500] - delta_y[0:length - 500])
    sink = pd.DataFrame(sink)
    sink.set_index(fn.index, inplace=True)

    current = interpolate(slip70_trio_files[k].index[0:length - 500] - slip70_trio_files[k].index[0], fn.index,
                       slip70_trio_files[k].maxon_current[0:length - 500])
    current = pd.DataFrame(current / 1000)
    current.set_index(fn.index, inplace=True)

    # export processed timeseries data to csv
    size = np.amin(np.array([fn.size, dp.size, sink.size]))
    timeseries_data = pd.concat([dp[0:size], fn[0:size], sink[0:size], torque[0:size], dp_raw[0:size], current[0:size]],
                                axis=1)
    timeseries_data.columns = ['dp', 'fn', 'z', 'torque', 'dp_raw', 'current']
    timeseries_data.to_csv('processed data/lunar-g_70-%s.csv' % k, index=True, index_label='Timestamp')

# format plot and export to png
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
export_png(plot, filename="sinkage-lunar.png")
