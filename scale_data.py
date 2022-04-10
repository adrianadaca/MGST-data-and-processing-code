import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

# import data
files = glob.glob("processed data/*.csv")

# set up some lists
slips = []
g = []
avg_P_ = []
avg_dp_ = []
max_z_ = []

# set up plots
dp_plot_20, axdp20 = plt.subplots()
dp_plot_70, axdp70 = plt.subplots()
sinkage_plot_20, axz20 = plt.subplots()
sinkage_plot_70, axz70 = plt.subplots()
power_plot_20, axp20 = plt.subplots()
power_plot_70, axp70 = plt.subplots()

for i, file in enumerate(files):
    # extract slip:
    slip = re.findall(".*g_(\d{2}).*", file)
    slips.append(int(slip[0]))

    # extract g-level:
    g_level = re.findall(".*/(.*)_.*", file)
    g.append(g_level)

    g_accel = {"1-g": 9.81,  # gravitational acceleration
               "lunar-g": 1.62}
    L = {"1-g": 0.075,  # wheel radius
         "lunar-g": 0.15}
    w = {"1-g": 0.462,  # wheel angular velocity
         "lunar-g": 0.1333}
    h = {"1-g": 0.185,  # distance between center of force/torque sensor and center of wheel
         "lunar-g": 0.11}
    torque_offset = {"1-g": 0.54,  # force/torque sensor offset
                     "lunar-g": 0.42}

    # import file as pandas dataframe
    file = pd.read_csv('%s' % file)

    # calculate dimensionless time
    time_float = pd.to_timedelta(file.Timestamp).dt.total_seconds()
    t_ = time_float.multiply(np.sqrt(g_accel[g_level[0]] / L[g_level[0]]))

    # calculate non-dimensional dp
    dp_ = file.dp / file.fn

    # calculate non-dimensional sinkage
    z = file.z / 1000  # convert to m
    z_ = z / L[g_level[0]]

    # calculate motor torque from z torque
    t_m = file.current * 6.2

    # calculate non-dimensional power
    P_ = (t_m * w[g_level[0]]) / (file.fn * np.sqrt(L[g_level[0]] * g_accel[g_level[0]]))

    # smooth a bit
    # P_ = savgol_filter(P_, 41, 2)

    # making the plots
    if g_level[0] == 'lunar-g':
        style = "#1F77B4"
        t1 = 1430
        t2 = -1

        # save data to .csv (timestamp, dp, fn, sinkage, and torque)
        scaled_time = t_.multiply(np.sqrt(L['lunar-g'] / g_accel['lunar-g']))
        dp_w = pd.DataFrame(dp_)
        dp_w.set_index(scaled_time, inplace=True)
        sinkage = pd.DataFrame(z_ * L['lunar-g'] * 1000)
        sinkage.set_index(scaled_time, inplace=True)
        power = pd.DataFrame(P_ * file.fn * np.sqrt(L['lunar-g'] * g_accel['lunar-g']))
        power.set_index(scaled_time, inplace=True)

        timeseries_data = pd.concat([dp_w, sinkage, power], axis=1)
        timeseries_data.columns = ['dp', 'z', 'p']
        timeseries_data.to_csv('scaled data/lunar-g_%s-%s.csv' % (int(slip[0]), i), index=True, index_label='Timestamp')

    elif g_level[0] == '1-g':

        style = "r--"

        t_lunar = time_float.multiply(np.sqrt(1.62 / 0.15))
        t1 = np.where(t_ >= t_lunar[1430])
        t1 = t1[0][0]
        t2 = np.where(t_ >= 68)
        t2 = t2[0][0]

        scaled_time = t_.multiply(np.sqrt(L['lunar-g'] / g_accel['lunar-g']))
        dp_w = pd.DataFrame(dp_)
        dp_w.set_index(scaled_time, inplace=True)
        sinkage = pd.DataFrame(z_ * L['lunar-g'] * 1000)
        sinkage.set_index(scaled_time, inplace=True)
        power = pd.DataFrame(P_ * file.fn * np.sqrt(L['lunar-g'] * g_accel['lunar-g']))
        power.set_index(scaled_time, inplace=True)

        # save data to .csv (timestamp, dp, sinkage, and power)
        timeseries_data = pd.concat([dp_w, sinkage, power], axis=1)
        timeseries_data.columns = ['dp', 'z', 'p']
        timeseries_data.to_csv('scaled data/gsl_%s-%s.csv' % (int(slip[0]), i), index=True, index_label='Timestamp')

    # average P_
    avg_P_.append(np.mean(P_[int(t1):int(t2)]))

    # average z_
    max_z_.append(np.max(z_[int(t1):int(t2)]))

    # average dp_
    avg_dp_.append(np.mean(dp_[int(t1):int(t2)]))

    if int(slip[0]) == 20:
        axz20.plot(t_, z_, style, linewidth=0.8)
        axdp20.plot(t_, dp_, style, linewidth=0.8)
        axp20.plot(t_, P_, style, linewidth=0.8)

    if int(slip[0]) == 70:
        axz70.plot(t_, z_, style, linewidth=0.8)
        axdp70.plot(t_, dp_, style, linewidth=0.8)
        axp70.plot(t_, P_, style, linewidth=0.8)

axz20.set_ylim(-0.01, 0.2)
axz70.set_ylim(-0.01, 0.2)
axdp20.set_ylim(0, 0.9)
axdp70.set_ylim(0, 0.9)

axz20.set_xlabel(r"$\tilde{t}$", fontsize=14)
axdp20.set_xlabel(r"$\tilde{t}$", fontsize=14)
axp20.set_xlabel(r"$\tilde{t}$", fontsize=14)
axz70.set_xlabel(r"$\tilde{t}$", fontsize=14)
axdp70.set_xlabel(r"$\tilde{t}$", fontsize=14)
axp70.set_xlabel(r"$\tilde{t}$", fontsize=14)

axz20.set_ylabel(r"$\tilde{z}$", fontsize=14)
axdp20.set_ylabel(r"$\tilde{F}_{DP}$", fontsize=14)
axp20.set_ylabel(r"$\tilde{P}$", fontsize=14)
axz70.set_ylabel(r"$\tilde{z}$", fontsize=14)
axdp70.set_ylabel(r"$\tilde{F}_{DP}$", fontsize=14)
axp70.set_ylabel(r"$\tilde{P}$", fontsize=14)

custom_lines = [Line2D([0], [0], color='#1F77B4', lw=0.8),
                Line2D([0], [0], color='r', lw=0.8, linestyle='--')]

axz20.legend(custom_lines, ['Big wheel in lunar-g', 'Small wheel in 1-g'])
axz70.legend(custom_lines, ['Big wheel in lunar-g', 'Small wheel in 1-g'])
axdp20.legend(custom_lines, ['Big wheel in lunar-g', 'Small wheel in 1-g'])
axdp70.legend(custom_lines, ['Big wheel in lunar-g', 'Small wheel in 1-g'])
axp20.legend(custom_lines, ['Big wheel in lunar-g', 'Small wheel in 1-g'])
axp70.legend(custom_lines, ['Big wheel in lunar-g', 'Small wheel in 1-g'])

plt.show(dp_plot_20)
plt.show(dp_plot_70)
plt.show(sinkage_plot_20)
plt.show(sinkage_plot_70)
plt.show(power_plot_20)
plt.show(power_plot_70)

dp_plot_20.savefig('scaled-dp-20.png')
dp_plot_70.savefig('scaled-dp-70.png')
sinkage_plot_20.savefig('scaled-sinkage-20.png')
sinkage_plot_70.savefig('scaled-sinkage-70.png')
power_plot_20.savefig('scaled-power-20.png')
power_plot_70.savefig('scaled-power-70.png')

# make a file w/ all the averages:
all_data = pd.concat([pd.DataFrame(slips), pd.DataFrame(g), pd.DataFrame(avg_P_), pd.DataFrame(avg_dp_),
                      pd.DataFrame(max_z_)], axis=1)
all_data.columns = ['Slip', 'G-Level', 'Average Dimensionless Power', 'Average DP/W', 'Maximum z/L']

all_data.to_csv('scaled_gsl_averages.csv', index=False)
