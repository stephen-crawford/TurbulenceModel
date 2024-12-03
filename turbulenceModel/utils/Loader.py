from random import gauss

import numpy as np
import pandas as pd
import scipy
import scipy.io
import matplotlib.pyplot as plt
import re

from matplotlib import cm
from scipy.interpolate import griddata

def load_free_flight_data():
    # Define paths
    datapath60m = '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/free_flight_data_hover_downwash/downwash_crossing_20240908_1613_sep_0.6_m.mat'
    datapath57m = '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/free_flight_data_hover_downwash/downwash_crossing_20240908_1657_sep_0.57_m.mat'
    datapath55m = '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/free_flight_data_hover_downwash/downwash_crossing_20240908_1633_sep_0.55_m.mat'
    datapath53m = '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/free_flight_data_hover_downwash/downwash_crossing_20240908_1642_sep_0.53_m.mat'

    # Load data
    data60 = scipy.io.loadmat(datapath60m)
    data57 = scipy.io.loadmat(datapath57m)
    data55 = scipy.io.loadmat(datapath55m)
    data53 = scipy.io.loadmat(datapath53m)
    datasets = [data60, data57, data55, data53]

    combined_data = []
    datanames = ["0.60", "0.57", "0.55", "0.53"]
    i = 0  #

    for data in datasets:
        sepDist = datanames[i]
        i += 1

        # Drone 1 (Lower) rows
        drone1_rows = [0, 1, 3, 5, 7, 9, 11, 13]
        drone1_data = np.array(data['states_array'])[drone1_rows, :].T
        time1 = drone1_data[:, 0]
        print("Drone 1 data shape: " + str(drone1_data.shape))

        # Drone 2 (Above) rows
        drone2_rows = [0, 2, 4, 6, 8, 10, 12, 14]
        drone2_data = np.array(data['states_array'])[drone2_rows, :].T
        time2 = drone2_data[:, 0]
        print("Drone 2 data shape: " + str(drone2_data.shape))

        # Time threshold and target x-position for Drone 1
        time_threshold = 13
        target_x_position = 1.908

        # Find the first index where Drone 1's x-position is close to 1.908 meters
        x_position_condition = np.abs(drone1_data[:, 1] - target_x_position) < 0.01
        start_idx = np.where(x_position_condition)[0][0]

        # Slice the data for Drone 1 starting from the identified index
        time1_limited = time1[start_idx:]
        drone1_data_limited = drone1_data[start_idx:, :]

        # Limit drone data to also be within the first 13 seconds
        idx1 = time1_limited < time_threshold
        time1_limited = time1_limited[idx1]
        drone1_data_limited = drone1_data_limited[idx1, :]
        drone2_data_limited = drone2_data[(time2 >= time1_limited[0]) & (time2 < time_threshold), :]

        # Combine the data for each time step
        for j in range(len(drone1_data_limited)):
            row = [sepDist, time1_limited[j]] + drone1_data_limited[j, 1:].tolist() + drone2_data_limited[j,
                                                                                      1:].tolist()
            combined_data.append(row)

        combined_data_df = pd.DataFrame(combined_data,
                                        columns=["sepDist", "time", 'drone1_x', 'drone1_y', 'drone1_z', 'drone1_roll',
                                                 'drone1_pitch', 'drone1_yaw', 'drone1_thrust',
                                                 'drone2_x', 'drone2_y', 'drone2_z', 'drone2_roll', 'drone2_pitch',
                                                 'drone2_yaw', 'drone2_thrust'])
    return combined_data_df


def load_force_data():
    lW = 0.295  # weight of a Crazyflie in newtons
    l_arm = 3.25  # arm length in cm
    path = '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/FM_plot_code/workspace.mat'
    data = scipy.io.loadmat(path)

    # Define the necessary variables
    zl_vars = ['zl13', 'zl17', 'zl21', 'zl25', 'zl29', 'zl35', 'zl40', 'zl50', 'zl60', 'zl70', 'zl80', 'zl90', 'zl100',
               'zl110']

    # Extract the 5th column of each zl variable and normalize by lW
    forces = []
    for var in zl_vars:
        if var in data:
            force = data[var][:, 4].flatten()  # Extract the 5th column (index 4)
            forces.append(force)

    lF_new_fine = np.concatenate(forces) / lW

    xl_new_fine = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 30, 60])
    yl_new_fine = np.array([13, 17, 21, 25, 29, 35, 40, 50, 60, 70, 80, 90, 100, 110])


    AlF_new_fine, BlF_new_fine = np.meshgrid(xl_new_fine / l_arm, yl_new_fine / l_arm)

    # Reshape lF_new_fine to match the shape of the meshgrid
    lF_new_fine = lF_new_fine.reshape(AlF_new_fine.shape)

    # Interpolation grid with finer resolution (0.1 increments for delta_x and delta_z)
    delta_x_fine = np.arange(np.min(AlF_new_fine), np.max(AlF_new_fine), 0.1)
    delta_z_fine = np.arange(np.min(BlF_new_fine), np.max(BlF_new_fine), 0.1)

    AlF_fine_grid, BlF_fine_grid = np.meshgrid(delta_x_fine, delta_z_fine)

    # Flatten the grids and the original data points
    points = np.column_stack((AlF_new_fine.flatten(), BlF_new_fine.flatten()))
    values = lF_new_fine.flatten()

    # Perform the interpolation (using 'cubic' method for smoother results)
    lF_interpolated = griddata(points, values, (AlF_fine_grid, BlF_fine_grid), method='cubic')

    # Create a Pandas DataFrame to store the interpolated dataset
    dataset_interpolated = pd.DataFrame({
        'delta_x/l': AlF_fine_grid.flatten(),
        'delta_z/l': BlF_fine_grid.flatten(),
        'force': lF_interpolated.flatten()
    })

    dataset_interpolated.to_csv('interpolated_force_data.csv', index=False)

    # Plotting the interpolated force data
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(AlF_fine_grid, BlF_fine_grid, lF_interpolated, 200, cmap='BuPu')
    plt.colorbar(cp)

    # Set plot labels and title
    plt.xlabel(r'Horizontal separation, ${\Delta x}/l$', fontsize=20, family='Arial', weight='bold')
    plt.ylabel(r'Vertical separation, ${\Delta z}/l$', fontsize=20, family='Arial', weight='bold')
    plt.title(r'$\bar{F_{z}}$ lower Crazyflie (Interpolated)', fontsize=20)

    plt.gca().invert_yaxis()

    plt.gca().tick_params(axis='both', which='major', labelsize=20)

    # Show the plot
    plt.show()

    # Return the interpolated dataset
    return dataset_interpolated


def add_downwash_force_to_dataset(combined_data_df):

    downwash_forces = []
    interpolated_data = pd.read_csv('interpolated_force_data.csv')

    for _, row in combined_data_df.iterrows():

        rel_x = row['drone2_x'] - row['drone1_x']
        rel_y = row['drone2_y'] - row['drone1_y']
        rel_z = row['drone2_z'] - row['drone1_z']

        delta_xy = np.sqrt((rel_x) ** 2 + (rel_y) ** 2)
        delta_z = np.sqrt((rel_z) ** 2)

        # Find the closest delta_x and delta_z in the interpolated dataset
        closest_idx = np.argmin(
            (interpolated_data['delta_x/l'] - delta_xy) ** 2 + (interpolated_data['delta_z/l'] - delta_z) ** 2
        )

        # Get the corresponding force from the interpolated data
        downwash_force = interpolated_data.iloc[closest_idx]['force']

        # Append the computed downwash force to the list
        downwash_forces.append(downwash_force)

    # Add the downwash force column to the combined data DataFrame
    combined_data_df['downwash_force_at_pos_drone1'] = downwash_forces

    return combined_data_df

def get_combined_dataframe():
    free_flight_df = load_free_flight_data()
    combined_data_df = add_downwash_force_to_dataset(free_flight_df)
    return combined_data_df


def main():
    get_combined_dataframe()

if __name__ == '__main__':
    main()

