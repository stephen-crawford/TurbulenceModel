from random import gauss

import numpy as np
import pandas as pd
import scipy
import scipy.io
import matplotlib.pyplot as plt
import re

from matplotlib import cm
from scipy.interpolate import griddata


def compute_induced_velocity(T_r, rho_a, A_s):
    # Induced velocity based on momentum theory
    V_ind = np.sqrt(T_r / (2 * rho_a * A_s))

    return V_ind


def compute_downwash_force(V_i, rho_a, A_s, C_T):
    print(f"Calculating downwash forces with {V_i}, {rho_a}, {A_s}, {C_T}")
    # Calculate downwash force (simplified model)
    F_downwash = rho_a * A_s * C_T * V_i ** 2

    return F_downwash

def gaussian_decay(x, y, value_to_decay, spread):
    """Apply Gaussian decay to a given value based on the radial distance."""
    r = np.sqrt(x**2 + y**2)  # Radial distance
    ratio = np.exp(-((r**2) / (2 * spread**2)))  # Gaussian function
    decayed_value = value_to_decay * ratio  # Apply decay
    return decayed_value

def bimodal_decay(x, y, value_to_decay, spread):
    """Apply Bimodal decay based on the closest rotor of the upper drone."""
    # Calculate distance to the closest rotor of Drone 2 (upper drone)
    distance_to_closest_rotor = np.sqrt((x - (22.5 / 1000))**2 + y**2)
    ratio = np.exp(-((distance_to_closest_rotor**2) / (2 * spread**2)))  # Bimodal decay
    decayed_value = value_to_decay * ratio  # Apply decay
    return decayed_value


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

    # Initialize a list to store all rows before converting to a DataFrame
    combined_data = []

    # Data names corresponding to each dataset for separation distance
    datanames = ["0.60", "0.57", "0.55", "0.53"]
    i = 0  # Initialize index for accessing datanames

    # Loop through each dataset
    for data in datasets:
        # Extract separation distance from the datanames list
        sepDist = datanames[i]
        i += 1  # Increment index for next dataset

        # Labels for the states
        labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'thrust']

        # Drone 1 (Lower) rows
        drone1_rows = [0, 1, 3, 5, 7, 9, 11, 13]  # Adjusted for Python (0-indexed)
        drone1_data = np.array(data['states_array'])[drone1_rows, :].T
        time1 = drone1_data[:, 0]
        print("Drone 1 data shape: " + str(drone1_data.shape))

        # Drone 2 (Above) rows
        drone2_rows = [0, 2, 4, 6, 8, 10, 12, 14]  # Adjusted for Python (0-indexed)
        drone2_data = np.array(data['states_array'])[drone2_rows, :].T
        time2 = drone2_data[:, 0]
        print("Drone 2 data shape: " + str(drone2_data.shape))

        # Time threshold and target x-position for Drone 1
        time_threshold = 13
        target_x_position = 1.908

        # Find the first index where Drone 1's x-position is close to 1.908 meters
        x_position_condition = np.abs(drone1_data[:, 1] - target_x_position) < 0.01
        start_idx = np.where(x_position_condition)[0][0]  # First index where Drone 1 hits 1.908 meters

        if start_idx is None:
            raise ValueError("Drone 1 never crosses the target x-position of 1.908 meters")

        # Slice the data for Drone 1 starting from the identified index
        time1_limited = time1[start_idx:]
        drone1_data_limited = drone1_data[start_idx:, :]

        # Limit Drone 1's data to the first 13 seconds after crossing the threshold
        idx1 = time1_limited < time_threshold
        time1_limited = time1_limited[idx1]
        drone1_data_limited = drone1_data_limited[idx1, :]

        # For Drone 2, align it to the same time range
        time2_limited = time2[(time2 >= time1_limited[0]) & (time2 < time_threshold)]
        drone2_data_limited = drone2_data[(time2 >= time1_limited[0]) & (time2 < time_threshold), :]

        # Combine the data for each time step
        for j in range(len(drone1_data_limited)):
            row = [sepDist, time1_limited[j]] + drone1_data_limited[j, 1:].tolist() + drone2_data_limited[j,
                                                                                      1:].tolist()
            combined_data.append(row)

        # Convert the list of rows into a pandas DataFrame
        combined_data_df = pd.DataFrame(combined_data,
                                        columns=["sepDist", "time", 'drone1_x', 'drone1_y', 'drone1_z', 'drone1_roll',
                                                 'drone1_pitch', 'drone1_yaw', 'drone1_thrust',
                                                 'drone2_x', 'drone2_y', 'drone2_z', 'drone2_roll', 'drone2_pitch',
                                                 'drone2_yaw', 'drone2_thrust'])

    print(f"Combined data shape: {combined_data_df.shape}")
    print(f"Combined data tail: {combined_data_df.tail()}")
    print(f"Combined data head: {combined_data_df.head()}")
    print(f"Combined data cols: {combined_data_df.columns}")
    return combined_data_df


def load_force_data():
    lW = 0.295  # weight of a Crazyflie in newtons
    l_arm = 3.25  # arm length in cm
    path = '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/FM_plot_code/workspace.mat'
    data = scipy.io.loadmat(path)

    # Inspect the keys to find the relevant variables (e.g., zl13, zl17, zl21, ...)
    print(data.keys())

    # Define the necessary variables
    zl_vars = ['zl13', 'zl17', 'zl21', 'zl25', 'zl29', 'zl35', 'zl40', 'zl50', 'zl60', 'zl70', 'zl80', 'zl90', 'zl100',
               'zl110']

    # Extract the 5th column of each zl variable and normalize by lW
    forces = []
    for var in zl_vars:
        if var in data:
            force = data[var][:, 4].flatten()  # Extract the 5th column (index 4)
            forces.append(force)

    # Stack all the forces vertically (in the same way as the MATLAB code does)
    lF_new_fine = np.concatenate(forces) / lW

    # Define xvals and yl_new_fine (this should be defined based on your specific data)
    xl_new_fine = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 30, 60])  # Replace with actual xvals
    yl_new_fine = np.array([13, 17, 21, 25, 29, 35, 40, 50, 60, 70, 80, 90, 100, 110])

    # Create meshgrid for plotting
    AlF_new_fine, BlF_new_fine = np.meshgrid(xl_new_fine / l_arm, yl_new_fine / l_arm)

    # Reshape lF_new_fine to match the shape of the meshgrid (this assumes a matching number of elements)
    lF_new_fine = lF_new_fine.reshape(AlF_new_fine.shape)

    # Interpolation grid with finer resolution (0.1 increments for delta_x and delta_z)
    delta_x_fine = np.arange(np.min(AlF_new_fine), np.max(AlF_new_fine), 0.1)
    delta_z_fine = np.arange(np.min(BlF_new_fine), np.max(BlF_new_fine), 0.1)

    # Create a fine meshgrid for interpolation
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

    # Optionally, save the interpolated dataset to a CSV file
    dataset_interpolated.to_csv('interpolated_force_data.csv', index=False)

    # Plotting the interpolated force data
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(AlF_fine_grid, BlF_fine_grid, lF_interpolated, 200, cmap='BuPu')
    plt.colorbar(cp)

    # Set plot labels and title
    plt.xlabel(r'Horizontal separation, ${\Delta x}/l$', fontsize=20, family='Arial', weight='bold')
    plt.ylabel(r'Vertical separation, ${\Delta z}/l$', fontsize=20, family='Arial', weight='bold')
    plt.title(r'$\bar{F_{z}}$ lower Crazyflie (Interpolated)', fontsize=20)

    # Reverse the y-axis
    plt.gca().invert_yaxis()

    # Adjust font size
    plt.gca().tick_params(axis='both', which='major', labelsize=20)

    # Show the plot
    plt.show()

    # Return the interpolated dataset
    return dataset_interpolated


# Call the function to load the data, interpolate, and create the dataset
dataset_interpolated = load_force_data()

# Inspect the first few rows of the interpolated dataset
print(dataset_interpolated.head())


def load_velocity_data():

    thrust = (0.03 * 9.81) # kg * m/s^2
    air_density = 1.2 # kg/m^3
    swept_area = np.pi * (22.5/1000)**2 # m^2
    V_i = np.sqrt(thrust / (2 * air_density * np.pi * swept_area)) # m/s

    paths = [
        '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/velocity_analysis_plots/ZoneC_directOverlap0001.dat',
        '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/velocity_analysis_plots/ZoneC_directOverlap0002.dat',
        '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/velocity_analysis_plots/ZoneD_directOverlap0001.dat',
        '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/velocity_analysis_plots/ZoneD_directOverlap0002.dat',
        '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/velocity_analysis_plots/ZoneC_MaxM0001.dat',
        '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/velocity_analysis_plots/ZoneC_MaxM0002.dat',
        '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/velocity_analysis_plots/ZoneD_maxM0001.dat',
        '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/velocity_analysis_plots/ZoneD_maxM0002.dat'
    ]

    # Corresponding separation distances: direct overlap -> 0.0m, max moment -> 0.065m
    sep_distances = [0.0, 0.0, 0.0, 0.0, 0.065, 0.065, 0.065, 0.065]

    # Initialize list to store the combined data
    all_data = []

    # Loop over the files
    for i, file_path in enumerate(paths):
        # Load the data from the file (skip the header rows)
        data = np.loadtxt(file_path, skiprows=4)

        if data is None or len(data) == 0:
            raise ValueError(f"Data could not be read from file: {file_path}")

        # Extract columns (assuming structure based on your example)
        x = data[:, 0]  # x position
        y = data[:, 1]  # y position
        vel_u = data[:, 2]  # Velocity u (m/s)
        vel_v = data[:, 3]  # Velocity v (m/s)
        vel_magnitude = data[:, 4]  # Magnitude of velocity (m/s)
        vorticity = data[:, 5]  # Vorticity (1/s)
        is_valid = data[:, 6]  # Validity flag (1 or 0)

        # Create a DataFrame for this specific dataset with separation distance as a column
        df = pd.DataFrame({
            'sepDist': [sep_distances[i]] * len(x),  # Repeat sepDist for all rows in the current file
            'x': x/1000,
            'y': y/1000,
            'Velocity_u': vel_u,
            'Velocity_v': vel_v,
            'Velocity_magnitude': vel_magnitude,
            'Vorticity': vorticity
        })

        df = df[df['Vorticity'] != 0]
        # Append this DataFrame to the all_data list
        all_data.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    velocity_data = pd.concat(all_data, ignore_index=True)

    # Print a sample to check the data
    print(f"Vel data shape: {velocity_data.shape}")
    print(f"Vel data tail: {velocity_data.tail()}")
    print(f"Vel data head: {velocity_data.head()}")
    print(f"Vel data cols: {velocity_data.columns}")

    return velocity_data


def add_downwash_force_to_dataset(combined_data_df):
    """
    Adds the downwash force as a target column for Drone 1 based on its relative position to Drone 2.
    It applies Gaussian decay if the relative z is less than the threshold, otherwise applies Bimodal decay.
    """
    downwash_forces = []
    interpolated_data = pd.read_csv('interpolated_force_data.csv')
    print("Interpolated data shape is " + str(interpolated_data.shape))
    for _, row in combined_data_df.iterrows():
        # Compute the relative position between Drone 1 and Drone 2
        rel_x = row['drone2_x'] - row['drone1_x']
        rel_y = row['drone2_y'] - row['drone1_y']
        rel_z = row['drone2_z'] - row['drone1_z']

        # Compute horizontal (delta_x) and vertical (delta_z) separations
        delta_x = np.abs(rel_x)  # Horizontal separation (distance)
        delta_z = np.abs(rel_z)  # Vertical separation (distance)

        # Find the closest delta_x and delta_z in the interpolated dataset
        closest_idx = np.argmin(
            (interpolated_data['delta_x/l'] - delta_x) ** 2 + (interpolated_data['delta_z/l'] - delta_z) ** 2
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
    force_data_df = load_force_data()
    combined_data_df = add_downwash_force_to_dataset(free_flight_df)
    return combined_data_df


def main():
    free_flight_df = load_free_flight_data()
    #vel_data_df = load_velocity_data()
    combined_data_df = add_downwash_force_to_dataset(free_flight_df)
    print("Combined data sghape is " + str(combined_data_df.shape))
    # Print the result
    print(combined_data_df[['time', 'downwash_force_at_pos_drone1']].head())
    print(f"Combined data columns: {combined_data_df.columns}")
    force = load_force_data()

if __name__ == '__main__':
    main()

