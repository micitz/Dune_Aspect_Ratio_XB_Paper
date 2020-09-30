"""
All the data sources are scattered around the D drive, this script
organizes it and consolidates it into the "Data" subfolder in the
"Chapter 2 Dune Aspect Ratio" folder.

Michael Itzkin, 5/6/2020
"""

import shutil as sh
import pandas as pd
import numpy as np
import os

# Set the data directory to save files into
DATA_DIR = os.path.join('..', 'Data')

# Set the directory with most of the XBeach data
XB_DIR = os.path.join('..', '..', 'XBeach Modelling', 'Dune Complexity Experiments')


def bogue_lidar_data():
    """
    Load all Bogue Banks morphometrics from 1997-2016
    and return a dataframe of aspect ratios and natural
    dune volumes
    """

    # Set a list of years
    years = [1997, 1998, 1999, 2000, 2004, 2005, 2010, 2011, 2014, 2016]

    # Set an empty dataframe
    morpho = pd.DataFrame()

    # Loop through the years and load the data
    for year in years:

        # Set a path to the data and load
        path = os.path.join('..', '..', 'Chapter 1 Sand Fences', 'Data', f'Morphometrics for Bogue {year}.csv')
        temp = pd.read_csv(path, delimiter=',', header=0)

        # Add a column for the year
        temp['Year'] = year

        # Append the data to the main dataframe
        morpho = pd.concat([morpho, temp])

    # Make a new dataframe with just aspect ratios and volumes
    data = pd.DataFrame()
    data['Year'] = morpho['Year']
    data['Ratio'] = (morpho['y_crest'] - morpho['y_toe']) / (morpho['x_heel'] - morpho['x_toe'])
    data['Volume'] = morpho['Natural Dune Volume']

    # Save the Dataframe to the data folder
    save_name = os.path.join(DATA_DIR, 'Bogue Banks Volumes and Aspect Ratios.csv')
    data.to_csv(save_name, index=False)
    print(f'File Saved: {save_name}')


def initial_profiles():
    """
    Take all the initial profiles and place them
    into a Dataframe to save as a .csv

    Make a column for the experiment names, a column for
    the X-grids, and columns for the profiles
    """

    # Set the experiment names. The initial profiles are the same regardless of
    # the surge level so just take from the half surge simulations
    experiments = ['Toes Joined', 'Crests Joined', 'Heels Joined', 'Fenced']

    # Set an empty dataframe
    profiles = pd.DataFrame()

    # Loop through the experiments
    for experiment in experiments:

        # Set a path to the profiles
        PROFILE_DIR = os.path.join(XB_DIR, f'{experiment} Half Surge')

        # Load the x-grid
        x_grid_fname = os.path.join(PROFILE_DIR, 'Dune Complexity 1 1', 'x.grd')
        x_grid = np.loadtxt(x_grid_fname)

        # Load the dunes
        dune_1 = np.loadtxt(fname=os.path.join(PROFILE_DIR, 'Dune Complexity 1 1', 'bed.dep'))
        dune_2 = np.loadtxt(fname=os.path.join(PROFILE_DIR, 'Dune Complexity 20 1', 'bed.dep'))
        dune_3 = np.loadtxt(fname=os.path.join(PROFILE_DIR, 'Dune Complexity 40 1', 'bed.dep'))
        dune_4 = np.loadtxt(fname=os.path.join(PROFILE_DIR, 'Dune Complexity 60 1', 'bed.dep'))
        dune_5 = np.loadtxt(fname=os.path.join(PROFILE_DIR, 'Dune Complexity -20 1', 'bed.dep'))
        dune_6 = np.loadtxt(fname=os.path.join(PROFILE_DIR, 'Dune Complexity -40 1', 'bed.dep'))
        dune_7 = np.loadtxt(fname=os.path.join(PROFILE_DIR, 'Dune Complexity -60 1', 'bed.dep'))

        # Put all of the stretched dunes into a dataframe
        dune_dict = {
            'Experiment': experiment.replace('Joined', 'Aligned'),
            'X': x_grid,
            '1 pct': dune_1,
            '20 pct': dune_2,
            '40 pct': dune_3,
            '60 pct': dune_4,
            '-20 pct': dune_5,
            '-40 pct': dune_6,
            '-60 pct': dune_7,
        }
        dune_data = pd.DataFrame(data=dune_dict)

        # Concatenate the Dataframes
        profiles = pd.concat([profiles, dune_data])

    # Save the Dataframe to the data folder
    save_name = os.path.join(DATA_DIR, 'Initial Profiles.csv')
    profiles.to_csv(save_name, index=False)
    print(f'File Saved: {save_name}')


def initial_ratios():
    """
    Make a .csv file with the initial dune aspect ratios and
    dune volumes for the profiles used in the simulations
    """

    # Set the experiment names. The initial profiles are the same regardless of
    # the surge level so just take from the half surge simulations
    experiments = ['Toes Joined', 'Crests Joined', 'Heels Joined', 'Fenced']

    # Set an empty dataframe
    ratios = pd.DataFrame()

    # Loop through the experiments
    for experiment in experiments:

        # Load the initial dune ratios
        init_ratio_fname = os.path.join(XB_DIR, f'{experiment} Half Surge', 'Setup Data', 'Initial Dune Ratios.csv')
        init_ratios = pd.read_csv(init_ratio_fname, delimiter=',', header=None, names=['Stretch', 'Ratio', 'Volume'])

        # Add a column for the experiment name
        init_ratios['Experiment'] = experiment.replace('Joined', 'Aligned')

        # Concatenate the data
        ratios = pd.concat([ratios, init_ratios])

    # Save the Dataframe to the data folder
    save_name = os.path.join(DATA_DIR, 'Initial Dune Ratios.csv')
    ratios.to_csv(save_name, index=False)
    print(f'File Saved: {save_name}')


def joaquin_and_florence():
    """
    Load the storm surge time series' from
    Tropical Storm Joaquin and Hurricane
    Florence, put them in a .csv file
    """

    # Loop through the storms
    for storm in ['Joaquin', 'Florence']:

        # Load the tide predictions and observations as a Pandas dataframe
        filename = os.path.join(XB_DIR, 'Setup Data', f'{storm}.csv')
        if storm == 'Joaquin':
            parse_dates_cols = ['Date', 'Time']
            data_columns = ['Time', 'Predicted', 'Observed']
        else:
            parse_dates_cols = ['Date', 'Time (GMT)']
            data_columns = ['Time', 'Predicted', 'Preliminary', 'Observed']
        data = pd.read_csv(filename, delimiter=',', parse_dates=[parse_dates_cols], header=0)
        data.columns = data_columns

        # Calculate the non-tidal residual
        data['NTR'] = data['Observed'] - data['Predicted']

        # Load the time data
        times = data['Time'].tolist()
        data['String Times'] = [t.strftime('%Y-%m-%d %H') for t in times]

        # Save the DataFrame as a .csv
        save_name = os.path.join(DATA_DIR, f'{storm}.csv')
        data.to_csv(save_name, index=False)


def move_csv_output():
    """
    Take the .csv files and move them into the "Data" folder,
    then rename them from "xboutput.nc" to the name of the simulation
    """

    # Set lists with the dune configurations, storm surge
    # modifications, storm duration increases, and dune aspect
    # ratio stretches
    dunes = ['Toes Joined', 'Crests Joined', 'Heels Joined', 'Fenced']
    surges = ['Half', 'Normal', 'One Half']
    durations = [1, 12, 18, 24, 36, 48]
    stretches = [-60, -40, -20, 1, 20, 40, 60]

    # Loop through the dunes and surges
    for dune in dunes:
        for surge in surges:

            # Set the experiment folder name
            experiment_name = f'{dune} {surge} Surge'
            experiment_folder = os.path.join(XB_DIR, experiment_name)

            # Make a target folder to move the runs into
            save_folder = os.path.join(DATA_DIR, 'XBeach Morphometrics', experiment_name)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            # Loop through the dunes and durations within the experiment
            for stretch in stretches:
                for duration in durations:

                    # Set the simulation folder
                    run_name = f'Dune Complexity {stretch} {duration}'
                    simulation_folder = os.path.join(experiment_folder, run_name)

                    # Set the XBeach output file as the source. Set the destination
                    # name. Then copy the file over
                    source = os.path.join(simulation_folder, f'{run_name} Morphometrics.csv')
                    if os.path.exists(source):
                        destination = os.path.join(save_folder, f'{run_name} Morphometrics.csv')
                        if not os.path.exists(destination):
                            sh.copy(source, destination)
                            print(f'File Successfully Copied: {destination}')
                        else:
                            print(f'File already exists: {destination}')
                    else:
                        print(f'FILE DOES NOT EXIST: {source}')


def move_field_data():
    """
    Move the field data morphometrics from 2017
    and 2018 into the data folder
    """

    # Set the years
    years = [2017, 2018]

    # Set a path to the field data
    field_dir = os.path.join('..', '..', 'Bogue Banks Field Data')

    # Loop through the years
    for year in years:

        # Identify the source file
        source = os.path.join(field_dir, str(year), f'Morphometrics for Bogue Banks {year}.csv')

        # Set the target
        destination = os.path.join(DATA_DIR, f'Morphometrics for Bogue Banks {year}.csv')

        # Copy the file
        sh.copy(source, destination)


def move_netcdf_output():
    """
    Take the netCDF files and move them into the "Data" folder,
    then rename them from "xboutput.nc" to the name of the simulation
    """

    # Set lists with the dune configurations, storm surge
    # modifications, storm duration increases, and dune aspect
    # ratio stretches
    dunes = ['Toes Joined', 'Crests Joined', 'Heels Joined', 'Fenced']
    surges = ['Half', 'Normal', 'One Half']
    durations = [1, 12, 18, 24, 36, 48]
    stretches = [-60, -40, -20, 1, 20, 40, 60]

    # Loop through the dunes and surges
    for dune in dunes:
        for surge in surges:

            # Set the experiment folder name
            experiment_name = f'{dune} {surge} Surge'
            experiment_folder = os.path.join(XB_DIR, experiment_name)

            # Make a target folder to move the runs into
            save_folder = os.path.join(DATA_DIR, 'XBeach Output', experiment_name)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            # Loop through the dunes and durations within the experiment
            for stretch in stretches:
                for duration in durations:

                    # Set the simulation folder
                    run_name = f'Dune Complexity {stretch} {duration}'
                    simulation_folder = os.path.join(experiment_folder, run_name)

                    # Set the XBeach output file as the source. Set the destination
                    # name. Then copy the file over
                    source = os.path.join(simulation_folder, 'xboutput.nc')
                    if os.path.exists(source):
                        destination = os.path.join(save_folder, f'{run_name}.nc')
                        if not os.path.exists(destination):
                            sh.copy(source, destination)
                            print(f'File Successfully Copied: {destination}')
                        else:
                            print(f'File already exists: {destination}')
                    else:
                        print(f'FILE DOES NOT EXIST: {source}')


def surge_time_series():
    """
    Put all the storm time series' into
    a .csv file that can be loaded as a
    DataFrame
    """

    # Set a list of storm surge modifiers
    # and storm duration increases
    surges, surge_labels = [0.5, 1.0, 1.5], ['Half', 'Normal', 'One Half']
    durations = [1, 12, 18, 24, 36, 48]

    # Make an empty DataFrame to loop into
    surge_df = pd.DataFrame()

    # Loop through the surges
    for surge, label in zip(surges, surge_labels):

        # Loop through the durations
        for duration in durations:

            # The DataFrame won't work if the columns are different
            # lengths so place them all in a preset 125 "hour" long
            # array so that they'll fit in the DataFrame
            time_series = np.full((1, 125), fill_value=np.nan)[0]

            # Load the data and place it in the time series NaN array
            filename = os.path.join(XB_DIR, f'Toes Joined {label} Surge', f'Dune Complexity 1 {duration}', 'ntr.txt')
            ntr = np.genfromtxt(filename, dtype=np.float32)
            time_series[:len(ntr)] = ntr

            # Place the time series in the dict
            surge_df[f'{label} {duration}'] = time_series

    # Save the DataFrame as a .csv file
    save_name = os.path.join(DATA_DIR, 'Storm Surge Time Series.csv')
    surge_df.to_csv(save_name, index=False)


def main():
    """
    Main program function to consolidate all the
    data sources
    """

    # Make a .csv file with the initial profiles used
    # initial_profiles()

    # Make a .csv file with the initial dune ratios
    # initial_ratios()

    # Make a .csv file with all the natural dune volumes
    # and aspect ratios measured from Bogue Banks LiDAR
    # bogue_lidar_data()

    # Make a .csv file with the storm surge time
    # series' for all the model runs
    # surge_time_series()

    # Make a .csv file with storm surge data
    # for Tropical Storm Joaquin and Hurricane Florence
    # joaquin_and_florence()

    # Move the netCDF output files into the Data folder
    # and rename them for the run name. Move the .csv
    # files with the morphometrics from the runs too
    # move_csv_output()
    # move_netcdf_output()

    # Move the Bogue Banks field data morphometrics
    # from 2017 and 2018 into the data folder
    move_field_data()


if __name__ == '__main__':
    main()
