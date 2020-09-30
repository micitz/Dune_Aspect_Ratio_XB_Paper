"""
This function makes the Figures associated with the paper
for Dune Aspect Ratio.

Figures 1 and 6 were made in Illustrator!

Michael Itzkin, 5/6/2020
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime as dt
import pandas as pd
import numpy as np
import os

# Set general variables
stretches = [-60, -40, -20, 1, 20, 40, 60]
stretch_labels = ['1.6', '1.4', '1.2', '1.0', '0.8', '0.6', '0.4']
experiments = ['Toes Aligned', 'Crests Aligned', 'Heels Aligned', 'Fenced']

# Setup the figure style
font = {
    'fontname': 'Arial',
    'fontweight': 'normal',
    'fontsize': 14}
figure_dpi = 300
figure_cm = 8
figure_inches = figure_cm * 0.393701
figure_size = (figure_inches, figure_inches)
line_width = 0.75
edge_color = 'black'
stretch_cmap = plt.cm.cividis_r(np.linspace(0, 1, len(stretches)))
duration_cmap = plt.cm.viridis_r(np.linspace(0, 1, len([1, 12, 18, 24, 36, 48])))

# Set paths
FIGURE_DIR = os.path.join('..', 'Figures')
DATA_DIR = os.path.join('..', 'Data')


"""
Functions for Machine Learning
"""


def run_grid_search_tree(X, y, cv=5):
    """
    Run a grid search with cross-validation
    and print out the results. This is for
    a Decision Tree regression for dune volume
    change based on initial beach width and
    aspect ratio. Return the best model
    """

    # Set a dictionary of parameters to check
    params = {
        'clf__max_depth': range(1, 33),
        'clf__min_samples_split': np.arange(start=0.1, stop=0.6, step=0.1),
        'clf__min_samples_leaf': np.arange(start=0.1, stop=0.6, step=0.1),
    }

    # Scale the data
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', RandomForestRegressor(random_state=0))])

    # Setup the grid search
    clf = GridSearchCV(estimator=pipe,
                       param_grid=params,
                       n_jobs=-1,
                       refit=True,
                       cv=cv,
                       verbose=True)

    # Run the grid search
    clf.fit(X, y)

    return clf


def run_grid_search_svr(X, y, cv=5):
    """
    Run a grid search with cross-validation
    and print out the results. This is for
    a support vector regression for dune volume
    change based on initial beach width and
    aspect ratio. Return the best model
    """

    # Set a dictionary of parameters to check
    params = {
        'clf__C': np.logspace(-6, 3, 10),
        'clf__epsilon': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'clf__degree': [1, 2, 3, 4]
    }

    # Scale the data
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', SVR(kernel='linear', C=1))])

    # Setup the grid search
    clf = GridSearchCV(estimator=pipe,
                       param_grid=params,
                       n_jobs=-1,
                       refit=True,
                       cv=cv,
                       verbose=True)

    # Run the grid search
    clf.fit(X, y)

    return clf


"""
Functions to load and format data
"""


def load_field_data(year):
    """
    Load morphometrics measured from field data
    for Bogue Banks for the given year
    """

    # Set a path to the data
    fname = os.path.join(DATA_DIR, f'Morphometrics for Bogue Banks {year}.csv')

    # Load the data into a dataframe
    df = pd.read_csv(fname, delimiter=',', header=0)

    # Add a column for the dune shape
    df['Ratio'] = (df['yCrest'] - df['yToe']) / (df['xToe'] - df['xHeel'])

    return df


def load_volume_loss(experiment, backup=False):
    """
    Calculate the volume loss for XBeach runs as a magnitude and proportion
    """

    # Set the stretches
    dune_stretches = [-60, -40, -20, 1, 20, 40, 60]
    storm_stretches = [1, 12, 18, 24, 36, 48]

    # Set empty lists
    d_volume, d_volume_proportion, beach_width, beach_slope, o_volume = [], [], [], [], []

    # Set an empty dict
    volume_time = dict()

    # Loop through the stretches
    for dune in dune_stretches:
        for weather in storm_stretches:
            # Load the data
            data_fname = os.path.join(DATA_DIR,
                                      'XBeach Morphometrics',
                                      experiment,
                                      f'Dune Complexity {dune} {weather} Morphometrics.csv')
            data = pd.read_csv(data_fname)

            # Pull out the overwash volume column before removing NaNs
            overwash = data['Overwash Volume'].dropna()

            # Remove NaNs
            data = pd.read_csv(data_fname).dropna()

            # Calculate the change in volume
            d_volume.append(data['Dune Volume'].iloc[-1] - data['Dune Volume'].iloc[0])

            # Calculate the percent change in volume
            vol_proportion = data['Dune Volume'].iloc[-1] / data['Dune Volume'].iloc[0]
            d_volume_proportion.append(vol_proportion)

            # Get the initial beach width and slope
            beach_width.append(data['xToe'].iloc[0] - data['xMhw'].iloc[0])
            beach_slope.append(data['Beach Slope'].iloc[0])

            # Calculate the change in overwash volume
            delta_overwash = overwash.iloc[-1] - overwash[0]
            o_volume.append(delta_overwash)

            # Add all of the volumes to the volume_time dict
            volume_time[str(dune) + ' ' + str(weather)] = data['Dune Volume']

    # Convert all to arrays
    d_volume = np.asarray(d_volume, dtype=np.float64)
    d_volume_proportion = np.asarray(d_volume_proportion, dtype=np.float64)
    beach_width = np.asarray(beach_width, dtype=np.float64)
    beach_slope = np.asarray(beach_slope, dtype=np.float64)
    o_volume = np.asarray(o_volume, dtype=np.float64)

    # Convert the volume_time dict to a dataframe
    volume_time = pd.DataFrame.from_dict(volume_time)

    return d_volume, d_volume_proportion, volume_time, beach_width, beach_slope, o_volume


def load_plot_data(experiment, backup=False):
    """
    Load the initial height, dune ratio, and stretches for the current experiment
    """

    # Set the stretches
    dune_stretches = [-60, -40, -20, 1, 20, 40, 60]
    storm_stretches = [1, 12, 18, 24, 36, 48]

    # Set empty lists
    dune_ratio, init_height, use_stretches, final_dune_ratio = [], [], [], []

    # Set an empty dict
    volume_time = dict()

    # Loop through the stretches
    for dune in dune_stretches:
        for weather in storm_stretches:

            # Load the data
            data_fname = os.path.join(DATA_DIR,
                                      'XBeach Morphometrics',
                                      experiment,
                                      f'Dune Complexity {dune} {weather} Morphometrics.csv')
            data = pd.read_csv(data_fname)

            # Identify the initial dune ratio and dune height
            dune_ratio.append(data['Dune Ratio'].iloc[0])
            start_height = data['yCrest'].iloc[0]

            # Drop NAs from the dataframe
            data = data.dropna()

            # Identify the final dune ratio
            final_dune_ratio.append(data['Dune Ratio'].iloc[-1])

            # Identify the initial dune height
            init_height.append(start_height - data['yToe'].iloc[0])

            # Add the stretch being used
            use_stretches.append(weather)

    # Convert all to arrays
    dune_ratio = np.asarray(dune_ratio, dtype=np.float64)
    final_dune_ratio = np.asarray(final_dune_ratio, dtype=np.float64)
    init_height = np.asarray(init_height, dtype=np.float64)
    use_stretches = np.asarray(use_stretches, dtype=np.int64)

    return init_height, dune_ratio, use_stretches, final_dune_ratio


"""
Functions for common figure making methods
"""


def add_grids(axs):
    """
    Add a grid to the background of all the
    axes in the figure

    axs: axis object or list of axes objects
    """

    if isinstance(axs, list):
        [ax.grid(color='lightgrey', linewidth=0.25, zorder=0) for ax in axs]
    else:
        axs.grid(color='lightgrey', linewidth=0.25, zorder=0)


def add_label(ax, s, type):
    """
    Add an axis or title label

    ax: Axis object to apply the label to
    s: String object with the label
    type: String object to apply either a (X), (Y), or (T)itle
    """

    if type.upper() == 'X':
        ax.set_xlabel(s, **font)
    elif type.upper() == 'Y':
        ax.set_ylabel(s, **font)
    elif type.upper() == 'T':
        ax.set_title(s, **font)


def axis_limits(ax, l=None, r=None, b=None, t=None):
    """
    Set the axis limits

    ax: Axis object to set the limits on
    l: Lower X-axis limit
    r: Upper X-axis limit
    b: Lower Y-axis limit
    t: Upper Y-axis limit
    """

    # Set the x-axis
    ax.set_xlim(left=l, right=r)

    # Set the y-axis
    ax.set_ylim(bottom=b, top=t)


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def delta_volume_loss_phases(ax, x, y, z, overwash=False):
    """
    Plot the phase diagrams for the data based on differences in volume loss. Add contour
    lines with labels

    Set overwash to True to modify the color range for overwash measurements
    """

    # Set the levels
    if overwash:
        clevels = np.linspace(0, 50, 100)
        contours = [0, 10, 20, 30, 40, 50]
        vlimit = max(contours)
    else:
        clevels = np.linspace(-50, 50, 100)
        contours = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
        vlimit = max(contours)

    # Use a Pink-Green diverging colormap for volume change. The map
    # will be centered on 0 with pink values indicating the natural dune
    # experienced more overwash than the natural-fenced dune and green
    # values indicating the natural-fenced dune experienced more overwash
    # than the natural dune
    cmap = plt.cm.get_cmap('PiYG_r')

    # Plot a filled contour of the data
    plot = ax.tricontourf(x, y, z,
                          levels=clevels,
                          cmap=cmap,
                          vmin=-vlimit,
                          vmax=vlimit)

    # Add contour lines for every 10m3/m eroded and add labels
    cs = ax.tricontour(x, y, z,
                       linewidths=0.5,
                       colors='black',
                       levels=contours)
    ax.clabel(cs, inline=1, fontsize=5, fmt='%1.f')

    # Return plot information to add the colorbar
    return plot


def delta_volume_loss_colorbar(fig, plot, ax=None, overwash=False):
    """
    Add a colorbar for the change in volume loss 

    Set overwash to True to use this for overwash volume changes
    """""

    if overwash:
        label = 'Volume$_{OW, Diff}$ (m$^{3}$/m)'
        contours = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    else:
        label = '$\Delta Volume_{Dune} (Fenced - Natural, m^{3}/m)$'
        contours = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]

    if ax:
        cbar = fig.colorbar(plot, orientation='vertical', ax=ax)
        cbar.set_label('Volume Loss\n(Natural - Fenced, m$^{3}$/m)', **font)
    else:
        fig.subplots_adjust(right=0.87)
        cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.78])
        cbar = fig.colorbar(plot, orientation='vertical', cax=cbar_ax)
        cbar.set_label(label, **font)

    cbar.set_ticks(contours)

    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontname(font['fontname'])
        l.set_fontsize(font['fontsize'])


def save_and_close(fig, title, tight=True):
    """
    Set a tight and transparent background for the
    figure, then save and close it

    fig: Figure object
    title: Str with the title of the figure
    tight: Make a tight figure (default = True)
    """

    # Set a tight layout and a transparent background
    if tight:
        plt.tight_layout()
        fig.patch.set_color('w')
        fig.patch.set_alpha(0.0)

    # Save and close the figure
    title_w_extension = os.path.join(FIGURE_DIR, f'{title}.png')
    plt.savefig(title_w_extension, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi='figure')
    print('Figure saved: %s' % title_w_extension)

    # Setup a .pdf version for publishing and save
    publish_version = f'{title}.pdf'
    publish_save_name = os.path.join(FIGURE_DIR, 'Paper Figures', publish_version)
    plt.savefig(publish_save_name, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    print('Figure saved: %s' % publish_version)
    plt.close()


def time_ticks(df, nticks):
    """
    Make ticks with timestamps from
    a DataFrame columns of times

    df: DataFrame with a 'Time' column
    nticks: Int with the number of tick marks
    """

    time = df['Time']
    start = pd.Timestamp(time.iloc[0]) + dt.timedelta(hours=12)
    end = pd.Timestamp(time.iloc[-1]) + dt.timedelta(hours=12)
    t = np.linspace(start.value, end.value, 7)
    t = pd.to_datetime(t)
    return [tval.strftime('%Y-%m-%d') for tval in t]


def volume_loss_phases(ax, x, y, z):
    """
    Plot the phase diagrams for the data based on volume loss. Add contour
    lines with labels
    """

    # Use the "afmhot" colormap for the plots. Darker colors will
    # signify more volume loss
    cmap = plt.cm.get_cmap('afmhot')

    # Plot a filled contour of the data
    plot = ax.tricontourf(x, y, z,
                          levels=np.linspace(-70, 10, 40),
                          cmap=cmap,
                          vmin=-70,
                          vmax=10)

    # Add contour lines for every 10m3/m eroded and add labels. Add
    # an additional contour line at 53m3/m to show where total dune
    # erosion occurred
    cs = ax.tricontour(x, y, z,
                       linewidths=0.5,
                       colors='black',
                       levels=[-70, -60, -53, -50, -40, -30, -20, -10, 0, 10])
    ax.clabel(cs, inline=1, fontsize=5, fmt='%1.f')

    # Highlight the contour line at -53m3/m siginifying where dunes
    # were completely eroded
    ax.tricontour(x, y, z,
                  linewidths=1.5,
                  linestyles='solid',
                  colors='cyan',
                  levels=[-53])

    # Return plot information to add the colorbar
    return plot


def volume_loss_colorbar(fig, plot, ax=None):
    """
    Add a colorbar for the volume loss 
    """""

    if ax:
        cbar = fig.colorbar(plot, orientation='vertical', ax=ax)
        cbar.set_label('Volume Loss\n(m$^{3}$/m)', **font)
    else:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.99, 0.14, 0.02, 0.78])
        cbar = fig.colorbar(plot, orientation='vertical', cax=cbar_ax)
        cbar.set_label('$\Delta Volume_{Dune}$ (m$^{3}$/m)', **font)

    cbar.set_ticks([-70, -60, -50, -40, -30, -20, -10, 0, 10])

    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontname(font['fontname'])
        l.set_fontsize(font['fontsize'])


"""
Functions to make figures
"""


def figure_2():
    """
    Figure 2: Synthetic dune profiles; (A) toes aligned, (B) crests aligned, (C) heels aligned,
    and (D) fenced. The proportional change in aspect ratio increase/decrease relative to the
    reference profile (1.0x) is shown in panel A (aspect ratio in parentheses).
    """

    # Load the data
    profiles = pd.read_csv(os.path.join(DATA_DIR, 'Initial Profiles.csv'))
    ratios = pd.read_csv(os.path.join(DATA_DIR, 'Initial Dune Ratios.csv'))

    # Setup the figure and axes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                                                 dpi=figure_dpi, figsize=(figure_inches * 2, figure_inches * 2))
    axes = [ax1, ax2, ax3, ax4]
    letters = ['(a)', '(b)', '(c)', '(d)']
    xtext, ytext, spacing = -147, 7.25, 0.75
    color_darken = 0.75

    # Add grids
    add_grids(axes)

    # Loop through the axes and profiles and plot common things
    for ax, experiment, letter in zip(axes, experiments, letters):

        # Pull out the portion of the data to plot for
        # the current experiment
        curr_profiles = profiles.loc[profiles['Experiment'] == experiment]
        curr_ratios = ratios.loc[ratios['Experiment'] == experiment]

        # Loop through the stretches
        for ii, stretch in enumerate(stretches):

            # Plot the profile
            ax.plot(curr_profiles['X'], curr_profiles[f'{stretch} pct'],
                    color=adjust_lightness(stretch_cmap[ii], amount=color_darken),
                    zorder=ii + 1)

            # Add a label showing the aspect ratio to the top left subplot
            if ax == ax3:
                ax.text(x=xtext,
                        y=ytext - (ii * spacing),
                        s=f'{stretch_labels[ii]}X ({np.around(curr_ratios["Ratio"].iloc[ii], decimals=2)})',
                        color=adjust_lightness(stretch_cmap[ii], amount=color_darken),
                        zorder=ii + 1,
                        **font)

        # Add a subplot letter label
        ax.text(s=f'{letter}.', x=-15, y=7.25,
                zorder=4,
                **font)

        # Set the axis limits
        axis_limits(ax=ax, l=-150, r=0, b=0, t=8)

        # Set the axis title
        add_label(ax, s=experiment, type='T')

    # Set x-axis labels
    for ax in [ax3, ax4]:
        add_label(ax, s='Cross-Shore Distance (m)', type='X')

    # Set y-axis labels
    for ax in [ax1, ax3]:
        add_label(ax, s='Elevation (m)', type='Y')

    # Save and close the figure
    title = 'Figure 2'
    save_and_close(fig=fig, title=title)


def figure_3():
    """
    Figure 3: Dune volume versus aspect ratio measured from LiDAR profiles from Bogue Banks, NC collected between
    1997-2016. The red box represents the range of conditions simulated in this study.
    """

    # Load the data
    df = pd.read_csv(os.path.join(DATA_DIR, 'Bogue Banks Volumes and Aspect Ratios.csv'))

    # Setup the figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=figure_dpi, figsize=(figure_inches, figure_inches))

    # Add a grid
    add_grids(axs=ax)

    # Plot the data
    ax.scatter(x=df['Ratio'],
               y=df['Volume'],
               c='none',
               linewidth=0.25,
               edgecolor='black',
               zorder=2)

    # Add a box around the ratios and volumes used in the simulations
    box = patches.Rectangle(xy=(0.02, 50.6),
                            width=0.25,
                            height=2.9,
                            linewidth=1,
                            edgecolor='red',
                            facecolor='none',
                            zorder=4)
    ax.add_patch(box)

    # Set the axis limits
    axis_limits(ax=ax, l=0, r=1, b=0, t=350)

    # Set the axis labels
    add_label(ax=ax, s='Aspect Ratio (-)', type='X')
    add_label(ax=ax, s='Volume (m$^{3}$/m)', type='Y')

    # Save and close the figure
    save_and_close(fig=fig, title='Figure 3')


def figure_4():
    """
    Figure 4: Synthetic storm surge time series used in this study. Colors
    refer to the storm duration increase. Dashed lines represent the 0.5x surge,
    solid lines represent the 1.0x surge, and dotted lines represent the 1.5x surge.
    """

    # Load the surge data
    filename = os.path.join(DATA_DIR, 'Storm Surge Time Series.csv')
    surge_df = pd.read_csv(filename)

    # Setup the figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figure_inches, figure_inches), dpi=figure_dpi)
    lines = ['--', '-', ':']
    stretches = [1, 12, 18, 24, 36, 48]
    labels = ['Base', '12hr', '18hr', '24hr', '36hr', '48hr']
    surges = ['Half', 'Normal', 'One Half']

    # Add a grid
    add_grids(ax)

    # Plot the storm surges
    for ii, stretch in enumerate(stretches):
        label_str = labels[ii]

        # Loop through the surge modifiers and plot
        for surge, label, line in zip(surges, labels, lines):
            ax.plot(surge_df[f'{surge} {stretch}'],
                    color=duration_cmap[ii],
                    linewidth=line_width,
                    linestyle=line,
                    zorder=2 + ii)

        # Add a text label for the storm duration
        y_loc = 1.35 - (0.14 * ii)
        ax.text(x=95, y=y_loc,
                s=label_str,
                color=duration_cmap[ii],
                ha='left',
                zorder=3 + ii,
                **font)

    # Add a zero-line
    ax.axhline(y=0,
               color='black',
               linewidth=1,
               linestyle='--',
               zorder=20)

    # Set the axis labels
    add_label(ax=ax, s='Time (Hours)', type='X')
    add_label(ax=ax, s='Surge (m)', type='Y')

    # Set the axis limits
    axis_limits(ax=ax, l=0, r=125, b=-0.5, t=1.5)
    ax.set_xticks([0, 25, 50, 75, 100, 125])
    ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])

    save_and_close(fig=fig, title='Figure 4')


def figure_5():
    """
    Figure 5: Hydrographs for Hurricane Florence (top) and Tropical Storm
    Joaquin (bottom) showing observed water levels (blue), predicted water
    levels (red), and the non-tidal residual (NTR, black). The timing of
    peak surge for each storm is highlighted by the vertical dashed line.
    """

    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=figure_dpi, figsize=(figure_inches * 2, figure_inches*2))
    columns = ['Observed', 'Predicted', 'NTR']
    colors = ['blue', 'red', 'black']
    storms = ['Florence', 'Joaquin']

    # Add a grid
    add_grids([ax1, ax2])

    # Loop through the axes, plot Florence on top and Joaquin on the bottom
    for ax, storm in zip([ax1, ax2], storms):

        # Load the data
        filename = os.path.join(DATA_DIR, f'{storm}.csv')
        data = pd.read_csv(filename)

        # Make a vector of datetime objects for the x-axis
        t = time_ticks(df=data, nticks=7)
        xticks = np.linspace(0, len(data['Observed']), num=7)

        # Plot the zero line
        ax.axhline(y=0, xmin=0, xmax=73, color='black', linewidth=0.75, linestyle='--', zorder=1)

        # Plot a line at the timing of peak surge
        ax.axvline(x=data['NTR'].idxmax(), ymin=-0.5, ymax=1.50, color='gray', linewidth=0.75, linestyle='--', zorder=1)

        # Plot the data
        for ii, (column, col) in enumerate(zip(columns, colors)):
            ax.plot(data[column], color=col, linewidth=1.50, zorder=ii + 2)

        # Add text labels to the top plot
        if ax == ax1:
            xtext, ytext, spacing, counter = 95, 1.65, 0.35, 0
            for ii, (label, col) in enumerate(zip(columns, colors)):
                ax.text(x=xtext, y=ytext - (spacing * ii), s=label, color=col, ha='right', zorder=5, **font)

        # Add text for the time and date of peak surge
        peak_surge_time = data['String Times'].iloc[data['NTR'].idxmax()] + ':00'
        ax.text(x=data['NTR'].idxmax() + 2, y=1.60, s=peak_surge_time, color='gray', zorder=5, **font)

        # Label the storm name in the top left corner of each plot
        ax.text(x=2.5, y=1.55, s=storm, ha='left', zorder=5, **font)

        # Set the axes limits
        axis_limits(ax=ax, l=0, r=len(data[column]), b=-1, t=2)

        # Label the y-axes
        add_label(ax=ax, s='Water Level (m)', type='Y')

        # Set the x-axis
        ax.set_xticks(xticks)
        ax.set_xticklabels(t, rotation=45, ha='right')

    # Save the figure
    save_and_close(fig=fig, title='Figure 5')


def figure_7(titles=True):
    """
    Figure 7: Dune aspect ratio versus storm duration for all simulations.
    Color depicts change in dune volume. Each row of plots represents a
    different storm intensity (increasing from top to bottom) and each column
    of plots represents a different dune configuration. Highlighted regions
    (cyan line) indicate simulations where the (natural) dune was inundated
    (low aspect ratio) or eroded through laterally (high aspect ratio).
    """

    # List of experiments in the order they will be plotted in
    experiments = ['Toes Joined Half Surge', 'Crests Joined Half Surge', 'Heels Joined Half Surge', 'Fenced Half Surge',
                   'Toes Joined Normal Surge', 'Crests Joined Normal Surge', 'Heels Joined Normal Surge', 'Fenced Normal Surge',
                   'Toes Joined One Half Surge', 'Crests Joined One Half Surge', 'Heels Joined One Half Surge', 'Fenced One Half Surge']

    # Setup the figure
    fig, ((ax1, ax2, ax3, ax4),
          (ax5, ax6, ax7, ax8),
          (ax9, ax10, ax11, ax12)) = plt.subplots(nrows=3, ncols=4,
                                                  sharex='all', sharey='all',
                                                  figsize=(figure_inches * 2, figure_inches * 1.5),
                                                  dpi=figure_dpi)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

    # Loop through the experiments and axes and plot
    for ax, experiment in zip(axes, experiments):

        # Load the specific data for the current experiment
        height, dune_ratio, use_stretches, _ = load_plot_data(experiment)
        d_volume, _, _, _, _, _ = load_volume_loss(experiment)

        # Plot the phase diagrams
        plot = volume_loss_phases(ax=ax, x=use_stretches, y=dune_ratio, z=d_volume)

    # Add a colorbar
    volume_loss_colorbar(fig=fig, plot=plot)

    # Set the x-axes
    for ax in [axes[8], axes[9], axes[10], axes[11]]:
        ax.set_xlim(left=0, right=50)
        ax.set_xlabel('+Duration (hr)', **font)

    # Set the y-axes
    for ax, modifier in zip([axes[0], axes[4], axes[8]], [0.5, 1.0, 1.5]):
        ax.set_ylim(bottom=0, top=0.3)
        ax.set_ylabel(r'$\bf{' + str(modifier) + 'x\ Surge}$\n' + 'Ratio (-)', **font)

    # Set the titles
    if titles:
        for ax, title in zip([axes[0], axes[1], axes[2], axes[3]],
                             ['Toes Aligned', 'Crests Aligned', 'Heels Aligned', 'Fenced']):
            ax.set_title(title, **font)

    # Save and close the figure
    save_and_close(fig=fig, title='Figure 7')


def figure_8():
    """
    Figure 8:  The difference in overwash volumes for the equivalent fenced
    and non-fenced simulations calculated by taking the fenced dune overwash
    volume and subtracting the natural dune overwash volume.Darker colors
    indicate a greater overwash volume for the non-fenced simulation.
    """

    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey='all',
                                   figsize=(figure_inches, figure_inches),
                                   dpi=figure_dpi)
    axes = [ax1, ax2]
    natural = ['Toes Joined Normal Surge', 'Toes Joined One Half Surge']
    fences = ['Fenced Normal Surge', 'Fenced One Half Surge']

    # Loop through the experiments and axes and plot
    for ax, natural_experiment, fenced_experiment in zip(axes, natural, fences):

        # Load the natural dune dara
        height, dune_ratio, use_stretches, _ = load_plot_data(natural_experiment, backup=False)
        _, _, _, _, _, o_volume = load_volume_loss(natural_experiment, backup=False)

        # Load the fenced dune data
        _, _, _, _, _, o_volume_fenced = load_volume_loss(fenced_experiment, backup=False)

        # Calculate the difference in volume loss (natural - fenced)
        volume_difference = np.abs(o_volume_fenced - o_volume)

        # Plot the phase diagrams
        plot = delta_volume_loss_phases(ax=ax, x=use_stretches, y=dune_ratio, z=volume_difference, overwash=True)

    # Add a colorbar
    delta_volume_loss_colorbar(fig=fig, plot=plot, overwash=True)

    # Set the x-axes
    for ax in axes:
        ax.set_xlim(left=0, right=50)
        ax.set_xlabel('+Duration (hr)', **font)

    # Set the y-axes
    ax1.set_ylim(bottom=0, top=0.3)
    ax1.set_ylabel('Aspect Ratio (-)\n', **font)

    # Set the titles
    for ax, title in zip(axes, ['1.0x Surge', '1.5x Surge']):
        ax.set_title(title, **font)

    # Set common axis properties
    for ax in axes:

        # Set the ticks
        ax.tick_params(colors='black')
        ax.set_xticks([0, 25, 50])
        ax.set_yticks([0, 0.1, 0.2, 0.3])

    # Save the figure and close the text file
    save_and_close(fig=fig, title='Figure 8', tight=False)


def figure_9():
    """
    Figure 9: Beach width versus storm duration for natural dune simulations
    colored by the change in dune volume (red: erosion, blue: accretion), similar
    to Figure 7 but with the y-axis re-scaled based on the initial beach width in
    each simulation. Each row of plots represents a different storm intensity
    (increasing from top to bottom) and each column of plots represents a different
    dune configuration. Because simulations for profiles having a fenced dune are
    synthesized using the toes-aligned configuration and therefore with a narrower
    beach with the toes-aligned and simulations with a fenced dune do not have any
    variations in the beach width, they are not included in this analysis.
    Highlighted regions (cyan line) indicate where the dune was inundated.
    """

    # List of experiments in the order they will be plotted in
    experiments = ['Crests Joined Half Surge', 'Heels Joined Half Surge',
                   'Crests Joined Normal Surge', 'Heels Joined Normal Surge',
                   'Crests Joined One Half Surge', 'Heels Joined One Half Surge']

    # Setup the figure
    fig, ((ax1, ax2),
          (ax3, ax4),
          (ax5, ax6)) = plt.subplots(nrows=3, ncols=2,
                                     sharex='all', sharey='all',
                                     figsize=(figure_inches * 2, figure_inches * 1.5),
                                     dpi=figure_dpi)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # Loop through the experiments and axes and plot
    for ax, experiment in zip(axes, experiments):

        # Load the specific data for the current experiment
        _, dune_ratio, use_stretches, _ = load_plot_data(experiment)
        d_volume, _, _, beach_width, _, _ = load_volume_loss(experiment)

        # Plot the phase diagrams
        plot = volume_loss_phases(ax=ax, x=use_stretches, y=beach_width, z=d_volume)

    # Add a colorbar
    volume_loss_colorbar(fig=fig, plot=plot)

    # Set the x-axes
    for ax in [axes[-2], axes[-1]]:
        ax.set_xlim(left=0, right=50)
        ax.set_xticks([0, 25, 50])
        ax.set_xlabel('+Duration (hr)', **font)

    # Set the y-axes
    for ax, modifier in zip([axes[0], axes[2], axes[4]], [0.5, 1.0, 1.5]):
        ax.set_ylim(bottom=0, top=60)
        ax.set_ylabel(r'$\bf{' + str(modifier) + 'x\ Surge}$\n' + 'Width$_{Beach}$ (m)', **font)

    # Set the titles
    for ax, title in zip([axes[0], axes[1]], ['Crests Aligned', 'Heels Aligned']):
        ax.set_title(title, **font)

    # Save and close the figure
    save_and_close(fig=fig, title='Figure 9')


def figure_10():
    """
    Plot volume loss versus dune shape for Bogue Banks
    field profiles measured in October 2017 and
    October 2018
    """

    # Load field data from 2017 and 2018
    data_2017, data_2018 = load_field_data(2017), load_field_data(2018)

    # Setup the figure
    fig, ax = plt.subplots(figsize=(figure_inches * 1.5, figure_inches), dpi=figure_dpi)

    # Add a grid
    add_grids(axs=ax)

    # Plot the data
    volume_loss = data_2018['2016 Dune Volume'] - data_2017['2016 Dune Volume']
    plot = ax.scatter(x=data_2017['Beach Width'],
                      y=data_2017['Ratio'],
                      c=volume_loss,
                      cmap=plt.cm.seismic_r,
                      edgecolor='black',
                      linewidth=0.5,
                      vmin=-40,
                      vmax=40,
                      zorder=4)

    # Add a colorbar
    cbar = fig.colorbar(plot, orientation='vertical')
    cbar.set_label('$\Delta$Volume (m$^{3}$/m)', **font)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontname(font['fontname'])
        l.set_fontsize(font['fontsize'])

    # Set the axis limits
    axis_limits(ax=ax, l=0, r=100, b=0, t=0.3)

    # Set the axis labels
    add_label(ax=ax, s='Beach Width (m)', type='X')
    add_label(ax=ax, s='Aspect Ratio (-)', type='Y')

    # Set a tight layout and a transparent background
    save_and_close(fig=fig, title='Figure 10')


def supp_figure_1():
    """
    Plot the CDF of dune volumes on Bogue Banks
    measured from LiDAR and mark off where 52m3/m
    falls

    CDF help:
    https://stackoverflow.com/a/37254481
    """

    # Load the data
    df = pd.read_csv(os.path.join(DATA_DIR, 'Bogue Banks Volumes and Aspect Ratios.csv'))

    # Set bins edges
    data_set = sorted(set(df['Volume'].dropna()))
    bins = np.append(data_set, data_set[-1] + 1)

    # Use the histogram function to bin the data and find the CDF
    counts, bin_edges = np.histogram(df['Volume'].dropna(), bins=bins, normed=True, density=False)
    counts = counts.astype(float) / len(df['Volume'])
    cdf = np.cumsum(counts)

    # Find the percentile for a volume of 52m3/m
    use_vol = (53.5 + 50.6) / 2
    vol_diff = np.abs(bin_edges[1:] - use_vol)
    min_ix = np.argmin(vol_diff)
    use_percentile = cdf[min_ix]

    # Setup the figure
    fig, ax = plt.subplots(figsize=(figure_inches, figure_inches), dpi=figure_dpi)

    # Add a grid
    add_grids(ax)

    # Plot the CDF
    ax.plot(bin_edges[1:], cdf, zorder=2)

    # Add a line for the model volume
    ax.axvline(x=use_vol, ymin=0, ymax=use_percentile, color='black', linestyle='--', zorder=4)
    ax.axhline(y=use_percentile, xmin=0, xmax=(use_vol / 400), color='black', linestyle='--', zorder=4)
    ax.scatter(x=use_vol, y=use_percentile, color='red', marker='o', s=25, zorder=6)

    # Set axes limits
    axis_limits(ax, l=0, r=400, b=0, t=1)

    # Label the axes
    add_label(ax, s='Dune Volume (m$^{3}$/m)', type='X')
    add_label(ax, s='CDF', type='Y')
    add_label(ax, s=f'{use_vol} m3/m\n{np.around(use_percentile * 100, decimals=2)}th Percentile', type='T')

    # Save and close
    save_and_close(fig=fig, title='Bogue Banks Dune Volume CDF', tight=True)


def supp_figure_2():
    """
    Perform a multiple linear regression on the field
    data to get a function:
    volume_change = (2017 beach width)X1 + (2017 Ratio)X2

    Save an image file with the results of the regression
    """

    # Load field data from 2017 and 2018
    data_2017, data_2018 = load_field_data(2017), load_field_data(2018)

    # Calculate the volume loss and put in a new DataFrame with the
    # dependent variables. Remove any NaNs for scikit-learn
    volume_loss = data_2018['2016 Dune Volume'] - data_2017['2016 Dune Volume']
    df = pd.DataFrame()
    df['Beach Width'] = data_2017['Beach Width']
    df['Ratio'] = data_2017['Ratio']
    df['Volume Loss'] = volume_loss
    df = df.dropna()

    # Make a DataFrame for the X and Y values
    ex = df[['Beach Width', 'Ratio']]
    why = df['Volume Loss']

    # Perform the regression with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(ex, why)

    # Perform the regression with Statsmodels
    ex = sm.add_constant(ex)  # adding a constant
    model = sm.OLS(why, ex).fit()
    print_model = model.summary()

    # Save the model summary as an image
    plt.subplots(dpi=figure_dpi, figsize=(12, 7))
    plt.text(0.01, 0.05,
             str(model.summary()),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'Field Data Multiple Linear Regressions.png'))
    plt.close()

    # Calculate a grid of values from the regression model
    X = np.linspace(start=0, stop=120, num=50)
    y = np.linspace(start=0, stop=0.3, num=50)
    xx, yy = np.meshgrid(X, y)
    z = regr.intercept_ + (regr.coef_[0] * xx) + (regr.coef_[1] * yy)

    # Setup the figure
    fig, ax = plt.subplots(dpi=300, figsize=(figure_inches * 1.5, figure_inches))

    # Add a filled in contour plot of the model results and
    # label the contour lines
    ax.contourf(xx, yy, z, levels=200, cmap=plt.cm.seismic_r, vmin=-40, vmax=40, zorder=0)
    cs = ax.contour(xx, yy, z,
                    linewidths=0.5,
                    levels=np.arange(start=-40, stop=40, step=5),
                    colors='black',
                    zorder=2)
    ax.clabel(cs, inline=1, fontsize=10, fmt='%1.f')

    # Plot the field data over the surface
    plot = ax.scatter(x=data_2017['Beach Width'],
                      y=data_2017['Ratio'],
                      c=volume_loss,
                      cmap=plt.cm.seismic_r,
                      edgecolor='black',
                      linewidth=1,
                      vmin=-40,
                      vmax=40,
                      zorder=4)

    # Add a colorbar
    cbar = fig.colorbar(plot, orientation='vertical')
    cbar.set_label('$\Delta$Volume (m$^{3}$/m)', **font)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontname(font['fontname'])
        l.set_fontsize(font['fontsize'])

    # Label the axes
    add_label(ax=ax, s='Beach Width (m)', type='X')
    add_label(ax=ax, s='Aspect Ratio (-)', type='Y')

    # Make the figure tight and transparent,
    # then save it
    plt.show()
    # save_and_close(fig=fig, title='Field Data Multiple Regressions Surface', tight=True)


def supp_figure_2_SVR(cv=5, plot=True):
    """
    Perform a support vector regression on the field
    data to get a function:
    volume_change = (2017 beach width)X1 + (2017 Ratio)X2

    Save an image file with the results of the regression

    cv: Number of folds to pass to the GridSearchCV
    """

    # Load field data from 2017 and 2018
    data_2017, data_2018 = load_field_data(2017), load_field_data(2018)

    # Calculate the volume loss and put in a new DataFrame with the
    # dependent variables. Remove any NaNs for scikit-learn
    volume_loss = data_2018['2016 Dune Volume'] - data_2017['2016 Dune Volume']
    df = pd.DataFrame()
    df['Beach Width'] = data_2017['Beach Width']
    df['Ratio'] = data_2017['Ratio']
    df['Volume Loss'] = volume_loss
    df = df.dropna()

    # Make a DataFrame for the X and Y values
    ex = df[['Beach Width', 'Ratio']]
    why = df['Volume Loss']

    # Perform the regression with sklearn
    clf = run_grid_search_tree(ex, why, cv=cv)

    # Calculate a grid of values from the regression model
    X = np.linspace(start=0, stop=120, num=50)
    y = np.linspace(start=0, stop=0.3, num=50)
    xx, yy = np.meshgrid(X, y)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # Setup the figure
    fig, ax = plt.subplots(dpi=300, figsize=(figure_inches * 1.5, figure_inches))

    # Add a filled in contour plot of the model results and
    # label the contour lines
    ax.contourf(xx, yy, z,
                levels=200,
                cmap=plt.cm.seismic_r,
                vmin=-40,
                vmax=40,
                zorder=0)
    cs = ax.contour(xx, yy, z,
                    levels=np.arange(start=-40, stop=40, step=5),
                    linewidths=0.5,
                    colors='black',
                    zorder=2)
    ax.clabel(cs, inline=1, fontsize=10, fmt='%1.f')

    # Plot the field data over the surface
    plot = ax.scatter(x=data_2017['Beach Width'],
                      y=data_2017['Ratio'],
                      c=volume_loss,
                      cmap=plt.cm.seismic_r,
                      edgecolor='black',
                      linewidth=1,
                      vmin=-40,
                      vmax=40,
                      zorder=4)

    # Add a colorbar
    cbar = fig.colorbar(plot, orientation='vertical')
    cbar.set_label('$\Delta$Volume (m$^{3}$/m)', **font)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontname(font['fontname'])
        l.set_fontsize(font['fontsize'])

    # Label the axes
    add_label(ax=ax, s='Beach Width (m)', type='X')
    add_label(ax=ax, s='Aspect Ratio (-)', type='Y')

    plt.show()


"""
Main program
"""


def main():
    """
    Main program function to make the figures
    """

    # Make Figure 2. This shows all the initial profiles for all the
    # simulations laid out in subplots based on the different configurations
    figure_2()

    # Make Figure 3. This shows the natural dune volume versus natural dune
    # aspect ratio for Bogue Banks lidar data with a box highlighting the
    # parameter space of the XBeach simulations
    # figure_3()

    # Make Figure 4. This shows the storm surge component of each simulated
    # storm's time series colored by storm duration and with the linestyle
    # based on the surge modifier
    # figure_4()

    # Make Figure 5. This shows the storm surge time series for
    # Tropical Storm Joaquin and Hurricane Florence
    # figure_5()

    # Make Figure 7. This figures shows the volume loss for all
    # simulations arranged as a 3x4 grid of phase diagrams colored
    # by volume loss with a special contour to delineate where the
    # dune was completely eroded
    # figure_7(titles=True)

    # Make Figure 8. This figure shows the overwash volume differences
    # as a phase diagram
    # figure_8()

    # Make Figure 9. This figures shows the volume loss for all
    # simulations arranged as a 3x4 grid of phase diagrams colored
    # by volume loss with a special contour to delineate where the
    # dune was completely eroded. Unlike figure 7, the Y-axes in these
    # plots shows the initial beach width
    # figure_9()

    # Make Figure 10. This shows the 2017 Aspect Ratio v. 2017 Beach
    # Width colored by the volume loss between 2017-2018 (post-Florence)
    # figure_10()

    # Make Supplemental Figure 1. This shows the CDF of
    # dune volumes on Bogue Banks measured from LiDAR with
    # the percentile of the dune volume used marked off
    # supp_figure_1()

    # Make supplemental Figure 2. Perform a multiple
    # linear regression on the field data
    # supp_figure_2_SVR(cv=5)


if __name__ == '__main__':
    main()
