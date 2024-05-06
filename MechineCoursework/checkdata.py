import clean_data
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import rfftfreq
from pandas.plotting._matplotlib import plot
from scipy.fft import rfft
import numpy as np



labels_dict = {
        'circle': 0,  # Assuming 'circle.xls' corresponds to the 'circle' gesture
        'come': 1,  # Assuming 'come.xls' corresponds to the 'come here' gesture
        'go': 2,  # Assuming 'go.xls' corresponds to the 'go away' gesture
        'wave': 3  # Assuming 'wave.xls' corresponds to the 'wave' gesture
    }
names=["c","p","l"]

# Load each dataset and create a combined dataframe with labels
circle_dataframes = []
go_dataframes = []
come_dataframes = []
wave_dataframes = []

original_circle_dataframes = []
original_go_dataframes = []
original_come_dataframes = []
original_wave_dataframes = []

num=0
for i in names:
    for j in range(1,11):
        i=str(i)
        j=str(j)
        file_circle=f'./data/circle/{i}_circle_{j}.xls'
        file_come = f'./data/come/{i}_come_{j}.xls'
        file_go= f'./data/go/{i}_go_{j}.xls'
        file_wave= f'./data/wave/{i}_wave_{j}.xls'
        # Read the data after cleaning
        circle_data = clean_data.cleanData(file_circle)
        come_data = clean_data.cleanData(file_come)
        go_data = clean_data.cleanData(file_go)
        wave_data = clean_data.cleanData(file_wave)
        # Read the original data
        orignal_circle_data=pd.read_excel(file_circle)
        orignal_come_data = pd.read_excel(file_come)
        orignal_go_data = pd.read_excel(file_go)
        orignal_wave_data = pd.read_excel(file_wave)
        # Extract the data
        circle_data = circle_data.drop(['Time (s)'], axis=1)[:1400]
        orignal_circle_data = orignal_circle_data.drop(['Time (s)'], axis=1)[:1400]
        come_data = come_data.drop(['Time (s)'], axis=1)[:1400]
        orignal_come_data = orignal_come_data.drop(['Time (s)'], axis=1)[:1400]
        go_data = go_data.drop(['Time (s)'], axis=1)[:1400]
        orignal_go_data = orignal_go_data.drop(['Time (s)'], axis=1)[:1400]
        wave_data = wave_data.drop(['Time (s)'], axis=1)[:1400]
        orignal_wave_data = orignal_wave_data.drop(['Time (s)'], axis=1)[:1400]
        for k in range(0, 1400, 100): # Label each 100 rows as a gesture
            circle = pd.DataFrame(circle_data.values[k:k + 100])
            o_circle=pd.DataFrame(orignal_circle_data.values[k:k + 100])
            circle_dataframes.append(circle)
            original_circle_dataframes.append(o_circle)

            come = pd.DataFrame(come_data.values[k:k + 100])
            o_come=pd.DataFrame(orignal_come_data.values[k:k + 100])
            come_dataframes.append(come)
            original_come_dataframes.append(o_come)

            go = pd.DataFrame(go_data.values[k:k + 100])
            o_go=pd.DataFrame(orignal_go_data.values[k:k + 100])
            go_dataframes.append(go)
            original_go_dataframes.append(o_go)

            wave = pd.DataFrame(wave_data.values[k:k + 100])
            o_wave=pd.DataFrame(orignal_wave_data.values[k:k + 100])
            wave_dataframes.append(wave)
            original_wave_dataframes.append(o_wave)

column_names = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']
dataframe=[circle_dataframes,go_dataframes,come_dataframes,wave_dataframes,original_circle_dataframes,original_go_dataframes,
           original_come_dataframes,original_wave_dataframes]


def combine_dataframes(dataframes_list):
    # Merge all the DataFrame in the list into one
    combined_df = pd.concat(dataframes_list, ignore_index=True)
    combined_df.columns = column_names
    return combined_df


def add_time_column(df, start_time, sample_interval):
    # df is the data frame, start_time is the start time, sample_interval is the sample interval (in seconds)
    num_samples = df.shape[0]
    time_values = pd.date_range(start=start_time, periods=num_samples, freq=pd.DateOffset(seconds=sample_interval))
    df['Time'] = time_values
    return df

# Create a figure and a grid of subplots which has four rows and one column
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 20), sharex=True)

def prepare_dataframes(dataframe_lists, column_names, start_time, sample_interval):
    # Initialise a list to store the prepared dataframes after processing
    prepared_dataframes = []

    # Iterate over each list of dataframes provided in dataframe_lists
    for df_list in dataframe_lists:
        # Combine all dataframes in the current list into a single dataframe
        combined_df = combine_dataframes(df_list)
        # Set the column names of the combined dataframe
        combined_df.columns = column_names
        # Add a time column to the dataframe based on the start time and the sample interval
        combined_df = add_time_column(combined_df, start_time, sample_interval)
        # Append the processed dataframe to the list of prepared dataframes
        prepared_dataframes.append(combined_df)
    return prepared_dataframes

#---------------------Statistical descriptive data--------------------------------------------

# 合并数据帧并添加描述性统计的函数
def generate_descriptive_stats(dataframe_lists, column_names):
    for df_list in dataframe_lists:
        if df_list:  # Make sure the list is not none
            combined_df = combine_dataframes(df_list)  # Merge data frame list
            combined_df.columns = column_names  # Set the name of column
            pd.set_option('display.max_columns', None)  # Set Pandas to show all columns
            print(combined_df.describe())  # print the description of every column


# Use functions to generate descriptive statistics
generate_descriptive_stats([circle_dataframes, go_dataframes, come_dataframes, wave_dataframes,
                            original_circle_dataframes, original_go_dataframes,
                            original_come_dataframes, original_wave_dataframes],
                            ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)',
                             'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)'])


#--------------------Boxplots-------------------------------------------------
def plot_boxplots(dataframe_lists, column_names, gesture_names):
    for df_list, gesture_name in zip(dataframe_lists, gesture_names):
        # Combine DataFrame lists
        combined_df = combine_dataframes(df_list)
        # Set the names of columns
        combined_df.columns = column_names
        # Draw boxplot
        plt.figure(figsize=(10, 6))
        combined_df.boxplot()
        plt.title(f"{gesture_name} Gesture Boxplot")
        plt.ylabel('Values')
        plt.xticks(rotation=45)
        plt.show()

# Define column names and gesture names
column_names = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']
gesture_names = ['Circle', 'Come', 'Go', 'Wave']

# Draw the boxplots
plot_boxplots([circle_dataframes, come_dataframes, go_dataframes, wave_dataframes], column_names, gesture_names)


# #-----------------------------Time series plots----------------------------------------
# 绘图函数
def plot_each_timeseries(dataframes, labels, time_column='Time'):
    num_columns = len(dataframes[0].columns) - 1  # Drop the [Time(s)] column
    column_names = dataframes[0].columns[:-1]  # Exclude time columns
    num_dataframes = len(dataframes)

    # Plot each column of each DataFrame
    for i in range(num_columns):
        column_name = column_names[i]
        fig, axes = plt.subplots(nrows=num_dataframes, ncols=1, figsize=(10, 15), sharex=True)
        for df, label, ax in zip(dataframes, labels, axes):
            ax.plot(df[time_column], df[column_name], label=f'{label} {column_name}')
            ax.set_title(f'{label} - {column_name}')
            ax.set_xlabel('Time')
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.show()

# Set column names and time
start_time = pd.Timestamp('00:00:00')
sample_interval = 0.01

# Lists of data
dataframe_lists = [circle_dataframes, go_dataframes, come_dataframes, wave_dataframes,
                   original_circle_dataframes, original_go_dataframes,
                   original_come_dataframes, original_wave_dataframes]

# Prepare for the dataframes
prepared_dataframes = prepare_dataframes(dataframe_lists, column_names, start_time, sample_interval)

# Draw the labels
labels = ['Circle', 'Go', 'Come', 'Wave', 'Original Circle', 'Original Go', 'Original Come', 'Original Wave']

# Draw plots
plot_each_timeseries(prepared_dataframes, labels)