import numpy as np
import pandas
import pandas as pd
import pywt

from scipy.signal import lfilter, butter

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define labels based on gesture recognition
from tensorflow.python.ops.signal.fft_ops import fft

# Label every gesture as a number
labels_dict = {
    'circle': 0,
    'come': 1,
    'go': 2,
    'wave': 3
}
names = ["c", "p", "l"]  # Names of group members

# Replace outliers using the IQR method
def replace_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # Q1 is the first quartile
    Q3 = df[column].quantile(0.75)  # Q3 is the third quartile
    IQR = Q3 - Q1  # IQR is the difference between the third quartile and the first quartile
    lower_bound = Q1 - 1.5 * IQR  # Define the lower bound
    upper_bound = Q3 + 1.5 * IQR  # Define the upper bound
    median_value = df[column].median()
    # Replace values that are outside a specified range with median value
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), median_value, df[column])
    return df

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Calculate the Nyquist frequency, which is half the sampling rate
    nyq = 0.5 * fs
    # Normalize the cutoff frequency as a fraction of the Nyquist frequency
    normal_cutoff = cutoff / nyq
    # Generate the filter coefficients using the Butterworth filter design
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the data using the filter coefficients
    y = lfilter(b, a, data)
    return y

def wavelet_denoise(data, wavelet, mode='soft', level=1):
    # Decompose to wavelet coefficients
    coeff = pywt.wavedec(data, wavelet, mode="per")
    # Calculate threshold
    sigma = np.std(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    # Apply threshold
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode=mode) for i in coeff[1:]]
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode="per")


def cleanData(file_path):
    scaler = StandardScaler()
    try:
        # Load data
        data = pd.read_excel(file_path)
        # print(data)

        # Apply moving average filter

        guolv_list = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)',
                             'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']



        # Replace outliers

        for column in guolv_list:
            if column in data.columns:
                data = replace_outliers(data, column)

        # Z-score normalization
        if all(col in data.columns for col in guolv_list):
            data[guolv_list] = scaler.fit_transform(data[guolv_list])

        # Drop the NA value
        data.dropna()

        # Normalized data（Min-Max Normalization）
        # data = data.select_dtypes(include=[np.number])
        # column_names = data.columns
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # data = scaler.fit_transform(data)
        # # If need, convert the results back to a DataFrame
        # data = pd.DataFrame(data, columns=column_names)


        # # Rolling Average (White Noise Removal) This method is very effective in reducing random noise,
        # # but may not be suitable for retaining all significant signals (e.g. peaks) in the data.
        # for column in data:
        #     if column in data.columns:
        #         data[column] = data[column].rolling(window=5, center=True).mean()



        # # # Fourier Transform, the time series is converted from the time domain to the frequency domain,
        # # # which allows high frequency noise components to be identified and removed.
        # for column in data:
        #     if column in data.columns:
        #         data[column] = np.abs(fft(data[column]))

        # Low-pass filters, which allow low-frequency signals to pass through but cut the amplitude of
        # high-frequency signals, are suitable for removing noise caused by fast and brief vibrations.
        # cutoff = 3.5  # cutoff frequency
        # fs = 30  # sampling frequency
        # order = 6  # filter order
        # for column in data:
        #      if column in data.columns:
        #          data[column]=butter_lowpass_filter(data,3,1500,1)

        # Wavelet denoising, which is suitable for processing signals that
        # may contain non-smooth or multi-scale noise
        wavelet = 'db1'

        # Need a dictionary to store the denoised data
        newdata = {}

        for column in data.columns:  # Traverse column names directly

            column_data = data[column]
            # Make sure that the column data is numeric, as transform needs numeric data
            if column_data.dtype == np.number:
                newdata[column] = wavelet_denoise(column_data, wavelet=wavelet, mode='soft', level=1)

        # Convert the result back to DataFrame
        newdata_df = pd.DataFrame(newdata)


        return newdata_df




    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
# cleanData('./data/go/c_go_1.xls')

