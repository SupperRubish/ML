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

labels_dict = {
    'circle': 0,
    'come': 1,
    'go': 2,
    'wave': 3
}
names = ["c", "p", "l"]

# Replace outliers using the IQR method
def replace_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median_value = df[column].median()
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), median_value, df[column])
    return df

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
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



        # Replace outliers 异常值替换

        for column in guolv_list:
            if column in data.columns:
                data = replace_outliers(data, column)

        #标准化数据（Z-score normalization）
        # if all(col in data.columns for col in guolv_list):
        #     data[guolv_list] = scaler.fit_transform(data[guolv_list])

        # 去NA值
        data.dropna()

        #归一化数据（Min-Max Normalization）
        data = data.select_dtypes(include=[np.number])
        column_names = data.columns
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        # 如果需要将结果转换回DataFrame
        data = pd.DataFrame(data, columns=column_names)


        # 滚动平均值(去白噪音）这种方法对于减少随机噪声非常有效，但可能不适合保留数据中的所有重要信号（如峰值）。
        # for column in data:
        #     if column in data.columns:
        #         data[column] = data[column].rolling(window=5, center=True).mean()



        # # # 傅里叶变换，时间序列从时域转换到频域，这使得可以识别并去除高频噪声成分。处理完后，可以进行逆变换回时域。
        # for column in data:
        #     if column in data.columns:
        #         data[column] = np.abs(fft(data[column]))

        #低通滤波器，这种方法允许低频信号通过，但削减高频信号的幅度，适合去除因快速且短暂的震动引起的噪声。
        # cutoff = 3.5  # 截止频率（需根据数据特性调整）
        # fs = 30  # 采样频率（同样需根据数据特性调整）
        # order = 6  # 滤波器阶数
        # for column in data:
        #      if column in data.columns:
        #          data[column]=butter_lowpass_filter(data,3,1500,1)

        #小波去噪，适合处理可能包含非平稳或多尺度噪声的信号
        wavelet = 'db1'
        newdata = {}

        for column in data.columns:  # 直接遍历列名

            column_data = data[column]
            if column_data.dtype == np.number:  # 确保列数据是数值型，因为小波变换需要数值数据
                newdata[column] = wavelet_denoise(column_data, wavelet=wavelet, mode='soft', level=1)

        # 将结果转换回 DataFrame
        newdata_df = pd.DataFrame(newdata)



        return newdata_df





    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
# cleanData('./data/go/c_go_1.xls')

