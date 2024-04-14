import numpy as np
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

        # Apply moving average filter

        guolv_list = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)',
                             'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']



        # Replace outliers 异常值替换

        for column in guolv_list:
            if column in data.columns:
                data = replace_outliers(data, column)

        #标准化数据
        # if all(col in data.columns for col in guolv_list):
        #     data[guolv_list] = scaler.fit_transform(data[guolv_list])

        # 去NA值
        data.dropna()

        #归一化数据
        # data = data.select_dtypes(include=[np.number])
        # column_names = data.columns
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # data = scaler.fit_transform(data)
        # # 如果需要将结果转换回DataFrame
        # data = pd.DataFrame(data, columns=column_names)


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
        #          data[column]=butter_lowpass_filter(data,3.5,30,6)

        #小波去噪，适合处理可能包含非平稳或多尺度噪声的信号
        # wavelet = 'db8'
        # for column in data:
        #      if column in data.columns:
        #          data[column]=wavelet_denoise(data, wavelet=wavelet, mode='soft', level=1)





        return data





    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
# cleanData('./data/go/c_go_1.xls')


d=pd.read_excel("./data/go/c_go_1.xls")
print(d)

#################################检测白噪音################################################
# # 定义要测试的列
# columns_to_test = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)',
#                    'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']
#
# # 对每一列进行 Ljung-Box Q 测试
# results = {}
# for column in columns_to_test:
#     # 确保没有 NaN 值，删除它们
#     series = d[column].dropna()
#
#     # 进行 Ljung-Box Q 测试
#     lb_value, p_value = acorr_ljungbox(series, lags=[10], return_df=False)
#
#     # 存储测试结果
#     results[column] = {'Ljung-Box Q Statistic': lb_value, 'P-value': p_value}
#
# # 输出结果
# for col, result in results.items():
#     print(f"{col}:")
#     print(f"  Ljung-Box Q Statistic: {result['Ljung-Box Q Statistic']}")
#     print(f"  P-value: {result['P-value']}")
#     print()
# ######################################################################################