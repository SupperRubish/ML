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

#cheng的指导
num=0
for i in names:
    for j in range(1,11):
        i=str(i)
        j=str(j)
        file_circle=f'./data/circle/{i}_circle_{j}.xls'
        file_come = f'./data/come/{i}_come_{j}.xls'
        file_go= f'./data/go/{i}_go_{j}.xls'
        file_wave= f'./data/wave/{i}_wave_{j}.xls'

        circle_data = clean_data.cleanData(file_circle)
        come_data = clean_data.cleanData(file_come)
        go_data = clean_data.cleanData(file_go)
        wave_data = clean_data.cleanData(file_wave)
        if circle_data is not None and not circle_data.empty:
            circle_data = circle_data.drop(['Time (s)'], axis=1)[:1400]
        if come_data is not None and not come_data.empty:
            come_data = come_data.drop(['Time (s)'], axis=1)[:1400]
        if go_data is not None and not go_data.empty:
            go_data = go_data.drop(['Time (s)'], axis=1)[:1400]
        if wave_data is not None and not wave_data.empty:
            wave_data = wave_data.drop(['Time (s)'], axis=1)[:1400]

            for k in range(0, 1400, 100):
                circle = pd.DataFrame(circle_data.values[k:k + 100])
                circle_dataframes.append(circle)
                come = pd.DataFrame(come_data.values[k:k + 100])
                come_dataframes.append(come)
                go = pd.DataFrame(go_data.values[k:k + 100])
                go_dataframes.append(go)
                wave = pd.DataFrame(wave_data.values[k:k + 100])
                wave_dataframes.append(wave)


def combine_dataframes(dataframes_list):
    # 将列表中的所有 DataFrame 合并为一个
    combined_df = pd.concat(dataframes_list, ignore_index=True)
    return combined_df

# 假设 circle_dataframes, come_dataframes, go_dataframes, 和 wave_dataframes 已正确填充数据
combined_circle_df = combine_dataframes(circle_dataframes)
combined_come_df = combine_dataframes(come_dataframes)
combined_go_df = combine_dataframes(go_dataframes)
combined_wave_df = combine_dataframes(wave_dataframes)


def plot_boxplot(df, title):
    plt.figure(figsize=(10, 6))  # 设置图形的大小
    df.boxplot()  # 使用 DataFrame 的 boxplot 方法
    plt.title(title)
    plt.ylabel('Values')
    plt.xticks(rotation=45)  # 旋转 x 轴标签，以便清楚地看到每个特征的名称
    plt.show()

# 绘制每种手势的箱形图
plot_boxplot(combined_circle_df, "Circle Gesture Boxplot")
plot_boxplot(combined_come_df, "Come Gesture Boxplot")
plot_boxplot(combined_go_df, "Go Gesture Boxplot")
plot_boxplot(combined_wave_df, "Wave Gesture Boxplot")

# 假设 circle_dataframes 已经按照前面的描述填充数据
# if circle_dataframes:
#     # 选择第一个数据帧
#     signal = circle_dataframes[0].values.flatten()  # 将 DataFrame 转换为一维数组
#
#     # 计算 FFT
#     fft_spectrum = rfft(signal)
#     # 计算对应的频率
#     frequencies = rfftfreq(len(signal), 1/1500)  # 假设采样率为1500Hz
#
#     # 绘制频谱
#     plt.figure(figsize=(10, 6))
#     plt.plot(frequencies, np.abs(fft_spectrum))
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Amplitude')
#     plt.title('FFT Spectrum')
#     plt.grid(True)
#     plt.show()
# else:
#     print("No data available for FFT calculation.")