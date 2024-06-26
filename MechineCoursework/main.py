import joblib
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD, Adam
from scipy.signal import butter, lfilter
from clean_data import clean
# We will load each file and assign labels to each gesture class
# The labels are based on the filenames which seem to indicate the gesture

# Assign labels based on the file names (these should be verified with the user)
labels_dict = {
        'circle': 0,  # Assuming 'circle.xls' corresponds to the 'circle' gesture
        'come': 1,  # Assuming 'come.xls' corresponds to the 'come here' gesture
        'go': 2,  # Assuming 'go.xls' corresponds to the 'go away' gesture
        'wave': 3  # Assuming 'wave.xls' corresponds to the 'wave' gesture
    }
names=["c","p","l"]

# Load each dataset and create a combined dataframe with labels

dataframes = []
#cheng的指导

for i in names:
    for j in range(1,6):
        i=str(i)
        j=str(j)
        file_circle=f'./data/circle/{i}_circle_{j}.xls'
        file_come = f'./data/come/{i}_come_{j}.xls'
        file_go= f'./data/go/{i}_go_{j}.xls'
        file_wave= f'./data/wave/{i}_wave_{j}.xls'
        circle_data=pd.read_excel(file_circle)[:1500]
        come_data=pd.read_excel(file_come)[:1500]
        go_data=pd.read_excel(file_go)[:1500]
        wave_data=pd.read_excel(file_wave)[:1500]
        for k in range(0,1500,100):

            #circle
            circle=circle_data.iloc[k:k+100].copy()

            circle['label'] = 0
            dataframes.append(circle)
            #come
            come=come_data.iloc[k:k+100].copy()
            come['label']=1
            dataframes.append(come)
            #go
            go=go_data.iloc[k:k+100].copy()
            go['label']=2
            dataframes.append(go)
            #wave
            wave=wave_data.iloc[k:k+100].copy()
            wave['label']=3
            dataframes.append(wave)

# print(dataframes)
# # Combine all dataframes into one
combined_data = pd.concat(dataframes, ignore_index=True)

import pandas as pd

# 假设 'data' 是之前加载的DataFrame
window_size = 5  # 定义移动平均的窗口大小

# 对每个需要处理的列应用移动平均
smoothed_data = combined_data.copy()
smoothed_data['Linear Acceleration x (m/s^2)'] = combined_data['Linear Acceleration x (m/s^2)'].rolling(window=window_size).mean()
smoothed_data['Linear Acceleration y (m/s^2)'] = combined_data['Linear Acceleration y (m/s^2)'].rolling(window=window_size).mean()
smoothed_data['Linear Acceleration z (m/s^2)'] = combined_data['Linear Acceleration z (m/s^2)'].rolling(window=window_size).mean()

# 填充NaN值，这些NaN是由于滚动平均而产生的。可以用前向填充或后向填充。
smoothed_data.fillna(method='bfill', inplace=True)

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


# 应用滤波器（示例参数）
cutoff = 3.667  # 截止频率（需根据数据调整）
fs = 50.0       # 采样率（需根据数据调整）
order = 6       # 滤波器阶数

# 假设 'data' 是之前加载的DataFrame
filtered_data = combined_data.copy()
filtered_data['Linear Acceleration x (m/s^2)'] = butter_lowpass_filter(combined_data['Linear Acceleration x (m/s^2)'], cutoff, fs, order)
filtered_data['Linear Acceleration y (m/s^2)'] = butter_lowpass_filter(combined_data['Linear Acceleration y (m/s^2)'], cutoff, fs, order)
filtered_data['Linear Acceleration z (m/s^2)'] = butter_lowpass_filter(combined_data['Linear Acceleration z (m/s^2)'], cutoff, fs, order)
# #
# # # Check the combined dataframe
# print(combined_data.head()),
# print(combined_data.tail()),
# print(combined_data['label'].value_counts())

# combined_data = clean()
#
# 一维卷积神经网络（1D CNN）来训练模型。在开始之前，我们需要执行以下步骤：
#
# 数据预处理：确保所有输入数据都是数值类型，并且没有缺失值。
# 特征选择：虽然1D CNN可以直接从原始数据中学习，但有时候对数据进行一些变换（如快速傅里叶变换（FFT））或计算一些统计特征可能有助于模型学习。
# 数据划分：将数据分为训练集和测试集。
# 模型构建：构建1D CNN模型架构。
# 模型训练：使用训练集数据来训练模型。
# 模型评估：使用测试集数据来评估模型性能。
# 接下来，我将开始这个过程。首先，我们需要将数据划分为训练集和测试集。然后，我会构建一个简单的1D CNN模型并训练它。

y = to_categorical(filtered_data['label'])

# 数据标准化
X = filtered_data.drop(['Time (s)', 'label'], axis=1).values
print(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.save')
print(X_scaled)




# 划分数据为训练集和测试集
#  1:4 分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print(X_train.shape)
print(X_train)


# print(X_train)

# 构建1D CNN模型
# Sequential模型是Keras中的一种模型，用于线性堆叠层。这意味着您可以按顺序添加一层又一层，每层只有一个输入和一个输出。
model = Sequential()
# 添加一个一维卷积层（Conv1D）。这是1D CNN的核心，用于提取序列数据的特征。
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
# 128，0.7621.64, 0.7547
# 添加一个Dropout层，用于减少过拟合。Dropout通过在训练过程中随机丢弃（设置为零）网络中的一部分神经元输出，来提高模型的泛化能力。
# 0.5表示丢弃率为50%，即在训练过程中随机选择50%的神经元输出设置为0
model.add(Dropout(0.5))
# 添加一个一维最大池化层（MaxPooling1D）。池化层用于降低数据的维度，减少计算量，同时保持重要信息。即每2个值中选择最大的那个作为输出。
model.add(MaxPooling1D(pool_size=2))
# 添加一个Flatten层，将之前层的输出展平。这是从卷积层或池化层到全连接层（Dense层）过渡时常用的技术。
# model.add(Flatten())
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# 编译模型
# 优化器和学习率
# SGD
# from tensorflow.keras.optimizers import SGD
#
# # 使用SGD优化器，指定学习率
# optimizer = SGD(learning_rate=0.01)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizer,
#               metrics=['accuracy'])

# RMSprop
# from tensorflow.keras.optimizers import RMSprop
#
# # 使用RMSprop优化器，指定学习率
# optimizer = RMSprop(learning_rate=0.001)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizer,
#               metrics=['accuracy'])

# 学习率衰减
# from tensorflow.keras.optimizers import Adam

# 初始化一个学习率衰减函数
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

# # 使用带有学习率调度器的Adam优化器
optimizer = Adam(learning_rate=lr_schedule)

# 学习率越小，离最优解越近，效率越慢
# optimizer = SGD(learning_rate=0.01)

# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizer,
#               metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# 训练模型
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),callbacks=early_stopping)

# 评估模型
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print(f'Test accuracy: {accuracy}')

model.save('my_model.h5')  # HDF5文件，需要安装h5py





