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
num=0
for i in names:
    for j in range(1,6):
        i=str(i)
        j=str(j)
        file_circle=f'./data/circle/{i}_circle_{j}.xls'
        file_come = f'./data/come/{i}_come_{j}.xls'
        file_go= f'./data/go/{i}_go_{j}.xls'
        file_wave= f'./data/wave/{i}_wave_{j}.xls'
        circle_data=(pd.read_excel(file_circle)).drop(['Time (s)'], axis=1).values[:1400]
        come_data=(pd.read_excel(file_come)).drop(['Time (s)'], axis=1).values[:1400]
        go_data=(pd.read_excel(file_go)).drop(['Time (s)'], axis=1).values[:1400]
        wave_data=(pd.read_excel(file_wave)).drop(['Time (s)'], axis=1).values[:1400]
        for k in range(0,1400,100):

            #circle
            circle=circle_data[k:k+100].copy()
            dataframes.append([circle,0])
            #come
            come=come_data[k:k+100].copy()
            dataframes.append([come,1])
            #go
            go=go_data[k:k+100].copy()
            dataframes.append([go,2])
            #wave
            wave=wave_data[k:k+100].copy()
            dataframes.append([wave,3])
            num+=4

# print(dataframes)


actions, labels = zip(*dataframes)

# 将所有动作数据堆叠起来形成一个二维数组，以便进行标准化
actions_stacked = np.vstack(actions)

# 初始化标准化器
scaler = StandardScaler()

# 训练标准化器并标准化数据
actions_scaled = scaler.fit_transform(actions_stacked)

# 保存标准化器
joblib.dump(scaler, 'scaler.save')

# 将标准化的数据重新划分回原来的手势块
actions_scaled_reshaped = actions_scaled.reshape(num, 100, 4)

# 对标签进行独热编码
y = to_categorical(labels)


# 划分数据为训练集和测试集
#  1:4 分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(actions_scaled_reshaped, y, test_size=0.2, random_state=42)

print(X_train)
print(y_train)

# 构建1D CNN模型
# Sequential模型是Keras中的一种模型，用于线性堆叠层。这意味着您可以按顺序添加一层又一层，每层只有一个输入和一个输出。
model = Sequential()
# 添加一个一维卷积层（Conv1D）。这是1D CNN的核心，用于提取序列数据的特征。
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 4)))
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
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),callbacks=early_stopping)

# 评估模型
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print(f'Test accuracy: {accuracy}')

model.save('my_model.h5')  # HDF5文件，需要安装h5py





