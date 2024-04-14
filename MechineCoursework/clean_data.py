import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

# Define labels based on gesture recognition
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


def cleanData(file_path):
    scaler = StandardScaler()
    try:
        # Load data
        data = pd.read_excel(file_path)

        # Apply moving average filter

        guolv_list = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)',
                             'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']



        # Drop NA values that arise from rolling mean
        # data = data.dropna()
        # 滚动平均值(去白噪音）
        for column in data:
            if column in data.columns:
                data[column] = data[column].rolling(window=5, center=True).mean()

        # 傅里叶变换，去白噪音
        for column in data:
            if column in data.columns:
                data[column] = np.abs(fft(data[column]))

        # Replace outliers
        for column in guolv_list:
            if column in data.columns:
                data = replace_outliers(data, column)
        return data






    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
cleanData('./data/go/c_go_1.xls')
print(pd.read_excel("./data/go/c_go_1.xls"))