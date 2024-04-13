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


# # Add function to compute additional statistics
# def add_statistical_features(data, column):
#     data[f'{column}_mean'] = data[column].mean()
#     data[f'{column}_std'] = data[column].std()
#     data[f'{column}_skew'] = skew(data[column])
#     data[f'{column}_kurt'] = kurtosis(data[column])

scaler = StandardScaler()

# Process data from multiple files
for name in names:
    for j in range(1, 6):
        for gesture, label in labels_dict.items():
            file_path = f'./data/{gesture}/{name}_{gesture}_{j}.xls'
            output_file_path = f'./clean_data/{gesture}/clean_{name}_{gesture}_{j}.xlsx'
            try:
                # Load data
                data = pd.read_excel(file_path)

                # Apply moving average filter
                columns_to_filter = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)',
                                     'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']
                for column in columns_to_filter:
                    if column in data.columns:
                        data[column] = data[column].rolling(window=5, center=True).mean()
                    #     add_statistical_features(data, column)

                # Standardize data
                if all(col in data.columns for col in columns_to_filter):
                    data[columns_to_filter] = scaler.fit_transform(data[columns_to_filter])

                # # Fourier Transform for frequency domain features
                # if 'Absolute acceleration (m/s^2)' in data.columns:
                #     data['fft_abs_acceleration'] = abs(fft(data['Absolute acceleration (m/s^2)']))
                # else:
                #     print("FFT column 'Absolute acceleration (m/s^2)' not found.")

                # Drop NA values that arise from rolling mean
                data = data.dropna()

                # Replace outliers
                for column in columns_to_filter:
                    if column in data.columns:
                        data = replace_outliers(data, column)

                # Assign label
                # data['Label'] = label

                # Save the processed data to a new file
                data.to_excel(output_file_path, index=False, engine='openpyxl')
                print(f"Processed data saved to {output_file_path}")

            except FileNotFoundError:
                print(f"File {file_path} not found.")
            except Exception as e:
                print(f"An error occurred: {e}")
