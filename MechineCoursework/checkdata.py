import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np

# labels_dict = {
#     'circle': 0,  # Assuming 'circle.xls' corresponds to the 'circle' gesture
#     'come': 1,  # Assuming 'come.xls' corresponds to the 'come here' gesture
#     'go': 2,  # Assuming 'go.xls' corresponds to the 'go away' gesture
#     'waving': 3  # Assuming 'waving.xls' corresponds to the 'waving' gesture
# }

labels_dict=['circle','come','go','waving']

dataframes = []
for gesture in labels_dict:
    for i in range(1,16):
        file_path = f'./data/{gesture}/{gesture}{i}.xls'
        gesture_data = pd.read_excel(file_path)
        # gesture_data['label'] = label
        dataframes.append(gesture_data)

combined_data = pd.concat(dataframes, ignore_index=True)


print("--------------------查询是否有缺失值-----------------------------")
#查询是否有缺失值
combined_data.head(),combined_data.info()
missing_info = combined_data.isnull().sum()
print(missing_info)


print("--------------------检查重复记录-----------------------------")
#检查重复记录
# # Check for duplicate rows in the dataframe
duplicate_rows = combined_data[combined_data.duplicated()]
#
# # Count the number of duplicate rows
num_duplicate_rows = duplicate_rows.shape[0]
#
print(num_duplicate_rows)
print(duplicate_rows)

# Let's check for duplicate records in the data
duplicates = combined_data.duplicated().sum()
print("the abnormal value(异常值为) is :"+str(duplicates))

#判断数据格式
print("--------------------判断数据格式-----------------------------")

# Applying previous cleaning steps
# Dropping the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in combined_data.columns:
    combined_data.drop('Unnamed: 0', axis=1, inplace=True)

# Checking for and removing duplicates
original_data = combined_data.drop_duplicates()

# Now let's check the data types to ensure they are correct
print(original_data.dtypes)

print("--------------------zscore检查异常值-----------------------------")

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.boxplot(data=combined_data)
plt.title('Boxplot of Features to Identify Outliers')
plt.xticks(rotation=45)
plt.show()

combined_data['Z_Score_AccX'] = zscore(combined_data['Linear Acceleration x (m/s^2)'])
combined_data['Z_Score_AccY'] = zscore(combined_data['Linear Acceleration y (m/s^2)'])
combined_data['Z_Score_AccZ'] = zscore(combined_data['Linear Acceleration z (m/s^2)'])
combined_data['Z_Score_AbsoluteAcc'] = zscore(combined_data['Absolute acceleration (m/s^2)'])

# Display Z-scores
# print(combined_data['Z_Score_AccX'].head() )
# print(combined_data['Z_Score_AccY'].head() )
# print(combined_data['Z_Score_AccZ'].head() )
# print(combined_data['Z_Score_AbsoluteAcc'].head() )


# 标记异常值
combined_data['is_outlier'] = (combined_data['Z_Score_AccX'].abs() >= 3) | \
                              (combined_data['Z_Score_AccY'].abs() >= 3) | \
                              (combined_data['Z_Score_AccZ'].abs() >= 3) | \
                              (combined_data['Z_Score_AbsoluteAcc'].abs() >= 3)

# 计算标记为异常值的行数
outlier_count = combined_data['is_outlier'].sum()

print(f"异常值的行数: {outlier_count}")

print('--------------------处理异常值----------------------------')





