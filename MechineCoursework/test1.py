import pandas as pd
test_file = pd.read_excel('./data/c_come_4.xls')
dataframes=[]
num=0
i = 0
while i < len(test_file) - 1:
    if test_file.iloc[i, 3] < 0:
        j = i
        while test_file.iloc[j, 3] < 0 and j<len(test_file)-1:
            print(test_file.iloc[j, 3])
            j += 1

        come = test_file.iloc[i:j - 1].copy()
        come['label'] = 0
        dataframes.append(come)
        i = j  # 更新i的值为j，以跳过已经处理过的部分
        num+=1
    else:
        i += 1  # 如果当前的行不满足条件，则简单地将i增加1

print(num)
print(dataframes)
