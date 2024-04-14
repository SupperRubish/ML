import clean_data


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
    for j in range(1,6):
        i=str(i)
        j=str(j)
        file_circle=f'./data/circle/{i}_circle_{j}.xls'
        file_come = f'./data/come/{i}_come_{j}.xls'
        file_go= f'./data/go/{i}_go_{j}.xls'
        file_wave= f'./data/wave/{i}_wave_{j}.xls'

        circle_data=(clean_data.cleanData(file_circle)).drop(['Time (s)'], axis=1).values[:1400]
        come_data=(clean_data.cleanData(file_come)).drop(['Time (s)'], axis=1).values[:1400]
        go_data=(clean_data.cleanData(file_go)).drop(['Time (s)'], axis=1).values[:1400]
        wave_data=(clean_data.cleanData(file_wave)).drop(['Time (s)'], axis=1).values[:1400]

        for k in range(0,1400,100):

            #circle
            circle=circle_data[k:k+100].copy()
            circle_dataframes.append([circle])
            #come
            come=come_data[k:k+100].copy()
            come_dataframes.append([come])
            #go
            go=go_data[k:k+100].copy()
            go_dataframes.append([go])
            #wave
            wave=wave_data[k:k+100].copy()
            wave_dataframes.append([wave])
            num+=4

# print(dataframes)