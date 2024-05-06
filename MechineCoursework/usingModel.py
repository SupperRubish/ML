import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from scipy import stats

prediction={
    0: "circle",
    1: "come",
    2: "go",
    3: "wave"
}

# Load the previously saved scaler and model
scaler = joblib.load('scaler.save')
model = load_model('my_model.h5')

# Load new forecast data
X_new = pd.read_excel("./data/pred1_wave.xls")
X_new = X_new.drop(['Time (s)'], axis=1).values
num=(len(X_new)//100)*100
X_new = X_new[:num]

# # Standardise using the same scaler
# X_scaled = scaler.transform(X_new)
#
# # Ensure that the data shape matches the inputs expected by the model
# X_scaled = np.expand_dims(X_scaled, axis=0)  # Increase sample dimensions
# X_scaled = X_scaled.reshape(-1, 100, 4)  # Ensure shape matching


# Predict using models
i=0
while(i<num):
    x = X_new[i:i+100]
    X_scaled = scaler.transform(x)
    X_scaled = X_scaled.reshape(1, 100, 4)
    predictions = model.predict(X_scaled)
    print(predictions)
    predicted_class = np.argmax(predictions, axis=1)
    # print(f"Predicted class: {predicted_class}")
    print(predicted_class)
    mode_result = stats.mode(predicted_class)
    # print(mode_result.mode[0])
    print("预测结果："+str(prediction[predicted_class[0]]))
    i+=100
# predictions = model.predict(X_scaled)
# predicted_class = np.argmax(predictions, axis=1)
# # print(f"Predicted class: {predicted_class}")
# # print(type(predicted_class))
# mode_result = stats.mode(predicted_class)
# print(mode_result.mode[0])
# print("总体预测结果："+str(prediction[mode_result.mode[0]]))