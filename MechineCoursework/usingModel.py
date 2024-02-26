import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from scipy import stats

prediction={
    0: "circle",
    1 : "guolai",
    2: "huishou",
    3: "zoukai"
}

X_new = pd.read_excel("./prediction/1.xls")
X_new = X_new.drop(['Time (s)'], axis=1).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_new)
# 为1D CNN增加一个样本维度
X_scaled = np.expand_dims(X_scaled, axis=0)
X_scaled = X_scaled.reshape(-1, 4, 1)

# 加载模型
model = load_model('my_model.h5')
# 假设 X_new 是您要进行预测的新数据
predictions = model.predict(X_scaled)
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted class: {predicted_class}")
print(type(predicted_class))
mode_result = stats.mode(predicted_class)
print("预测结果："+str(prediction[mode_result.mode[0]]))

