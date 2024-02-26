import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from scipy import stats

prediction={
    0: "circle",
    1: "come",
    2: "go",
    3: "waving"
}

# 加载之前保存的scaler和模型
scaler = joblib.load('scaler.save')
model = load_model('my_model.h5')

# 加载新的预测数据
X_new = pd.read_excel("./prediction/go1.xls")
X_new = X_new.drop(['Time (s)'], axis=1).values

# 使用相同的scaler进行标准化
X_scaled = scaler.transform(X_new)

# 确保数据形状与模型期待的输入匹配
X_scaled = np.expand_dims(X_scaled, axis=0)  # 增加样本维度
X_scaled = X_scaled.reshape(-1, 4, 1)  # 确保形状匹配

# 使用模型进行预测
predictions = model.predict(X_scaled)
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted class: {predicted_class}")
print(type(predicted_class))
mode_result = stats.mode(predicted_class)
print(mode_result.mode[0])
print("预测结果："+str(prediction[mode_result.mode[0]]))

