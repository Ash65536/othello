
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
import os

# モデルの初期化
initial_model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    max_iter=1000,
    learning_rate='adaptive'
)

# 簡単な初期学習データの作成
X = np.random.rand(100, 128)  # 入力特徴量
y = np.random.rand(100)       # 教師データ

# モデルの初期学習
initial_model.fit(X, y)

# モデルの保存
if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(initial_model, 'model/othello_model.joblib')