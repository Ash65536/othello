import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
import os

# モデル保存ディレクトリの確認と作成
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, 'othello_model.joblib')

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

# モデルの保存（絶対パスを使用）
try:
    joblib.dump(initial_model, model_path)
    print(f"Model successfully saved to {model_path}")
except Exception as e:
    print(f"Error saving model: {e}")