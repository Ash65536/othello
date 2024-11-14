import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
import os
import flask
import numpy
import sklearn
import tensorflow
import torch
import time

class OthelloModel:
    def __init__(self):
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        self.model_path = os.path.join(model_dir, 'othello_model.joblib')
        self.model = self._load_or_create_model()
        self.training_data = []
        self.log_dir = os.path.join('model', 'model_logs')
        os.makedirs(self.log_dir, exist_ok=True)

    def _load_or_create_model(self):
        try:
            if os.path.exists(self.model_path):
                return joblib.load(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
        
        # モデルが存在しない場合は新規作成
        return MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1000,
            learning_rate='adaptive'
        )

    def board_to_features(self, board):
        # 盤面を1次元配列に変換
        features = []
        for row in board:
            for cell in row:
                if cell == 'black':
                    features.append(1)
                elif cell == 'white':
                    features.append(-1)
                else:
                    features.append(0)
        return np.array(features)

    def predict_move(self, board, possible_moves):
        if not possible_moves:
            return None
        
        features = self.board_to_features(board)
        predictions = []
        
        for move in possible_moves:
            move_features = np.zeros(64)
            move_features[move[0] * 8 + move[1]] = 1
            combined_features = np.concatenate([features, move_features])
            prediction = self.model.predict([combined_features])[0]
            predictions.append((move, prediction))
        
        return max(predictions, key=lambda x: x[1])[0]

    def store_game_data(self, game_history, winner):
        for state, move in game_history:
            features = self.board_to_features(state)
            move_features = np.zeros(64)
            move_features[move[0] * 8 + move[1]] = 1
            combined_features = np.concatenate([features, move_features])
            
            # 勝者の手は高い報酬、敗者の手は低い報酬
            reward = 1.0 if winner == state['current_player'] else -1.0
            self.training_data.append((combined_features, reward))

    def log_training(self, message):
        log_file = os.path.join(self.log_dir, f'training_{time.strftime("%Y%m%d")}.log')
        with open(log_file, 'a', encoding='utf-8') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{timestamp}] {message}\n')

    def train(self):
        if not self.training_data:
            return False
            
        X = np.array([x for x, _ in self.training_data])
        y = np.array([y for _, y in self.training_data])
        
        # 学習前の性能を記録
        pre_score = self.model.score(X, y)
        
        self.model.partial_fit(X, y)
        
        # 学習後の性能を記録
        post_score = self.model.score(X, y)
        
        # 学習ログを記録
        self.log_training(
            f"Training completed - Samples: {len(X)}, "
            f"Pre-score: {pre_score:.4f}, "
            f"Post-score: {post_score:.4f}"
        )
        
        # モデルを保存
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            self.log_training(f"Model saved to {self.model_path}")
        except Exception as e:
            self.log_training(f"Error saving model: {str(e)}")
        
        # 学習データをクリア
        self.training_data = []
        return True

print("All libraries installed successfully!")