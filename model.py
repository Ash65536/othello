import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
import os
import flask
import numpy
import sklearn
import tensorflow
import torch

class OthelloModel:
    def __init__(self):
        self.model_path = 'model/othello_model.joblib'
        self.model = self._load_or_create_model()
        self.training_data = []

    def _load_or_create_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
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

    def train(self):
        if not self.training_data:
            return False
            
        X = np.array([x for x, _ in self.training_data])
        y = np.array([y for _, y in self.training_data])
        
        self.model.partial_fit(X, y)
        joblib.dump(self.model, self.model_path)
        
        # 学習データをクリア
        self.training_data = []
        return True

print("All libraries installed successfully!")