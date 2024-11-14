from flask import Flask, request, jsonify
from random import choice
import random
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from flask_cors import CORS
from model import OthelloModel
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os  # osモジュールを追加
import json  # jsonモジュールを追加

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5500", "http://127.0.0.1:5500"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Access-Control-Allow-Origin"],
        "supports_credentials": True,
        "max_age": 3600
    }
})  # Add this line after creating the Flask app

model = OthelloModel()
game_history = []

# トランスジションテーブル（プロセス間で共有）
class TranspositionTable:
    def __init__(self):
        self.table = mp.Manager().dict()
    
    def get(self, key):
        return self.table.get(key)
    
    def set(self, key, value):
        self.table[key] = value
    
    def clear(self):
        self.table.clear()

trans_table = TranspositionTable()

# DQNモデルの定義
class OthelloDQN(nn.Module):
    def __init__(self):
        super(OthelloDQN, self).__init__()
        # 入力: 3チャンネル (自分の石、相手の石、空きマス) の8x8ボード
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64)  # 出力は8x8=64の行動空間
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 128 * 8 * 8)
        return self.fc_layers(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.cat(state), 
                torch.tensor(action), 
                torch.tensor(reward), 
                torch.cat(next_state),
                torch.tensor(done))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = OthelloDQN().to(self.device)
        self.target_net = OthelloDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer()
        
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 0.1
        self.target_update = 10
        self.steps_done = 0

    def get_state(self, board):
        state = np.zeros((3, 8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                if board[i][j] == 'white':
                    state[0][i][j] = 1
                elif board[i][j] == 'black':
                    state[1][i][j] = 1
                else:
                    state[2][i][j] = 1
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def select_action(self, state, valid_moves):
        if not valid_moves:
            return None
            
        self.steps_done += 1
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                valid_q_values = {tuple(move): q_values[0][move[0] * 8 + move[1]].item() 
                                for move in valid_moves}
                return max(valid_q_values.items(), key=lambda x: x[1])[0]
        return tuple(random.choice(valid_moves))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Q(s_t, a) computation
        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(-1))
        
        # max_a Q(s_{t+1}, a) computation
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
        
        # Expected Q values
        expected_state_action_values = rewards + (self.gamma * next_state_values)
        
        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)

    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_done = checkpoint['steps_done']

# DQNエージェントのインスタンス化と既存モデルの読み込み
dqn_agent = DQNAgent()
try:
    dqn_agent.load_model('model/othello_dqn_model.pth')
    print("Loaded existing DQN model")
except:
    print("Starting with a new DQN model")

# 棋譜管理用のクラスを追加
class GameRecord:
    def __init__(self):
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.moves = []
        self.result = None
        
    def add_move(self, board, move, color):
        self.moves.append({
            'board': copy.deepcopy(board),
            'move': move,
            'color': color
        })
    
    def set_result(self, winner):
        self.result = winner
    
    def save(self):
        record_dir = os.path.join('model', 'game_records')
        os.makedirs(record_dir, exist_ok=True)
        
        filename = f'game_{self.timestamp}.json'
        filepath = os.path.join(record_dir, filename)
        
        data = {
            'timestamp': self.timestamp,
            'moves': self.moves,
            'result': self.result
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

class LearningLogger:
    def __init__(self):
        self.log_dir = os.path.join('model', 'learning_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_log = os.path.join(
            self.log_dir, 
            f'learning_log_{time.strftime("%Y%m%d")}.txt'
        )
    
    def log(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.current_log, 'a', encoding='utf-8') as f:
            f.write(f'[{timestamp}] {message}\n')

# グローバルインスタンスの作成
learning_logger = LearningLogger()
current_game = None

@app.route('/game_start', methods=['POST'])
def game_start():
    global current_game
    current_game = GameRecord()
    return jsonify({"status": "success"})

@app.route('/ai_move', methods=['POST'])
def ai_move():
    data = request.json
    board = data.get('board')
    color = data.get('color', 'white')
    difficulty = data.get('difficulty', 'normal')
    
    # 現在の盤面を棋譜に記録
    if current_game:
        last_move = data.get('last_move')
        if last_move:
            current_game.add_move(board, last_move, 'black')
    
    if difficulty == 'dqn':
        # DQNによる手の選択
        state = dqn_agent.get_state(board)
        valid_moves = find_possible_moves(board, color)
        move = dqn_agent.select_action(state, valid_moves)
        
        if move:
            # 経験を保存
            new_board = copy.deepcopy(board)
            apply_move(new_board, move[0], move[1], color)
            next_state = dqn_agent.get_state(new_board)
            reward = evaluate_board(new_board, color)
            done = not has_valid_move(new_board, 'black') and not has_valid_move(new_board, 'white')
            
            dqn_agent.memory.push(state, move, reward, next_state, done)
            
            # 学習
            if len(dqn_agent.memory) >= dqn_agent.batch_size:
                dqn_agent.train()
            
            # AIの手を棋譜に記録
            if current_game:
                current_game.add_move(new_board, move, 'white')
            
            return jsonify({
                "move": {"row": move[0], "col": move[1]},
                "evaluation": reward
            })
    
    # 既存のAIロジック
    # Placeholder for existing AI logic
    return jsonify({"move": {"row": 0, "col": 0}, "evaluation": 0})

@app.route('/game_end', methods=['POST'])
def game_end():
    data = request.json
    winner = data.get('winner')
    
    if current_game:
        current_game.set_result(winner)
        current_game.save()
        
        # 学習を実行
        try:
            train_from_game_record(current_game)
            learning_logger.log(f"Trained on game record, winner: {winner}")
        except Exception as e:
            learning_logger.log(f"Error training on game record: {str(e)}")
    
    return jsonify({"status": "success"})

def train_from_game_record(game_record):
    """棋譜データから学習を行う"""
    training_data = []
    
    for move_data in game_record.moves:
        board = move_data['board']
        move = move_data['move']
        color = move_data['color']
        
        # 状態の取得
        state = dqn_agent.get_state(board)
        
        # 報酬の計算
        reward = 1.0 if color == game_record.result else -1.0
        
        # 次の状態の取得
        next_board = copy.deepcopy(board)
        if isinstance(move, dict):  # moveがdict型の場合の対応
            move = (move['row'], move['col'])
        apply_move(next_board, move[0], move[1], color)
        next_state = dqn_agent.get_state(next_board)
        
        # 最後の手かどうか
        done = (move_data == game_record.moves[-1])
        
        # 経験を記憶
        dqn_agent.memory.push(state, move, reward, next_state, done)
    
    # バッチ学習の実行
    if len(dqn_agent.memory) >= dqn_agent.batch_size:
        dqn_agent.train()
        learning_logger.log(f"Batch training completed, memory size: {len(dqn_agent.memory)}")

    # 遺伝的進化の適用
    current_model = genetic_learning.population[0]
    current_model['games_played'] += 1
    
    # 勝利時のフィットネス更新
    if current_model['games_played'] >= 10:
        win_rate = game_record.result == playerColor
        current_model['fitness'] = (current_model['fitness'] * 0.9 + 
                                  (1.0 if win_rate else 0.0) * 0.1)
        
        # 一定数のゲーム後に進化を実行
        if sum(p['games_played'] for p in genetic_learning.population) >= 100:
            genetic_learning.evolve()
            learning_logger.log(f"Genetic evolution completed. Generation: {genetic_learning.generation}")

# 定期的なモデルの保存と更新
def periodic_model_update():
    while True:
        time.sleep(3600)  # 1時間ごと
        dqn_agent.target_net.load_state_dict(dqn_agent.policy_net.state_dict())
        dqn_agent.save_model('model/othello_dqn_model.pth')

# モデルの読み込み
try:
    dqn_agent.load_model('model/othello_dqn_model.pth')
    dqn_agent.target_net.load_state_dict(dqn_agent.policy_net.state_dict())
except:
    print("No existing model found. Starting with a new model.")

# バックグラウンドでモデル更新を実行
from threading import Thread
update_thread = Thread(target=periodic_model_update, daemon=True)
update_thread.start()

@app.route('/game_end', methods=['POST'])
def game_end():
    data = request.json
    winner = data.get('winner')
    
    # ゲーム履歴を学習データとして保存
    model.store_game_data(game_history, winner)
    game_history.clear()
    
    # 定期的に学習を実行
    model.train()
    
    return jsonify({"status": "success"})

def get_game_phase(board):
    empty_count = sum(row.count(None) for row in board)
    if empty_count > 45:  # 序盤
        return 'opening'
    elif empty_count < 15:  # 終盤
        return 'endgame'
    return 'middlegame'

def handle_opening(board, color):
    # 序盤戦略
    priority_moves = [
        (2, 2), (2, 5), (5, 2), (5, 5),  # 優先度の高いマス
        (2, 3), (2, 4), (5, 3), (5, 4),
        (3, 2), (3, 5), (4, 2), (4, 5)
    ]
    
    possible_moves = find_possible_moves(board, color)
    for move in priority_moves:
        if move in possible_moves:
            return move
            
    return parallel_search(board, color, 3)  # 優先手がない場合は通常探索

def handle_endgame(board, color):
    empty_count = sum(row.count(None) for row in board)
    if empty_count <= 8:
        # 完全読み
        return parallel_search(board, color, min(empty_count, 8))
    return parallel_search(board, color, 5)

def handle_middlegame(board, color):
    return parallel_negaScout(board, color, 4)

# タイムアウト時間をさらに短縮
AI_TIMEOUT = 3  # 5秒から3秒に短縮

def parallel_search(board, color, depth):
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 4)) as executor:  # プロセス数を制限
        moves = find_possible_moves(board, color)
        
        # 序盤の即応答
        empty_cells = sum(row.count(None) for row in board)
        if empty_cells > 50:
            if moves:
                return choice(moves)
            return None

        # 探索順序を最適化して早期終了を促進
        sorted_moves = sort_moves_by_priority(board, moves, color)
        best_move = sorted_moves[0]  # デフォルトの手を設定
        
        # 明らかに良い手があれば即座に返す
        for move in sorted_moves[:3]:  # 上位3手のみチェック
            if is_critical_move(board, move, color):
                return move

        futures = []
        for move in sorted_moves[:min(len(sorted_moves), 6)]:  # 上位6手のみ詳細評価
            new_board = copy.deepcopy(board)
            apply_move(new_board, move[0], move[1], color)
            future = executor.submit(
                alpha_beta_with_timeout,
                new_board,
                depth,
                float('-inf'),
                float('inf'),
                False,
                color
            )
            futures.append((move, future))
        
        best_score = float('-inf')
        for move, future in futures:
            try:
                score = future.result(timeout=AI_TIMEOUT * 0.8)
                if score > best_score:
                    best_score = score
                    best_move = move
                    if score > 800:  # 十分良い手が見つかれば早期終了
                        break
            except TimeoutError:
                continue

        return best_move

def is_critical_move(board, move, color):
    """重要な手かどうかを高速判定"""
    row, col = move
    # 角の場合
    if (row, col) in [(0,0), (0,7), (7,0), (7,7)]:
        return True
    
    # 相手の角取りを防ぐ手の場合
    opponent = 'black' if color == 'white' else 'white'
    if any(is_valid_move(board, corner[0], corner[1], opponent) 
           for corner in [(0,0), (0,7), (7,0), (7,7)]):
        return True
    
    return False

def sort_moves_by_priority(board, moves, color):
    """手の優先順位付けを最適化"""
    move_scores = []
    for move in moves:
        score = quick_move_evaluation(board, move, color)
        move_scores.append((move, score))
    
    return [move for move, _ in sorted(move_scores, key=lambda x: x[1], reverse=True)]

def quick_move_evaluation(board, move, color):
    """高速な手の評価"""
    row, col = move
    score = 0
    
    # 角の評価
    if (row, col) in [(0,0), (0,7), (7,0), (7,7)]:
        return 1000
    
    # 危険な手の評価
    if (row in [1,6] and col in [1,6]):
        return -500
    
    # エッジの評価
    if row in [0,7] or col in [0,7]:
        score += 50
    
    # 石を返せる数の評価
    new_board = copy.deepcopy(board)
    flipped = len(get_flippable_stones(board, row, col, color))
    score += flipped * 10
    
    return score

def get_flippable_stones(board, row, col, color):
    """返せる石を高速カウント"""
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    opponent = 'black' if color == 'white' else 'white'
    flippable_stones = []

    for dx, dy in directions:
        x, y = row + dx, col + dy
        stones_to_flip = []

        while 0 <= x < 8 and 0 <= y < 8 and board[x][y] == opponent:
            stones_to_flip.append((x, y))
            x += dx
            y += dy

        if 0 <= x < 8 and 0 <= y < 8 and board[x][y] == color:
            flippable_stones.extend(stones_to_flip)

    return flippable_stones

def parallel_negaScout(board, color, depth):
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        moves = find_possible_moves(board, color)
        
        if not moves:
            return None

        futures = []
        for move in moves:
            new_board = copy.deepcopy(board)
            apply_move(new_board, move[0], move[1], color)
            future = executor.submit(
                negaScout,
                new_board,
                depth - 1,
                float('-inf'),
                float('inf'),
                'black' if color == 'white' else 'white'
            )
            futures.append((move, future))

        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move, future in futures:
            try:
                score = -future.result(timeout=10)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            except TimeoutError:
                continue

        return best_move

def find_best_move(board, color):
    best_move = None
    best_score = float('-inf')
    for move in find_possible_moves(board, color):
        new_board = copy.deepcopy(board)
        apply_move(new_board, move[0], move[1], color)
        score = alpha_beta(new_board, 3, float('-inf'), float('inf'), False, color)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

def alpha_beta(board, depth, alpha, beta, maximizing_player, player):
    # ハッシュ値の計算
    board_hash = str(board)
    cache_entry = trans_table.get(board_hash)
    if cache_entry and cache_entry.get('depth', 0) >= depth:
        return cache_entry['score']
    
    if depth == 0 or not has_valid_move(board, 'black') and not has_valid_move(board, 'white'):
        score = evaluate_board(board, player)
        trans_table.set(board_hash, {'depth': depth, 'score': score})
        return score

    opponent = 'black' if player == 'white' else 'white'
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in find_possible_moves(board, player):
            new_board = copy.deepcopy(board)
            apply_move(new_board, move[0], move[1], player)
            eval = alpha_beta(new_board, depth - 1, alpha, beta, False, player)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in find_possible_moves(board, opponent):
            new_board = copy.deepcopy(board)
            apply_move(new_board, move[0], move[1], opponent)
            eval = alpha_beta(new_board, depth - 1, alpha, beta, True, player)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def negaScout(board, depth, alpha, beta, color):
    if depth == 0 or not has_valid_move(board, 'black') and not has_valid_move(board, 'white'):
        return evaluate_board(board, color)

    board_hash = str(board)
    cache_entry = trans_table.get(board_hash)
    if cache_entry and cache_entry.get('depth', 0) >= depth:
        return cache_entry['score']

    moves = find_possible_moves(board, color)
    if not moves:
        return -negaScout(board, depth - 1, -beta, -alpha, 'black' if color == 'white' else 'white')

    first_child = True
    max_score = float('-inf')
    
    for move in moves:
        new_board = copy.deepcopy(board)
        apply_move(new_board, move[0], move[1], color)
        
        if first_child:
            score = -negaScout(new_board, depth - 1, -beta, -alpha, 'black' if color == 'white' else 'white')
        else:
            # Null Window Search
            score = -negaScout(new_board, depth - 1, -alpha - 1, -alpha, 'black' if color == 'white' else 'white')
            if alpha < score < beta:
                # Re-search with full window
                score = -negaScout(new_board, depth - 1, -beta, -score, 'black' if color == 'white' else 'white')

        max_score = max(max_score, score)
        alpha = max(alpha, score)
        
        if alpha >= beta:
            break
            
        first_child = False

    trans_table.set(board_hash, {'depth': depth, 'score': max_score})
    return max_score

def alpha_beta_with_timeout(board, depth, alpha, beta, maximizing_player, player):
    start_time = time.time()
    
    def should_timeout():
        return time.time() - start_time > (AI_TIMEOUT * 0.8)
    
    def alpha_beta_inner(board, depth, alpha, beta, maximizing_player, player):
        if should_timeout():
            raise TimeoutError()
            
        if depth == 0 or not has_valid_move(board, 'black') and not has_valid_move(board, 'white'):
            return evaluate_board(board, player)

        board_hash = str(board)
        cache_entry = trans_table.get(board_hash)
        if cache_entry and cache_entry.get('depth', 0) >= depth:
            return cache_entry['score']

        opponent = 'black' if player == 'white' else 'white'
        best_score = float('-inf') if maximizing_player else float('inf')
        
        if maximizing_player:
            for move in find_possible_moves(board, player):
                new_board = copy.deepcopy(board)
                apply_move(new_board, move[0], move[1], player)
                score = alpha_beta_inner(new_board, depth - 1, alpha, beta, False, player)
                best_score = max(best_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
        else:
            for move in find_possible_moves(board, opponent):
                new_board = copy.deepcopy(board)
                apply_move(new_board, move[0], move[1], opponent)
                score = alpha_beta_inner(new_board, depth - 1, alpha, beta, True, player)
                best_score = min(best_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
        
        trans_table.set(board_hash, {'depth': depth, 'score': best_score})
        return best_score

    try:
        return alpha_beta_inner(board, depth, alpha, beta, maximizing_player, player)
    except TimeoutError:
        return evaluate_board(board, player)

def evaluate_board(board, player):
    # 評価関数の強化
    weights = np.array([
        [120, -20,  20,  5,  5,  20, -20, 120],
        [-20, -40,  -5, -5, -5,  -5, -40, -20],
        [ 20,  -5,  15,  3,  3,  15,  -5,  20],
        [  5,  -5,   3,  3,  3,   3,  -5,   5],
        [  5,  -5,   3,  3,  3,   3,  -5,   5],
        [ 20,  -5,  15,  3,  3,  15,  -5,  20],
        [-20, -40,  -5, -5, -5,  -5, -40, -20],
        [120, -20,  20,  5,  5,  20, -20, 120]
    ])
    
    opponent = 'black' if player == 'white' else 'white'
    score = 0
    
    # 盤面評価
    for i in range(8):
        for j in range(8):
            if board[i][j] == player:
                score += weights[i][j]
            elif board[i][j] == opponent:
                score -= weights[i][j]
    
    # 機動力の評価（より重視）
    player_moves = len(find_possible_moves(board, player))
    opponent_moves = len(find_possible_moves(board, opponent))
    score += (player_moves - opponent_moves) * 10
    
    # 終盤評価の強化
    empty_cells = sum(row.count(None) for row in board)
    if empty_cells <= 12:  # 終盤判定を12手に拡大
        player_stones = sum(row.count(player) for row in board)
        opponent_stones = sum(row.count(opponent) for row in board)
        score += (player_stones - opponent_stones) * (16 - empty_cells)  # 残り手数に応じて重み付け
    
    # 安定石の評価を追加
    score += evaluate_stable_stones(board, player) * 30
    
    return score

def evaluate_stable_stones(board, player):
    stable_count = 0
    corners = [(0,0), (0,7), (7,0), (7,7)]
    
    # 角の安定石評価
    for corner in corners:
        if board[corner[0]][corner[1]] == player:
            stable_count += 1
            # 角から伸びる安定石の評価
            stable_count += evaluate_stable_line_from_corner(board, corner[0], corner[1], player)
    
    return stable_count

def evaluate_stable_line_from_corner(board, row, col, player):
    stable_count = 0
    directions = [(0,1), (1,0), (1,1)] if row == 0 and col == 0 else \
                 [(0,-1), (1,0), (1,-1)] if row == 0 and col == 7 else \
                 [(-1,0), (0,1), (-1,1)] if row == 7 and col == 0 else \
                 [(-1,0), (0,-1), (-1,-1)]
    
    for dx, dy in directions:
        x, y = row + dx, col + dy
        while 0 <= x < 8 and 0 <= y < 8 and board[x][y] == player:
            stable_count += 1
            x += dx
            y += dy
    
    return stable_count

def find_possible_moves(board, color):
    moves = []
    for i in range(8):
        for j in range(8):
            if is_valid_move(board, i, j, color):
                moves.append((i, j))
    return moves

def is_valid_move(board, row, col, player):
    if board[row][col] is not None:
        return False
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    opponent = 'black' if player == 'white' else 'white'
    for dx, dy in directions:
        x, y = row + dx
        has_opponent = False
        while 0 <= x < 8 and 0 <= y < 8 and board[x][y] == opponent:
            x += dx
            y += dy
            has_opponent = True
        if has_opponent and 0 <= x < 8 and 0 <= y < 8 and board[x][y] == player:
            return True
    return False

def apply_move(board, row, col, player):
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    opponent = 'black' if player == 'white' else 'white'
    board[row][col] = player

    for dx, dy in directions:
        x = row + dx
        y = col + dy
        cells_to_flip = []

        while 0 <= x < 8 and 0 <= y < 8 and board[x][y] == opponent:
            cells_to_flip.append((x, y))
            x += dx
            y += dy

        if 0 <= x < 8 and 0 <= y < 8 and board[x][y] == player:
            for fx, fy in cells_to_flip:
                board[fx][fy] = player

def has_valid_move(board, player):
    for i in range(8):
        for j in range(8):
            if is_valid_move(board, i, j, player):
                return True
    return False

# 自己対戦用の新しいクラスと変数を追加
class SelfPlayManager:
    def __init__(self):
        self.game_buffer = deque(maxlen=1000)  # 直近1000ゲームを保持
        self.total_games = 0
        self.win_rates = {'black': 0, 'white': 0, 'draw': 0}

    def record_game(self, moves, winner):
        self.game_buffer.append({'moves': moves, 'winner': winner})
        self.total_games += 1
        self.win_rates[winner] = self.win_rates.get(winner, 0) + 1
        self._update_win_rates()

    def _update_win_rates(self):
        for key in self.win_rates:
            self.win_rates[key] = (self.win_rates[key] / self.total_games) * 100

    def get_random_game(self):
        if not self.game_buffer:
            return None
        return random.choice(self.game_buffer)

# グローバルなインスタンスを作成
self_play_manager = SelfPlayManager()

# 自己対戦による学習を行うエンドポイント
@app.route('/self_play_training', methods=['POST'])
def self_play_training():
    num_games = request.json.get('num_games', 10)
    results = []

    for _ in range(num_games):
        game_record = play_self_game()
        results.append(game_record)
        
        # モデルの学習
        if len(self_play_manager.game_buffer) >= 32:  # バッチサイズ分のデータが溜まったら学習
            train_from_self_play()

    return jsonify({
        "status": "success",
        "games_played": num_games,
        "win_rates": self_play_manager.win_rates
    })

def play_self_game():
    board = [[None for _ in range(8)] for _ in range(8)]
    board[3][3] = board[4][4] = 'white'
    board[3][4] = board[4][3] = 'black'
    current_color = 'black'
    moves_history = []
    
    while True:
        if not has_valid_move(board, 'black') and not has_valid_move(board, 'white'):
            break

        if has_valid_move(board, current_color):
            state = dqn_agent.get_state(board)
            valid_moves = find_possible_moves(board, current_color)
            move = dqn_agent.select_action(state, valid_moves)
            
            if move:
                moves_history.append({
                    'color': current_color,
                    'move': move,
                    'board': copy.deepcopy(board)
                })
                apply_move(board, move[0], move[1], current_color)

        current_color = 'white' if current_color == 'black' else 'black'

    # 勝者の判定
    black_count = sum(row.count('black') for row in board)
    white_count = sum(row.count('white') for row in board)
    winner = 'black' if black_count > white_count else 'white' if white_count > black_count else 'draw'
    
    # ゲーム記録を保存
    self_play_manager.record_game(moves_history, winner)
    
    return {
        'moves': moves_history,
        'winner': winner,
        'final_score': {'black': black_count, 'white': white_count}
    }

def train_from_self_play():
    game_data = self_play_manager.get_random_game()
    if not game_data:
        return

    for move_data in game_data['moves']:
        state = dqn_agent.get_state(move_data['board'])
        move = move_data['move']
        reward = 1 if move_data['color'] == game_data['winner'] else -1
        
        # 次の状態を取得
        next_board = copy.deepcopy(move_data['board'])
        apply_move(next_board, move[0], move[1], move_data['color'])
        next_state = dqn_agent.get_state(next_board)
        
        # 経験を記憶
        done = (game_data['moves'][-1] == move_data)
        dqn_agent.memory.push(state, move, reward, next_state, done)

    # バッチ学習の実行
    if len(dqn_agent.memory) >= dqn_agent.batch_size:
        dqn_agent.train()

# 定期的な自己対戦トレーニングを行うバックグラウンド処理
def periodic_self_play_training():
    while True:
        play_self_game()
        time.sleep(10)  # 10秒ごとに1ゲーム実行

class GeneticLearning:
    def __init__(self, population_size=10):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.elite_size = 2
        
        # 初期集団の生成
        for _ in range(population_size):
            model = OthelloDQN().to(dqn_agent.device)
            self.population.append({
                'model': model,
                'fitness': 0,
                'games_played': 0
            })

    def crossover(self, parent1, parent2):
        """モデルのクロスオーバー"""
        child = OthelloDQN().to(dqn_agent.device)
        
        # 重みのクロスオーバー
        for (name1, param1), (name2, param2) in zip(
            parent1.named_parameters(), 
            parent2.named_parameters()
        ):
            # ランダムな交差点を選択
            cross_point = torch.rand(param1.shape) < 0.5
            new_param = torch.where(cross_point, param1, param2)
            dict(child.named_parameters())[name1].data.copy_(new_param)
            
        return child

    def mutate(self, model, mutation_rate=0.1):
        """モデルの突然変異"""
        for param in model.parameters():
            if torch.rand(1) < mutation_rate:
                noise = torch.randn(param.shape) * 0.1
                param.data += noise.to(param.device)

    def evolve(self):
        """世代を進化させる"""
        # 適応度でソート
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # エリート選択
        new_population = self.population[:self.elite_size]
        
        # 残りの個体を生成
        while len(new_population) < self.population_size:
            # トーナメント選択
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()
            
            # クロスオーバー
            child_model = self.crossover(parent1['model'], parent2['model'])
            
            # 突然変異
            self.mutate(child_model)
            
            new_population.append({
                'model': child_model,
                'fitness': 0,
                'games_played': 0
            })
        
        self.population = new_population
        self.generation += 1

    def tournament_select(self, tournament_size=3):
        """トーナメント選択"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x['fitness'])

# グローバルなインスタンスを作成
genetic_learning = GeneticLearning()

# 定期的な遺伝的進化の実行を追加
def periodic_genetic_evolution():
    while True:
        time.sleep(3600)  # 1時間ごと
        genetic_learning.evolve()
        dqn_agent.policy_net = genetic_learning.population[0]['model']
        dqn_agent.save_model('model/othello_dqn_model.pth')
        learning_logger.log(f"Periodic genetic evolution completed. Generation: {genetic_learning.generation}")

# 既存のスレッド起動部分を修正
if __name__ == '__main__':
    mp.freeze_support()
    trans_table = TranspositionTable()
    
    # モデル更新用スレッド
    update_thread = Thread(target=periodic_model_update, daemon=True)
    update_thread.start()
    
    # 自己対戦用スレッド
    self_play_thread = Thread(target=periodic_self_play_training, daemon=True)
    self_play_thread.start()
    
    # 遺伝的進化用スレッド
    evolution_thread = Thread(target=periodic_genetic_evolution, daemon=True)
    evolution_thread.start()
    
    app.run(debug=True)
