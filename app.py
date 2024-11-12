from flask import Flask, request, jsonify
from random import choice
import random
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from flask_cors import CORS
from model import OthelloModel

app = Flask(__name__)
CORS(app)  # Add this line after creating the Flask app

model = OthelloModel()
game_history = []

# トランスポジションテーブル（プロセス間で共有）
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

@app.route('/ai_move', methods=['POST'])
def ai_move():
    data = request.json
    board = data.get('board')
    color = data.get('color', 'white')
    game_phase = get_game_phase(board)
    
    # ML予測を試みる
    possible_moves = find_possible_moves(board, color)
    ml_move = model.predict_move(board, possible_moves)
    
    if ml_move and random.random() < 0.7:  # 70%の確率でML予測を使用
        move = ml_move
    else:
        # 通常の探索を実行
        if game_phase == 'opening':
            move = handle_opening(board, color)
        elif game_phase == 'endgame':
            move = handle_endgame(board, color)
        else:
            move = handle_middlegame(board, color)
    
    if move:
        game_history.append(({
            'board': board,
            'current_player': color
        }, move))
    
    return jsonify({
        "move": {"row": move[0], "col": move[1]} if move else None,
        "evaluation": evaluate_board(board, color)
    })

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

def parallel_search(board, color, depth):
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        moves = find_possible_moves(board, color)
        
        # 序盤はランダム性を持たせる
        empty_cells = sum(row.count(None) for row in board)
        if empty_cells > 50:  # 序盤
            if moves:
                return choice(moves)
            return None

        futures = []
        for move in moves:
            new_board = copy.deepcopy(board)
            apply_move(new_board, move[0], move[1], color)
            future = executor.submit(
                alpha_beta,
                new_board,
                depth,
                float('-inf'),
                float('inf'),
                False,
                color
            )
            futures.append((move, future))
        
        if not futures:
            return None

        best_move = None
        best_score = float('-inf')
        
        for move, future in futures:
            try:
                score = future.result(timeout=10)  # 10秒タイムアウト
                if score > best_score:
                    best_score = score
                    best_move = move
            except TimeoutError:
                continue

        return best_move

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
        x, y = row + dx, col + dy
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

if __name__ == '__main__':
    mp.freeze_support()  # ここを追加
    trans_table = TranspositionTable()
    app.run(debug=True)
