from flask import Flask, request, jsonify
from random import choice
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Add this line after creating the Flask app

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
    
    # 状況に応じた戦略選択
    if game_phase == 'opening':
        move = handle_opening(board, color)
    elif game_phase == 'endgame':
        move = handle_endgame(board, color)
    else:
        move = handle_middlegame(board, color)
        
    if move is None:
        return jsonify({"move": None})
    
    return jsonify({
        "move": {"row": move[0], "col": move[1]},
        "evaluation": evaluate_board(board, color)
    })

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
    return parallel_search(board, color, 4)

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

def evaluate_board(board, player):
    # 評価関数を強化
    weights = np.array([
        [100, -20,  10,   5,   5,  10, -20, 100],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [ 10,  -2,   1,   1,   1,   1,  -2,  10],
        [  5,  -2,   1,   0,   0,   1,  -2,   5],
        [  5,  -2,   1,   0,   0,   1,  -2,   5],
        [ 10,  -2,   1,   1,   1,   1,  -2,  10],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [100, -20,  10,   5,   5,  10, -20, 100]
    ])
    
    opponent = 'black' if player == 'white' else 'white'
    score = 0
    
    # 盤面の評価
    for i in range(8):
        for j in range(8):
            if board[i][j] == player:
                score += weights[i][j]
            elif board[i][j] == opponent:
                score -= weights[i][j]
    
    # 機動力の評価
    player_moves = len(find_possible_moves(board, player))
    opponent_moves = len(find_possible_moves(board, opponent))
    score += (player_moves - opponent_moves) * 5
    
    # 終盤評価の追加
    empty_cells = sum(row.count(None) for row in board)
    if empty_cells <= 8:
        player_stones = sum(row.count(player) for row in board)
        opponent_stones = sum(row.count(opponent) for row in board)
        score += (player_stones - opponent_stones) * 10
    
    return score

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
