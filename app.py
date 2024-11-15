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

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["OPTIONS", "POST"],
        "allow_headers": ["Content-Type"]
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

# AI難易度の定数を追加
AI_DIFFICULTY = {
    'easy': {
        'depth': 2,
        'randomness': 0.3,  # 30%の確率でランダムな手を選択
        'eval_weight': 0.7  # 評価関数の重みを70%に抑える
    },
    'normal': {
        'depth': 4,
        'randomness': 0.1,
        'eval_weight': 1.0
    },
    'hard': {
        'depth': 5,
        'randomness': 0.0,
        'eval_weight': 1.0
    },
    'extreme': {
        'depth': 6,
        'randomness': 0.0,
        'eval_weight': 1.0
    }
}

@app.route('/ai_move', methods=['POST'])
def ai_move():
    data = request.json
    board = data.get('board')
    color = data.get('color', 'white')
    difficulty = data.get('difficulty', 'normal')  # デフォルトは'normal'
    game_phase = get_game_phase(board)
    
    # 難易度設定を取得
    ai_settings = AI_DIFFICULTY[difficulty]
    
    # 難易度別の処理
    if difficulty == 'easy':
        return handle_easy_mode(board, color, ai_settings)
    elif difficulty == 'normal':
        return handle_normal_mode(board, color, ai_settings)
    elif difficulty == 'hard':
        return handle_hard_mode(board, color, ai_settings)
    elif difficulty == 'extreme':
        return handle_extreme_mode(board, color, ai_settings)
    
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

def handle_easy_mode(board, color, settings):
    """かんたんモードのAI処理"""
    moves = find_possible_moves(board, color)
    if not moves:
        return jsonify({"move": None, "evaluation": 0})
    
    # ランダムな手を選ぶ確率
    if random.random() < settings['randomness']:
        move = random.choice(moves)
        return jsonify({
            "move": {"row": move[0], "col": move[1]},
            "evaluation": 0
        })
    
    # 単純な評価関数で手を選択
    best_move = None
    best_score = float('-inf')
    
    for move in moves:
        # 角は避ける（わざと）
        if move in [(0,0), (0,7), (7,0), (7,7)]:
            continue
            
        new_board = copy.deepcopy(board)
        apply_move(new_board, move[0], move[1], color)
        # 浅い探索深さと単純な評価
        score = simple_evaluation(new_board, color) * settings['eval_weight']
        
        if score > best_score:
            best_score = score
            best_move = move
    
    # 有効な手が見つからない場合はランダムに選択
    if best_move is None and moves:
        best_move = random.choice(moves)
    
    return jsonify({
        "move": {"row": best_move[0], "col": best_move[1]} if best_move else None,
        "evaluation": best_score
    })

def handle_normal_mode(board, color, settings):
    """ふつうモードのAI処理"""
    moves = find_possible_moves(board, color)
    if not moves:
        return jsonify({"move": None, "evaluation": 0})

    # 時々ランダムな手を選択（10%の確率）
    if random.random() < settings['randomness']:
        move = random.choice(moves)
        return jsonify({
            "move": {"row": move[0], "col": move[1]},
            "evaluation": 0
        })

    # 通常の探索（深さ4）
    best_moves = []
    best_score = float('-inf')
    
    for move in moves:
        new_board = copy.deepcopy(board)
        apply_move(new_board, move[0], move[1], color)
        # 適度な深さでの探索
        score = alpha_beta(new_board, settings['depth'], 
                         float('-inf'), float('inf'), 
                         False, color) * settings['eval_weight']
        
        # スコアが近い手も記録（ベスト手の90%以上のスコアの手）
        if score > best_score:
            best_moves = [move]
            best_score = score
        elif score >= best_score * 0.9:
            best_moves.append(move)

    # 複数の善手からランダムに選択（より人間らしい選択に）
    best_move = random.choice(best_moves)
    
    return jsonify({
        "move": {"row": best_move[0], "col": best_move[1]},
        "evaluation": best_score
    })

def handle_hard_mode(board, color, settings):
    """つよいモードのAI処理"""
    moves = find_possible_moves(board, color)
    if not moves:
        return jsonify({"move": None, "evaluation": 0})

    empty_count = sum(row.count(None) for row in board)
    depth = settings['depth']
    
    # 終盤では探索深さを増やす
    if empty_count <= 12:
        depth = min(empty_count, 8)
    
    best_move = None
    best_score = float('-inf')
    
    # 並列探索を使用
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 4)) as executor:
        futures = []
        
        for move in moves:
            new_board = copy.deepcopy(board)
            apply_move(new_board, move[0], move[1], color)
            future = executor.submit(
                negaScout,  # より高度なアルゴリズムを使用
                new_board,
                depth,
                float('-inf'),
                float('inf'),
                'black' if color == 'white' else 'white'
            )
            futures.append((move, future))
        
        for move, future in futures:
            try:
                score = -future.result(timeout=AI_TIMEOUT)
                if score > best_score:
                    best_score = score
                    best_move = move
            except TimeoutError:
                continue

    # 角が取れる場合は必ず取る
    corner_moves = [(0,0), (0,7), (7,0), (7,7)]
    for corner in corner_moves:
        if corner in moves:
            best_move = corner
            break

    return jsonify({
        "move": {"row": best_move[0], "col": best_move[1]} if best_move else None,
        "evaluation": best_score
    })

def handle_extreme_mode(board, color, settings):
    """ゲキムズモードのAI処理"""
    moves = find_possible_moves(board, color)
    if not moves:
        return jsonify({"move": None, "evaluation": 0})

    empty_count = sum(row.count(None) for row in board)
    
    # 終盤の完全読み（残り16手以下）
    if empty_count <= 16:
        move = find_perfect_move(board, color, empty_count)
        if move:
            return jsonify({
                "move": {"row": move[0], "col": move[1]},
                "evaluation": 10000  # 必勝手の場合
            })
    
    # 機械学習による手の予測を試みる
    ml_move = model.predict_move(board, moves)
    if ml_move and empty_count > 45:  # 序盤のみML使用
        return jsonify({
            "move": {"row": ml_move[0], "col": ml_move[1]},
            "evaluation": 500
        })
    
    # 通常の探索（より深く）
    best_move = None
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    # 並列探索with反復深化
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 4)) as executor:
        max_depth = 8 if empty_count <= 20 else 6  # 終盤に近いほど深く読む
        futures = []
        
        # 探索順序の最適化
        sorted_moves = sort_moves_by_priority(board, moves, color)
        
        for move in sorted_moves:
            new_board = copy.deepcopy(board)
            apply_move(new_board, move[0], move[1], color)
            
            future = executor.submit(
                iterative_deepening_search,  # 反復深化探索
                new_board,
                max_depth,
                alpha,
                beta,
                color,
                empty_count
            )
            futures.append((move, future))
        
        for move, future in futures:
            try:
                score = future.result(timeout=AI_TIMEOUT)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            except TimeoutError:
                continue

    if best_move is None and moves:
        best_move = sorted_moves[0]  # フォールバック

    return jsonify({
        "move": {"row": best_move[0], "col": best_move[1]},
        "evaluation": best_score
    })

def simple_evaluation(board, color):
    """単純な評価関数（かんたんモード用）"""
    opponent = 'black' if color == 'white' else 'white'
    score = 0
    
    # 単純な石数の差だけで評価
    for row in board:
        for cell in row:
            if cell == color:
                score += 1
            elif cell == opponent:
                score -= 1
    
    return score

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
    # 探索開始時刻を記録
    start_time = time.time()
    
    def should_timeout():
        return time.time() - start_time > (AI_TIMEOUT * 0.8)  # 80%のタイムアウト閾値
    
    def alpha_beta_inner(board, depth, alpha, beta, maximizing_player, player):
        if should_timeout():
            raise TimeoutError()
            
        if depth == 0 or not has_valid_move(board, 'black') and not has_valid_move(board, 'white'):
            return evaluate_board(board, player)

        # キャッシュチェック
        board_hash = str(board)
        cache_entry = trans_table.get(board_hash)
        if cache_entry and cache_entry.get('depth', 0) >= depth:
            return cache_entry['score']

        # ...existing alpha_beta code...
        
        trans_table.set(board_hash, {'depth': depth, 'score': best_score})
        return best_score

    try:
        return alpha_beta_inner(board, depth, alpha, beta, maximizing_player, player)
    except TimeoutError:
        return evaluate_board(board, player)  # タイムアウト時は現在の評価値を返す

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
    
    # つよいモード用に評価関数を強化
    if game_phase == 'endgame':
        # 終盤では石数をより重視
        player_stones = sum(row.count(player) for row in board)
        opponent_stones = sum(row.count(opponent) for row in board)
        score += (player_stones - opponent_stones) * 20
        
    # 角の支配をより重視
    corners = [(0,0), (0,7), (7,0), (7,7)]
    for corner in corners:
        if board[corner[0]][corner[1]] == player:
            score += 200  # 角の重みを増加
    
    # ゲキムズモード用の追加評価
    if game_phase == 'endgame':
        # 終盤では石数をさらに重視
        player_stones = sum(row.count(player) for row in board)
        opponent_stones = sum(row.count(opponent) for row in board)
        score += (player_stones - opponent_stones) * 30
        
        # 勝敗が決まっている場合は最大/最小スコア
        if empty_cells == 0:
            if player_stones > opponent_stones:
                return float('inf')
            elif player_stones < opponent_stones:
                return float('-inf')
    
    # パリティ（奇数・偶数）の考慮
    empty_regions = analyze_empty_regions(board)
    for region in empty_regions:
        if len(region) % 2 == 1:  # 奇数の空きマス領域
            if has_more_access(board, region, player):
                score += 15 * len(region)
            else:
                score -= 15 * len(region)
    
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

def analyze_empty_regions(board):
    """空きマスの連結領域を解析"""
    regions = []
    visited = set()
    
    for i in range(8):
        for j in range(8):
            if board[i][j] is None and (i,j) not in visited:
                region = find_connected_empty_cells(board, i, j, visited)
                regions.append(region)
    
    return regions

def find_connected_empty_cells(board, row, col, visited):
    """連結している空きマスを探索"""
    region = set()
    stack = [(row, col)]
    
    while stack:
        r, c = stack.pop()
        if (r,c) in visited or board[r][c] is not None:
            continue
            
        visited.add((r,c))
        region.add((r,c))
        
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] is None:
                    stack.append((nr,nc))
    
    return region

def has_more_access(board, region, player):
    """領域へのアクセス可能性を評価"""
    player_access = 0
    opponent_access = 0
    
    for r, c in region:
        if any(is_valid_move(board, r, c, player) for (r,c) in region):
            player_access += 1
        if any(is_valid_move(board, r, c, 'black' if player == 'white' else 'white') for (r,c) in region):
            opponent_access += 1
    
    return player_access > opponent_access

def iterative_deepening_search(board, max_depth, alpha, beta, color, empty_count):
    """反復深化探索"""
    best_score = float('-inf')
    
    # 深さを徐々に増やしながら探索
    for depth in range(4, max_depth + 1):
        try:
            score = negaScout(board, depth, alpha, beta, color)
            best_score = score
        except TimeoutError:
            break
    
    return best_score

def find_perfect_move(board, color, empty_count):
    """完全読みによる必勝手の発見"""
    moves = find_possible_moves(board, color)
    best_move = None
    best_score = float('-inf')
    
    for move in moves:
        new_board = copy.deepcopy(board)
        apply_move(new_board, move[0], move[1], color)
        score = -perfect_search(new_board, color, -float('inf'), float('inf'), empty_count - 1)
        
        if score > best_score:
            best_score = score
            best_move = move
            if score == float('inf'):  # 必勝手を見つけた
                break
    
    return best_move

def perfect_search(board, color, alpha, beta, depth):
    """完全読みの探索関数"""
    if depth == 0:
        counts = count_stones(board)
        return counts[color] - counts['white' if color == 'black' else 'black']
    
    moves = find_possible_moves(board, color)
    if not moves:
        if not has_valid_move(board, 'black' if color == 'white' else 'white'):
            counts = count_stones(board)
            return counts[color] - counts['white' if color == 'black' else 'black']
        return -perfect_search(board, 'black' if color == 'white' else 'white', -beta, -alpha, depth)
    
    for move in moves:
        new_board = copy.deepcopy(board)
        apply_move(new_board, move[0], move[1], color)
        score = -perfect_search(new_board, 'black' if color == 'white' else 'white', -beta, -alpha, depth - 1)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    
    return alpha

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
