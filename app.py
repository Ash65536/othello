from flask import Flask, request, jsonify
from random import choice
import copy

app = Flask(__name__)

@app.route('/ai_move', methods=['POST'])
def ai_move():
    board = request.json.get('board')
    move = find_best_move(board, 'white')  # アルファベータ法で最適な手を計算
    if move is None:
        return jsonify({"row": -1, "col": -1})  # パスの場合
    return jsonify({"row": move[0], "col": move[1]})

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
    if depth == 0 or not has_valid_move(board, 'black') and not has_valid_move(board, 'white'):
        return evaluate_board(board, player)
    
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
    score = 0
    for row in board:
        for cell in row:
            if cell == player:
                score += 1
            elif cell is not None:
                score -= 1
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
    app.run(debug=True)
