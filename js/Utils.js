function createBoard() {
    const board = [];
    for (let i = 0; i < 8; i++) {
        const row = [];
        for (let j = 0; j < 8; j++) {
            row.push(null);
        }
        board.push(row);
    }
    return board;
}

function isValidMove(board, row, col, player) {
    if (board[row][col] !== null) return false;
    const directions = [
        [-1, 0], [1, 0], [0, -1], [0, 1],
        [-1, -1], [-1, 1], [1, -1], [1, 1]
    ];
    const opponent = player === 'black' ? 'white' : 'black';
    let valid = false;

    for (const [dx, dy] of directions) {
        let x = row + dx, y = col + dy;
        let hasOpponent = false;

        while (x >= 0 && x < 8 && y >= 0 && y < 8 && board[x][y] === opponent) {
            x += dx;
            y += dy;
            hasOpponent = true;
        }

        if (hasOpponent && x >= 0 && x < 8 && y >= 0 && y < 8 && board[x][y] === player) {
            valid = true;
            break;
        }
    }

    return valid;
}

function applyMove(board, row, col, player) {
    const directions = [
        [-1, 0], [1, 0], [0, -1], [0, 1],
        [-1, -1], [-1, 1], [1, -1], [1, 1]
    ];
    const opponent = player === 'black' ? 'white' : 'black';
    const flippedStones = [];
    board[row][col] = player;

    for (const [dx, dy] of directions) {
        let x = row + dx, y = col + dy;
        let cellsToFlip = [];

        while (x >= 0 && x < 8 && y >= 0 && y < 8 && board[x][y] === opponent) {
            cellsToFlip.push([x, y]);
            x += dx;
            y += dy;
        }

        if (x >= 0 && x < 8 && y >= 0 && y < 8 && board[x][y] === player) {
            for (const [fx, fy] of cellsToFlip) {
                board[fx][fy] = player;
                flippedStones.push([fx, fy]);
            }
        }
    }
    return flippedStones;
}

function hasValidMove(board, player) {
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            if (isValidMove(board, i, j, player)) {
                return true;
            }
        }
    }
    return false;
}

function countStones(board) {
    let blackCount = 0;
    let whiteCount = 0;
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            if (board[i][j] === 'black') blackCount++;
            if (board[i][j] === 'white') whiteCount++;
        }
    }
    return { black: blackCount, white: whiteCount };
}

function findPossibleMoves(board, player) {
    const moves = [];
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            if (isValidMove(board, i, j, player)) {
                moves.push([i, j]);
            }
        }
    }
    return moves;
}

function isStableStone(board, row, col, player) {
    // 方向ごとに安定性を確認
    const directions = [
        [0, 1], [1, 0], [1, 1], [1, -1]  // 水平、垂直、斜め
    ];
    
    for (const [dx, dy] of directions) {
        let stable1 = false;
        let stable2 = false;
        
        // 正方向
        let x = row + dx;
        let y = col + dy;
        while (x >= 0 && x < 8 && y >= 0 && y < 8) {
            if (board[x][y] !== player) break;
            if (x === 0 || x === 7 || y === 0 || y === 7) {
                stable1 = true;
                break;
            }
            x += dx;
            y += dy;
        }
        
        // 逆方向
        x = row - dx;
        y = col - dy;
        while (x >= 0 && x < 8 && y >= 0 && y < 8) {
            if (board[x][y] !== player) break;
            if (x === 0 || x === 7 || y === 0 || y === 7) {
                stable2 = true;
                break;
            }
            x -= dx;
            y -= dy;
        }
        
        if (!stable1 && !stable2) return false;
    }
    return true;
}

function evaluateBoard(board, player) {
    const weights = [
        [100, -20, 10, 5, 5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [10, -2, 1, 1, 1, 1, -2, 10],
        [5, -2, 1, 0, 0, 1, -2, 5],
        [5, -2, 1, 0, 0, 1, -2, 5],
        [10, -2, 1, 1, 1, 1, -2, 10],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [100, -20, 10, 5, 5, 10, -20, 100]
    ];

    const opponent = player === 'black' ? 'white' : 'black';
    let score = 0;

    // 位置の重み付け評価
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            if (board[i][j] === player) {
                score += weights[i][j];
                // 安定石の評価
                if (isStableStone(board, i, j, player)) {
                    score += 30;
                }
            } else if (board[i][j] === opponent) {
                score -= weights[i][j];
                if (isStableStone(board, i, j, opponent)) {
                    score -= 30;
                }
            }
        }
    }

    // 移動可能手数の評価
    const playerMoves = findPossibleMoves(board, player).length;
    const opponentMoves = findPossibleMoves(board, opponent).length;
    score += (playerMoves - opponentMoves) * 5;

    return score;
}
