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
            }
        }
    }
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
