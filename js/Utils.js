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

// ビットボード操作のユーティリティを追加
const BitBoard = {
    FULL_BOARD: 0xFFFFFFFFFFFFFFFFn,
    EMPTY: 0n,
    
    // 初期配置のビットボード
    INITIAL_BLACK: 0x0000000810000000n,
    INITIAL_WHITE: 0x0000001008000000n,
    
    // 方向定義
    DIRECTIONS: [
        8n,   // 下
        -8n,  // 上
        1n,   // 右
        -1n,  // 左
        7n,   // 右下
        -7n,  // 左上
        9n,   // 左下
        -9n   // 右上
    ],
    
    // マスクテーブル（事前計算）
    HORIZONTAL_MASK: Array(64).fill(0n),
    VERTICAL_MASK: Array(64).fill(0n),
    DIAGONAL_MASK: Array(64).fill(0n),
    
    // マスクテーブルの初期化
    initializeMasks() {
        for (let pos = 0; pos < 64; pos++) {
            const row = Math.floor(pos / 8);
            const col = pos % 8;
            
            // 水平マスク
            this.HORIZONTAL_MASK[pos] = 0xFFn << BigInt(row * 8);
            
            // 垂直マスク
            this.VERTICAL_MASK[pos] = 0x0101010101010101n << BigInt(col);
            
            // 対角マスクは複雑なので省略（実際の実装では必要）
        }
    },

    // 合法手を高速に生成
    getLegalMoves(black, white) {
        const empty = ~(black | white);
        let moves = 0n;
        
        for (const dir of this.DIRECTIONS) {
            const shift = dir > 0n ? black << dir : black >> -dir;
            let candidates = shift & white;
            
            for (let i = 0; i < 5; i++) {
                candidates |= (candidates << dir) & white;
            }
            
            moves |= (candidates << dir) & empty;
        }
        
        return moves;
    },

    // 石を打って反転処理を実行
    makeMove(black, white, pos, isBlack) {
        const move = 1n << BigInt(pos);
        let flipped = 0n;
        
        for (const dir of this.DIRECTIONS) {
            let flip = 0n;
            let current = move;
            
            while (true) {
                current = dir > 0n ? current << dir : current >> -dir;
                if ((current & (isBlack ? white : black)) === 0n) break;
                flip |= current;
            }
            
            if (current & (isBlack ? black : white)) {
                flipped |= flip;
            }
        }
        
        if (isBlack) {
            black |= move | flipped;
            white &= ~flipped;
        } else {
            white |= move | flipped;
            black &= ~flipped;
        }
        
        return { black, white };
    },

    // 配列表現とビットボード表現の相互変換
    fromArray(board) {
        let black = 0n;
        let white = 0n;
        
        for (let i = 0; i < 64; i++) {
            const row = Math.floor(i / 8);
            const col = i % 8;
            if (board[row][col] === 'black') {
                black |= 1n << BigInt(i);
            } else if (board[row][col] === 'white') {
                white |= 1n << BigInt(i);
            }
        }
        
        return { black, white };
    },

    toArray(black, white) {
        const board = Array(8).fill().map(() => Array(8).fill(null));
        
        for (let i = 0; i < 64; i++) {
            const row = Math.floor(i / 8);
            const col = i % 8;
            const pos = 1n << BigInt(i);
            
            if (black & pos) {
                board[row][col] = 'black';
            } else if (white & pos) {
                board[row][col] = 'white';
            }
        }
        
        return board;
    },

    // 石数をカウント（ビットカウント）
    countBits(bitboard) {
        let count = 0n;
        let bb = bitboard;
        while (bb) {
            count++;
            bb &= bb - 1n;
        }
        return Number(count);
    }
};

// マスクテーブルの初期化を実行
BitBoard.initializeMasks();

// ビットボードをエクスポート
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BitBoard };
}
