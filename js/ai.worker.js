let baseUrl = 'http://localhost:5000';

self.onmessage = async function(e) {
    const { board, color } = e.data;
    
    try {
        // CORSヘッダーを追加
        const response = await fetch(`${baseUrl}/ai_move`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            mode: 'cors', // CORSモードを明示的に指定
            body: JSON.stringify({ 
                board: board,
                color: color
            })
        });

        if (!response.ok) {
            throw new Error('Server response was not ok');
        }

        const data = await response.json();
        if (data.move) {
            self.postMessage({ 
                move: [data.move.row, data.move.col],
                evaluation: data.evaluation
            });
        } else {
            // フォールバック処理
            const move = findLocalBestMove(board, color);
            self.postMessage({ 
                move: move,
                evaluation: 0
            });
        }
    } catch (error) {
        console.error('AI Error:', error);
        // エラー時のフォールバック処理
        const move = findLocalBestMove(board, color);
        self.postMessage({ 
            move: move,
            evaluation: 0 
        });
    }
};

function findLocalBestMove(board, color) {
    const moves = findPossibleMoves(board, color);
    if (moves.length === 0) return null;

    // 角を優先
    for (const [cornerX, cornerY] of [[0,0], [0,7], [7,0], [7,7]]) {
        const cornerMove = moves.find(([x, y]) => x === cornerX && y === cornerY);
        if (cornerMove) return cornerMove;
    }

    // 危険な手を避ける
    const safeMoves = moves.filter(([x, y]) => {
        if ((x === 0 || x === 7) && (y === 1 || y === 6)) return false;
        if ((x === 1 || x === 6) && (y === 0 || y === 7)) return false;
        if ((x === 1 || x === 6) && (y === 1 || y === 6)) return false;
        return true;
    });

    if (safeMoves.length > 0) {
        return safeMoves[0];
    }

    return moves[0];
}

// Fallback utilities in case server fails
function findPossibleMoves(board, color) {
    const moves = [];
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            if (isValidMove(board, i, j, color)) {
                moves.push([i, j]);
            }
        }
    }
    return moves;
}

function isValidMove(board, row, col, color) {
    if (board[row][col] !== null) return false;
    const directions = [
        [-1, 0], [1, 0], [0, -1], [0, 1],
        [-1, -1], [-1, 1], [1, -1], [1, 1]
    ];
    const opponent = color === 'black' ? 'white' : 'black';
    
    for (const [dx, dy] of directions) {
        let x = row + dx, y = col + dy;
        let hasOpponent = false;
        
        while (x >= 0 && x < 8 && y >= 0 && y < 8 && board[x][y] === opponent) {
            x += dx;
            y += dy;
            hasOpponent = true;
        }
        
        if (hasOpponent && x >= 0 && x < 8 && y >= 0 && y < 8 && board[x][y] === color) {
            return true;
        }
    }
    return false;
}

// Zobrist乱数テーブルの初期化
const ZOBRIST_TABLE = {
    black: Array(64).fill(0).map(() => Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)),
    white: Array(64).fill(0).map(() => Math.floor(Math.random() * Number.MAX_SAFE_INTEGER))
};

// トランスポジションテーブル
const transpositionTable = new Map();

// ボードのハッシュ値を計算
function computeZobristHash(board) {
    let hash = 0;
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            if (board[i][j]) {
                const index = i * 8 + j;
                hash ^= ZOBRIST_TABLE[board[i][j]][index];
            }
        }
    }
    return hash;
}

function dynamicDepth(board) {
    const remainingMoves = board.flat().filter(cell => cell === null).length;
    if (remainingMoves <= 8) {
        return 8;      // 終盤（残り8手以下）
    } else if (remainingMoves <= 16) {
        return 6;      // 終盤近く（残り16手以下）
    } else if (remainingMoves <= 32) {
        return 5;      // 中盤（残り32手以下）
    }
    return 4;          // 序盤
}

// メモリ使用量を監視し、必要に応じてテーブルをクリア
function cleanTranspositionTable() {
    if (transpositionTable.size > 1000000) {  // 100万エントリーを超えたらクリア
        transpositionTable.clear();
    }
}

// 評価用パターンデータベース（角周辺のパターン）
const CORNER_PATTERNS = {
    // 角が自分の石の場合の理想的なパターン
    'own_corner': [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0]
    ],
    // その他のパターンは実際の実装時に追加
};

function evaluateBoard(board, player) {
    // ...existing code...

    // 終盤に近づくほど石数の差を重視
    const remainingMoves = board.flat().filter(cell => cell === null).length;
    if (remainingMoves <= 16) {
        const counts = countStones(board);
        const stoneDiff = (player === 'black' ? counts.black - counts.white : counts.white - counts.black);
        score += stoneDiff * (32 - remainingMoves);
    }

    // パターン評価を追加
    score += evaluateCornerPatterns(board, player) * 50;
    
    // 角の支配状況をより重視
    score += evaluateCornerControl(board, player) * 100;

    return score;
}

function evaluateCornerPatterns(board, player) {
    let score = 0;
    const corners = [[0,0], [0,7], [7,0], [7,7]];
    
    for (const [cornerX, cornerY] of corners) {
        if (board[cornerX][cornerY] === player) {
            score += evaluateCornerArea(board, cornerX, cornerY, player);
        }
    }
    return score;
}

function evaluateCornerArea(board, cornerX, cornerY, player) {
    let score = 0;
    const xDir = cornerX === 0 ? 1 : -1;
    const yDir = cornerY === 0 ? 1 : -1;
    
    // 角周辺の3x3エリアを評価
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            const x = cornerX + (i * xDir);
            const y = cornerY + (j * yDir);
            if (x >= 0 && x < 8 && y >= 0 && y < 8) {
                if (board[x][y] === player) {
                    score += (3 - Math.max(i, j)); // 角に近いほど高得点
                }
            }
        }
    }
    return score;
}

function evaluateCornerControl(board, player) {
    const corners = [[0,0], [0,7], [7,0], [7,7]];
    let score = 0;
    
    for (const [x, y] of corners) {
        if (board[x][y] === player) {
            score += 25;
        } else if (board[x][y] === null) {
            // 角が空いている場合、その角を取れる位置にあるかを評価
            if (isValidMove(board, x, y, player)) {
                score += 15;
            }
        }
    }
    return score;
}

// findBestMove関数内でcleanTranspositionTableを呼び出す
function findBestMove(board, color) {
    cleanTranspositionTable();  // テーブルのクリーンアップ
    const depth = dynamicDepth(board);
    const moves = findPossibleMoves(board, color);
    const remainingMoves = board.flat().filter(cell => cell === null).length;
    
    // 完全読みが可能な終盤では必勝手を探索
    if (remainingMoves <= 10) {
        const perfectMove = findPerfectMove(board, color);
        if (perfectMove) return perfectMove;
    }
    
    // 序盤はランダム選択
    if (remainingMoves > 50) {
        return moves[Math.floor(Math.random() * moves.length)];
    }
    
    let bestMove = null;
    let bestScore = -Infinity;
    let alpha = -Infinity;
    let beta = Infinity;
    
    for (const move of moves) {
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        const score = -negaScout(newBoard, depth, -beta, -alpha, color === 'black' ? 'white' : 'black');
        
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
        alpha = Math.max(alpha, score);
    }
    
    return bestMove;
}

function findPerfectMove(board, color) {
    const moves = findPossibleMoves(board, color);
    let bestMove = null;
    let bestScore = -Infinity;
    
    for (const move of moves) {
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        // 終盤は完全読み（深さ制限なし）
        const score = alphaBeta(newBoard, 20, -Infinity, Infinity, false, color);
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }
    return bestMove;
}

// 探索の打ち切り判定を緩和
function alphaBeta(board, depth, alpha, beta, maximizingPlayer, player) {
    // メモリ使用量が危険な場合のみ打ち切り
    if (performance.memory && performance.memory.usedJSHeapSize > 0.8 * performance.memory.jsHeapSizeLimit) {
        return evaluateBoard(board, player);
    }

    const hash = computeZobristHash(board);
    const tableEntry = transpositionTable.get(hash);
    
    // キャッシュヒット時の処理
    if (tableEntry && tableEntry.depth >= depth) {
        return tableEntry.score;
    }

    if (depth === 0 || (!hasValidMove(board, 'black') && !hasValidMove(board, 'white'))) {
        const score = evaluateBoard(board, player);
        // 評価値をキャッシュ
        transpositionTable.set(hash, { depth, score });
        return score;
    }

    const opponent = player === 'white' ? 'black' : 'white';

    if (maximizingPlayer) {
        let maxEval = -Infinity;
        for (const move of findPossibleMoves(board, player)) {
            const newBoard = JSON.parse(JSON.stringify(board));
            applyMove(newBoard, move[0], move[1], player);
            const evaluation = alphaBeta(newBoard, depth - 1, alpha, beta, false, player);
            maxEval = Math.max(maxEval, evaluation);
            alpha = Math.max(alpha, evaluation);
            if (beta <= alpha) {
                break;
            }
        }
        // 評価値をキャッシュ
        transpositionTable.set(hash, { depth, score: maxEval });
        return maxEval;
    } else {
        let minEval = Infinity;
        for (const move of findPossibleMoves(board, opponent)) {
            const newBoard = JSON.parse(JSON.stringify(board));
            applyMove(newBoard, move[0], move[1], opponent);
            const evaluation = alphaBeta(newBoard, depth - 1, alpha, beta, true, player);
            minEval = Math.min(minEval, evaluation);
            beta = Math.min(beta, evaluation);
            if (beta <= alpha) {
                break;
            }
        }
        // 評価値をキャッシュ
        transpositionTable.set(hash, { depth, score: minEval });
        return minEval;
    }
}

function monteCarloSimulation(board, move, player, simulations = 100) {
    let wins = 0;
    const newBoard = JSON.parse(JSON.stringify(board));
    applyMove(newBoard, move[0], move[1], player);

    for (let i = 0; i < simulations; i++) {
        const simBoard = JSON.parse(JSON.stringify(newBoard));
        const result = playRandomGame(simBoard, player === 'black' ? 'white' : 'black');
        if (result === player) wins++;
    }
    return wins / simulations;
}

function playRandomGame(board, startPlayer) {
    const simBoard = JSON.parse(JSON.stringify(board));
    let currentPlayer = startPlayer;

    while (true) {
        if (!hasValidMove(simBoard, 'black') && !hasValidMove(simBoard, 'white')) {
            break;
        }

        const moves = findPossibleMoves(simBoard, currentPlayer);
        if (moves.length === 0) {
            currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
            continue;
        }

        const randomMove = moves[Math.floor(Math.random() * moves.length)];
        applyMove(simBoard, randomMove[0], randomMove[1], currentPlayer);
        currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
    }

    return evaluateGameWinner(simBoard);
}

function evaluateGameWinner(board) {
    const { black, white } = countStones(board);
    if (black > white) return 'black';
    if (white > black) return 'white';
    return 'draw';
}

// NegaScout探索を実装
function negaScout(board, depth, alpha, beta, color) {
    if (depth === 0 || (!hasValidMove(board, 'black') && !hasValidMove(board, 'white'))) {
        return evaluateBoard(board, color);
    }

    const hash = computeZobristHash(board);
    const tableEntry = transpositionTable.get(hash);
    if (tableEntry && tableEntry.depth >= depth) {
        return tableEntry.score;
    }

    const moves = findPossibleMoves(board, color);
    if (moves.length === 0) {
        return -negaScout(board, depth - 1, -beta, -alpha, color === 'black' ? 'white' : 'black');
    }

    let firstChild = true;
    let score = -Infinity;
    
    for (const move of moves) {
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        
        let currentScore;
        if (firstChild) {
            currentScore = -negaScout(newBoard, depth - 1, -beta, -alpha, color === 'black' ? 'white' : 'black');
        } else {
            // Null Window Search
            currentScore = -negaScout(newBoard, depth - 1, -(alpha + 1), -alpha, color === 'black' ? 'white' : 'black');
            if (currentScore > alpha && currentScore < beta) {
                // Re-search with full window
                currentScore = -negaScout(newBoard, depth - 1, -beta, -currentScore, color === 'black' ? 'white' : 'black');
            }
        }

        score = Math.max(score, currentScore);
        alpha = Math.max(alpha, score);
        
        if (alpha >= beta) {
            break;
        }
        
        firstChild = false;
    }

    transpositionTable.set(hash, { depth, score });
    return score;
}