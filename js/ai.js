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
        return 9;      // 終盤はより深く（深さ9）
    } else if (remainingMoves <= 16) {
        return 8;      // 終盤近く（深さ8）
    } else if (remainingMoves <= 32) {
        return 7;      // 中盤（深さ7）
    }
    return 6;          // 序盤（深さ6）
}

// メモリ使用量を監視し、必要に応じてテーブルをクリア
function cleanTranspositionTable() {
    if (transpositionTable.size > 2000000) {  // 200万エントリーまで許容
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

// 角に関する定数
const CORNERS = [[0,0], [0,7], [7,0], [7,7]];
const CORNER_ADJACENTS = {
    '0,0': [[0,1], [1,0], [1,1]],
    '0,7': [[0,6], [1,7], [1,6]],
    '7,0': [[6,0], [7,1], [6,1]],
    '7,7': [[6,7], [7,6], [6,6]]
};

function evaluateBoard(board, player) {
    const opponent = player === 'black' ? 'white' : 'black';
    let score = 0;
    
    // 基本の評価
    // ...existing evaluation code...

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

    // 角の支配とその周辺の評価を強化
    score += evaluateCornerStrategy(board, player, opponent) * 150;

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
            // 角が空いている場合、そ���角を取れる位置にあるかを評価
            if (isValidMove(board, x, y, player)) {
                score += 15;
            }
        }
    }
    return score;
}

function evaluateCornerStrategy(board, player, opponent) {
    let score = 0;

    for (const [cornerX, cornerY] of CORNERS) {
        const cornerKey = `${cornerX},${cornerY}`;
        const adjacents = CORNER_ADJACENTS[cornerKey];

        // 角が空いている場合
        if (board[cornerX][cornerY] === null) {
            // 相手が角を取れる状況を極めて重要視
            if (isValidMove(board, cornerX, cornerY, opponent)) {
                score -= 200; // 相手が角を取れる状況は大幅な減点
            }
            
            // 自分が角を取れる状況は高評価
            if (isValidMove(board, cornerX, cornerY, player)) {
                score += 100;
            }

            // 角が空いているときは周辺の石を置くことを避ける
            for (const [adjX, adjY] of adjacents) {
                if (board[adjX][adjY] === player) {
                    score -= 50; // 角が空いている状態での周辺への配置は減点
                }
            }
        } 
        // 角を自分が取っている場合
        else if (board[cornerX][cornerY] === player) {
            score += 150; // 角を確保している場合は高得点
            
            // 角を取った後は周辺の石を積極的に置く
            for (const [adjX, adjY] of adjacents) {
                if (board[adjX][adjY] === player) {
                    score += 25; // 安定した石を増やす
                }
            }
        }
        // 角を相手が取っている場合
        else {
            score -= 150; // 角を取られている場合は大幅な減点
            
            // 相手の角周りの拡大を防ぐ
            for (const [adjX, adjY] of adjacents) {
                if (board[adjX][adjY] === null && isValidMove(board, adjX, adjY, player)) {
                    score += 15; // 相手の展開を防ぐ手は優先
                }
            }
        }
    }

    // X打ちの評価（角の斜め内側）
    const xSquares = [[1,1], [1,6], [6,1], [6,6]];
    for (const [x, y] of xSquares) {
        if (board[x][y] === player) {
            // 対応する角が空いているときはX打ちを避ける
            const cornerX = x === 1 ? 0 : 7;
            const cornerY = y === 1 ? 0 : 7;
            if (board[cornerX][cornerY] === null) {
                score -= 45;
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
    
    // 序盤のランダム選択を制限（最初の5手程度まで）
    if (remainingMoves >= 55) {
        // 最初の数手はランダムだが、明らかに不利な手は避ける
        const goodMoves = moves.filter(move => {
            // X打ちを避ける
            if ((move[0] === 1 && move[1] === 1) || 
                (move[0] === 1 && move[1] === 6) ||
                (move[0] === 6 && move[1] === 1) ||
                (move[0] === 6 && move[1] === 6)) {
                return false;
            }
            // 角の隣も避ける
            if (isNextToCorner(move[0], move[1])) {
                return false;
            }
            return true;
        });
        
        return goodMoves.length > 0 ? 
            goodMoves[Math.floor(Math.random() * goodMoves.length)] : 
            moves[Math.floor(Math.random() * moves.length)];
    }
    
    // 角を取れる場合は即座に選択
    for (const [cornerX, cornerY] of CORNERS) {
        if (board[cornerX][cornerY] === null && isValidMove(board, cornerX, cornerY, color)) {
            return [cornerX, cornerY];
        }
    }

    // 相手が次に角を取れる手を防ぐ
    const opponent = color === 'black' ? 'white' : 'black';
    const dangerousCornerMoves = findDangerousCornerMoves(board, color, opponent);
    if (dangerousCornerMoves.length > 0) {
        return dangerousCornerMoves[0];
    }

    // 中盤はアルファベータ法を使用
    let bestMove = null;
    let bestScore = -Infinity;
    
    for (const move of moves) {
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        // 通常の深さに加えて、重要な局面ではさらに深く探索
        const extraDepth = isImportantPosition(newBoard) ? 2 : 0;
        const score = alphaBeta(newBoard, depth + extraDepth, -Infinity, Infinity, false, color);
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }
    return bestMove;
}

// 角の隣かどうかをチェックする関数を追加
function isNextToCorner(row, col) {
    const nextToCornerPositions = [
        [0,1], [1,0], [1,1],      // 左上角の隣
        [0,6], [1,7], [1,6],      // 右上角の隣
        [6,0], [7,1], [6,1],      // 左下角の隣
        [6,7], [7,6], [6,6]       // 右下角の隣
    ];
    
    return nextToCornerPositions.some(([r, c]) => r === row && c === col);
}

function findPerfectMove(board, color) {
    const moves = findPossibleMoves(board, color);
    let bestMove = null;
    let bestScore = -Infinity;
    
    for (const move of moves) {
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        // 終盤の完全読みの深さを増やす
        const score = alphaBeta(newBoard, 24, -Infinity, Infinity, false, color);
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }
    return bestMove;
}

function findDangerousCornerMoves(board, player, opponent) {
    const preventiveMoves = [];
    
    for (const [cornerX, cornerY] of CORNERS) {
        if (board[cornerX][cornerY] !== null) continue;
        
        // 相手が次に角を取れる状況をチェック
        const testBoard = JSON.parse(JSON.stringify(board));
        if (isValidMove(testBoard, cornerX, cornerY, opponent)) {
            // この状況を防げる手を探す
            const moves = findPossibleMoves(board, player);
            for (const move of moves) {
                const simBoard = JSON.parse(JSON.stringify(board));
                applyMove(simBoard, move[0], move[1], player);
                if (!isValidMove(simBoard, cornerX, cornerY, opponent)) {
                    preventiveMoves.push(move);
                }
            }
        }
    }
    
    return preventiveMoves;
}

function alphaBeta(board, depth, alpha, beta, maximizingPlayer, player) {
    const hash = computeZobristHash(board);
    const tableEntry = transpositionTable.get(hash);
    
    // キャッシュヒット時の処理
    if (tableEntry && tableEntry.depth >= depth) {
        return tableEntry.score;
    }

    // 終了条件をより詳細に
    if (depth === 0 || (!hasValidMove(board, 'black') && !hasValidMove(board, 'white'))) {
        const score = evaluateBoard(board, player);
        // 評価値をキャッシュ
        transpositionTable.set(hash, { depth, score });
        return score;
    }

    const opponent = player === 'white' ? 'black' : 'white';
    const moves = findPossibleMoves(board, maximizingPlayer ? player : opponent);
    
    // 手の並び替えを実施（より良い手を先に探索）
    moves.sort((a, b) => {
        const scoreA = getQuickEvaluation(board, a, player);
        const scoreB = getQuickEvaluation(board, b, player);
        return scoreB - scoreA;
    });

    if (maximizingPlayer) {
        let maxEval = -Infinity;
        for (const move of moves) {
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
        for (const move of moves) {
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

// 簡易評価関数（手の並び替え用）
function getQuickEvaluation(board, move, player) {
    let score = 0;
    
    // 角は最高評価
    if ((move[0] === 0 || move[0] === 7) && (move[1] === 0 || move[1] === 7)) {
        score += 100;
    }
    
    // 角の隣は低評価
    if (isNextToCorner(move[0], move[1])) {
        score -= 50;
    }
    
    return score;
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