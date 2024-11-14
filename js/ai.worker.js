let baseUrl = 'http://localhost:5000';

self.onmessage = async function(e) {
    const { board, color, remainingMoves, difficulty } = e.data;
    
    try {
        let move = null;
        let evaluation = 0;

        switch (difficulty) {
            case 'easy':
                // かんたん: 完全ランダム
                move = getRandomMove(board, color);
                break;
            
            case 'normal':
                // ふつう: 基本的な評価関数と浅い探索
                move = await getNormalMove(board, color);
                break;
            
            case 'hard':
                // つよい: より深い探索と改善された評価関数
                move = await getHardMove(board, color);
                break;
            
            case 'expert':
                // ゲキムズ: 完全な探索と高度な評価関数
                move = await getExpertMove(board, color);
                break;
            
            case 'god':
                // 神: すべての最適化を適用
                move = await getGodMove(board, color);
                break;

            case 'dqn':
                // DQNモード
                const response = await fetch(`${baseUrl}/ai_move`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'POST',
                        'Access-Control-Allow-Headers': 'Content-Type'
                    },
                    mode: 'cors',
                    body: JSON.stringify({ 
                        board: board,
                        color: color,
                        difficulty: 'dqn'
                    })
                });

                if (!response.ok) {
                    throw new Error('DQN response was not ok');
                }

                const data = await response.json();
                if (data.move) {
                    move = [data.move.row, data.move.col];
                    evaluation = data.evaluation;
                }
                break;
        }

        self.postMessage({ move, evaluation });
    } catch (error) {
        console.error('AI Error:', error);
        // エラー時は簡単な手を選択
        const move = getRandomMove(board, color);
        self.postMessage({ move, evaluation: 0 });
    }
};

function getRandomMove(board, color) {
    const moves = findPossibleMoves(board, color);
    if (!moves.length) return null;
    
    // かんたん: 50%ランダム、50%基本評価で少し強く
    if (Math.random() < 0.5) {
        return moves[Math.floor(Math.random() * moves.length)];
    }

    // 基本的な評価を行う
    const scoredMoves = moves.map(move => {
        const score = evaluateBasicMove(board, move, color);
        return { move, score };
    });
    
    scoredMoves.sort((a, b) => b.score - a.score);
    return scoredMoves[0].move;
}

function evaluateBasicMove(board, move, color) {
    let score = 0;
    const [row, col] = move;
    
    // 角は高評価
    if ((row === 0 || row === 7) && (col === 0 || col === 7)) {
        score += 100;
    }
    
    // 端も評価
    if (row === 0 || row === 7 || col === 0 || col === 7) {
        score += 20;
    }
    
    // ひっくり返せる数も考慮
    const flipped = getFlippableCells(board, row, col, color);
    score += flipped.length * 5;
    
    return score;
}

function getNormalMove(board, color) {
    const remainingMoves = board.flat().filter(cell => cell === null).length;
    const depth = remainingMoves <= 12 ? 4 : 3; // 終盤は深く読む
    
    // ふつう: AlphaBeta探索 + パターン認識
    const moves = sortMovesByPriority(board, color);
    if (!moves.length) return null;
    
    // 即詰み回避と必勝手を探索
    const criticalMove = findCriticalMove(board, color);
    if (criticalMove) return criticalMove;
    
    let bestMove = moves[0];
    let bestScore = -Infinity;
    
    for (const move of moves) {
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        const score = -alphaBetaWithMemory(
            newBoard, 
            depth - 1,
            -Infinity,
            Infinity,
            color === 'black' ? 'white' : 'black'
        );
        
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }
    
    return bestMove;
}

function getHardMove(board, color) {
    const remainingMoves = board.flat().filter(cell => cell === null).length;
    const isEndgame = remainingMoves <= 14;
    const depth = isEndgame ? 6 : remainingMoves <= 25 ? 5 : 4;
    
    // つよい: NegaScout + 評価関数強化 + 定石
    if (remainingMoves >= 45) {
        const bookMove = getOpeningBookMove(board, color);
        if (bookMove) return bookMove;
    }
    
    const moves = sortMovesByPriority(board, color);
    if (!moves.length) return null;
    
    let bestMove = moves[0];
    let bestScore = -Infinity;
    let alpha = -Infinity;
    let beta = Infinity;
    
    for (const move of moves) {
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        const score = -negaScoutEnhanced(
            newBoard,
            depth - 1,
            -beta,
            -alpha,
            color === 'black' ? 'white' : 'black'
        );
        
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
        alpha = Math.max(alpha, score);
    }
    
    return bestMove;
}

async function getExpertMove(board, color) {
    // ゲキムズ: 改良版NegaScout + トランスポジションテーブル + パターン認識
    const remainingMoves = board.flat().filter(cell => cell === null).length;
    const depth = calculateDynamicDepth(remainingMoves);
    cleanTranspositionTable();
    
    const moves = sortMovesByPriority(board, color);
    if (moves.length === 0) return null;
    
    // 明らかに良い手があれば即座に選択
    const criticalMove = findCriticalMove(board, color);
    if (criticalMove) return criticalMove;
    
    let bestMove = moves[0];
    let bestScore = -Infinity;
    let alpha = -Infinity;
    let beta = Infinity;
    
    for (const move of moves) {
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        const score = -await iterativeDeepening(
            newBoard,
            depth,
            -beta,
            -alpha,
            color === 'black' ? 'white' : 'black'
        );
        
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
        alpha = Math.max(alpha, score);
    }
    
    return bestMove;
}

async function getGodMove(board, color) {
    // 神: ハイブリッドエンジン + パターンデータベース + 完全読み
    const remainingMoves = board.flat().filter(cell => cell === null).length;
    
    let upperBound = Infinity;
    let lowerBound = -Infinity;
    
    while (lowerBound < upperBound) {
        let beta = (g === lowerBound) ? g + 1 : g;
        g = negaScoutEnhanced(board, depth, beta - 1, beta, color);
        if (g < beta) {
            upperBound = g;
        } else {
            lowerBound = g;
        }
    }
    
    return g;
}

// 定石データベース
const OPENING_BOOK = {
    // ボードの状態をキーとした定石手のマップ
    // 例: "empty_corner": [[0,0], [7,0], [0,7], [7,7]]
    // 実際の実装では、より多くの定石パターンを追加
};

function getOpeningBookMove(board, color) {
    const boardKey = generateBoardKey(board);
    const bookMoves = OPENING_BOOK[boardKey];
    if (bookMoves) {
        const validMoves = bookMoves.filter(move => 
            isValidMove(board, move[0], move[1], color)
        );
        if (validMoves.length > 0) {
            return validMoves[Math.floor(Math.random() * validMoves.length)];
        }
    }
    return null;
}

function generateBoardKey(board) {
    // ボードの状態を一意の文字列に変換
    return board.flat().map(cell => cell || '_').join('');
}

function getDeepNegaScoutMove(board, color) {
    const depth = Math.min(10, dynamicDepth(board) + 4); // より深い探索
    cleanTranspositionTable();
    
    let bestMove = null;
    let bestScore = -Infinity;
    const moves = sortMovesByPriority(board, color);
    
    for (const move of moves) {
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        const score = -negaScoutEnhanced(
            newBoard,
            depth,
            -Infinity,
            Infinity,
            color === 'black' ? 'white' : 'black'
        );
        
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }
    
    return bestMove;
}

function sortMovesByPriority(board, color) {
    const moves = findPossibleMoves(board, color);
    const scoredMoves = moves.map(move => {
        const score = evaluateMoveStrength(board, move, color);
        return { move, score };
    });
    
    return scoredMoves
        .sort((a, b) => b.score - a.score)
        .map(item => item.move);
}

function evaluateMoveStrength(board, move, color) {
    let score = 0;
    const [row, col] = move;
    
    // 角の評価を強化
    if ((row === 0 || row === 7) && (col === 0 || col === 7)) {
        score += 2000;  // 角の価値を大幅に上昇
    }
    
    // 角を相手に渡す手の評価
    if (willGiveCorner(board, row, col, color)) {
        score -= 3000;  // 角を渡す手に大幅なペナルティ
    }
    
    // 安定石の評価を強化
    const newBoard = JSON.parse(JSON.stringify(board));
    applyMove(newBoard, row, col, color);
    score += evaluateStableStones(newBoard, color) * 100;
    
    // 相手の有効手を制限する評価を追加
    const opponent = color === 'black' ? 'white' : 'black';
    const opponentMoves = findPossibleMoves(newBoard, opponent);
    score -= opponentMoves.length * 30;
    
    // 石を返せる数の評価（脅威度に応じて重み付け）
    const flippedStones = getFlippableCells(board, row, col, color);
    score += evaluateFlips(board, flippedStones, color) * 15;
    
    return score;
}

// 角を相手に渡す手かどうかを判定
function willGiveCorner(board, row, col, color) {
    const opponent = color === 'black' ? 'white' : 'black';
    const corners = [[0,0], [0,7], [7,0], [7,7]];
    const newBoard = JSON.parse(JSON.stringify(board));
    applyMove(newBoard, row, col, color);
    
    // この手を打った後に相手が角を取れるようになるか確認
    for (const [cornerX, cornerY] of corners) {
        if (newBoard[cornerX][cornerY] === null && 
            isValidMove(newBoard, cornerX, cornerY, opponent)) {
            return true;
        }
    }
    return false;
}

// 返せる石の評価を改良
function evaluateFlips(board, flippedStones, color) {
    let score = 0;
    
    for (const [x, y] of flippedStones) {
        // 端の石は価値が高い
        if (x === 0 || x === 7 || y === 0 || y === 7) {
            score += 10;
        }
        // 角に隣接する位置は要注意
        if ((x === 0 || x === 7) && (y === 1 || y === 6) ||
            (x === 1 || x === 6) && (y === 0 || y === 7)) {
            score -= 15;
        }
        // 基本点
        score += 5;
    }
    
    return score;
}

function negaScoutEnhanced(board, depth, alpha, beta, color) {
    const hash = computeZobristHash(board);
    const cached = transpositionTable.get(hash);
    
    if (cached && cached.depth >= depth) {
        if (cached.type === 'exact') return cached.score;
        if (cached.type === 'lower' && cached.score > alpha) alpha = cached.score;
        if (cached.type === 'upper' && cached.score < beta) beta = cached.score;
        if (alpha >= beta) return cached.score;
    }
    
    if (depth === 0) {
        const score = evaluateBoardEnhanced(board, color);
        transpositionTable.set(hash, { 
            depth, 
            score,
            type: 'exact'
        });
        return score;
    }
    
    const moves = sortMovesByPriority(board, color);
    if (moves.length === 0) {
        if (!hasValidMove(board, color === 'black' ? 'white' : 'black')) {
            const score = evaluateBoardEnhanced(board, color);
            transpositionTable.set(hash, {
                depth,
                score,
                type: 'exact'
            });
            return score;
        }
        return -negaScoutEnhanced(board, depth - 1, -beta, -alpha, 
            color === 'black' ? 'white' : 'black');
    }
    
    let firstChild = true;
    let bestScore = -Infinity;
    
    for (const move of moves) {
        // 角を相手に渡す手は深い探索で評価
        const isCornerGiving = willGiveCorner(board, move[0], move[1], color);
        const newDepth = isCornerGiving ? depth + 1 : depth - 1;
        
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        
        let score;
        if (firstChild) {
            score = -negaScoutEnhanced(newBoard, newDepth, -beta, -alpha, 
                color === 'black' ? 'white' : 'black');
        } else {
            score = -negaScoutEnhanced(newBoard, newDepth, -(alpha + 1), -alpha,
                color === 'black' ? 'white' : 'black');
            if (alpha < score && score < beta) {
                score = -negaScoutEnhanced(newBoard, newDepth, -beta, -score,
                    color === 'black' ? 'white' : 'black');
            }
        }
        
        bestScore = Math.max(bestScore, score);
        alpha = Math.max(alpha, score);
        if (alpha >= beta) break;
        firstChild = false;
    }
    
    // トランスポジションテーブルの更新
    const nodeType = 
        bestScore <= alpha ? 'upper' :
        bestScore >= beta ? 'lower' : 'exact';
    
    transpositionTable.set(hash, {
        depth,
        score: bestScore,
        type: nodeType
    });
    
    return bestScore;
}

function negaScoutEnhanced(board, depth, alpha, beta, color) {
    const hash = computeZobristHash(board);
    const cached = transpositionTable.get(hash);
    
    if (cached && cached.depth >= depth) {
        return cached.score;
    }
    
    if (depth === 0) {
        const score = evaluateBoardEnhanced(board, color);
        transpositionTable.set(hash, { depth, score });
        return score;
    }
    
    const moves = sortMovesByPriority(board, color);
    if (moves.length === 0) {
        if (!hasValidMove(board, color === 'black' ? 'white' : 'black')) {
            return evaluateBoardEnhanced(board, color);
        }
        return -negaScoutEnhanced(board, depth - 1, -beta, -alpha, color === 'black' ? 'white' : 'black');
    }
    
    let firstChild = true;
    let bestScore = -Infinity;
    
    for (const move of moves) {
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        
        let score;
        if (firstChild) {
            score = -negaScoutEnhanced(newBoard, depth - 1, -beta, -alpha, color === 'black' ? 'white' : 'black');
        } else {
            score = -negaScoutEnhanced(newBoard, depth - 1, -(alpha + 1), -alpha, color === 'black' ? 'white' : 'black');
            if (alpha < score && score < beta) {
                score = -negaScoutEnhanced(newBoard, depth - 1, -beta, -score, color === 'black' ? 'white' : 'black');
            }
        }
        
        bestScore = Math.max(bestScore, score);
        alpha = Math.max(alpha, score);
        if (alpha >= beta) break;
        firstChild = false;
    }
    
    transpositionTable.set(hash, { depth, score: bestScore });
    return bestScore;
}

function evaluateBoardEnhanced(board, color) {
    const opponent = color === 'black' ? 'white' : 'black';
    let score = 0;
    
    // 基本評価
    score += evaluatePosition(board, color) * 1.0;
    score += evaluateMobility(board, color) * 2.0;
    score += evaluateStability(board, color) * 3.0;
    score -= evaluatePosition(board, opponent) * 1.0;
    score -= evaluateMobility(board, opponent) * 2.0;
    score -= evaluateStability(board, opponent) * 3.0;
    
    // 終盤評価の強化
    const remainingMoves = board.flat().filter(cell => cell === null).length;
    if (remainingMoves <= 16) {
        score += evaluateEndgame(board, color) * (16 - remainingMoves);
    }
    
    return score;
}

function getExpertMove(board, color) {
    // NegaScout + トランスポジションテーブル（深さ6）
    return negaScout(board, 6, -Infinity, Infinity, color);
}

function getGodMove(board, color) {
    // モンテカルロ木探索 + パターンデータベース + 完全読み
    const remainingMoves = board.flat().filter(cell => cell === null).length;
    if (remainingMoves <= 10) {
        return findPerfectMove(board, color);
    }
    return monteCarloTreeSearch(board, color, 1000);
}

function findLocalBestMove(board, color) {
    const moves = findPossibleMoves(board, color);
    if (moves.length === 0) return null;

    // 角が取れる場合は即座に返す
    const cornerMove = moves.find(([x, y]) => 
        (x === 0 && y === 0) || (x === 0 && y === 7) || 
        (x === 7 && y === 0) || (x === 7 && y === 7)
    );
    if (cornerMove) return cornerMove;

    // その他の手を素早く評価
    const scoredMoves = moves.map(move => {
        const score = quickMoveEvaluation(board, move, color);
        return { move, score };
    });

    // スコアで降順ソートして最善手を返す
    scoredMoves.sort((a, b) => b.score - a.score);
    return scoredMoves[0].move;
}

function quickMoveEvaluation(board, [row, col], color) {
    let score = 0;
    
    // 位置による評価
    const positionScore = POSITION_WEIGHTS[row][col];
    score += positionScore;
    
    // 返せる石の数による評価
    const newBoard = JSON.parse(JSON.stringify(board));
    const flippedCount = countFlippableStones(board, row, col, color);
    score += flippedCount * 5;
    
    return score;
}

// 位置の重み付けテーブルを事前定義
const POSITION_WEIGHTS = [
    [100, -20, 20, 5, 5, 20, -20, 100],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [100, -20, 20, 5, 5, 20, -20, 100]
];

function quickEvaluate(board, color) {
    const opponent = color === 'black' ? 'white' : 'black';
    let score = 0;
    
    // 位置の重み付けマップ
    const weights = [
        [120, -20, 20, 5, 5, 20, -20, 120],
        [-20, -40, -5, -5, -5, -5, -40, -20],
        [20, -5, 15, 3, 3, 15, -5, 20],
        [5, -5, 3, 3, 3, 3, -5, 5],
        [5, -5, 3, 3, 3, 3, -5, 5],
        [20, -5, 15, 3, 3, 15, -5, 20],
        [-20, -40, -5, -5, -5, -5, -40, -20],
        [120, -20, 20, 5, 5, 20, -20, 120]
    ];

    // 簡易評価（高速化のため）
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            if (board[i][j] === color) score += weights[i][j];
            else if (board[i][j] === opponent) score -= weights[i][j];
        }
    }

    return score;
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