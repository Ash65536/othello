import { findPossibleMoves } from './Utils.js';

function findBestMove(board, color) {
    let bestMove = null;
    let bestScore = -Infinity;
    for (const move of findPossibleMoves(board, color)) {
        const newBoard = JSON.parse(JSON.stringify(board));
        applyMove(newBoard, move[0], move[1], color);
        const score = alphaBeta(newBoard, 3, -Infinity, Infinity, false, color);
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }
    return bestMove;
}

function alphaBeta(board, depth, alpha, beta, maximizingPlayer, player) {
    if (depth === 0 || (!hasValidMove(board, 'black') && !hasValidMove(board, 'white'))) {
        return evaluateBoard(board, player);
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
        return minEval;
    }
}

function evaluateBoard(board, player) {
    let score = 0;
    for (const row of board) {
        for (const cell of row) {
            if (cell === player) {
                score++;
            } else if (cell !== null) {
                score--;
            }
        }
    }
    return score;
}
