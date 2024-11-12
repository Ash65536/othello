document.addEventListener('DOMContentLoaded', () => {
    const boardElement = document.querySelector('.board');
    const messageElement = document.querySelector('.message');
    const board = createBoard();
    let currentPlayer = 'black';
    let isProcessing = false;  // 追加: 処理中フラグ
    let aiWorker = null;
    let lastMove = null;  // Add this line to track the last move

    // 初期配置
    board[3][3] = 'white';
    board[3][4] = 'black';
    board[4][3] = 'black';
    board[4][4] = 'white';

    function renderBoard() {
        boardElement.innerHTML = '';
        for (let i = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                if (board[i][j]) {
                    const stone = document.createElement('div');
                    stone.classList.add('stone', board[i][j]);
                    cell.appendChild(stone);
                } else if (isValidMove(board, i, j, currentPlayer)) {
                    const hint = document.createElement('div');
                    hint.classList.add('hint', currentPlayer);
                    cell.appendChild(hint);
                }
                cell.addEventListener('click', () => handleCellClick(i, j));
                boardElement.appendChild(cell);
            }
        }

        // 現在のプレイヤーが置ける場所がない場合、パス処理を実行
        if (!hasValidMove(board, currentPlayer)) {
            if (!isProcessing) {
                handlePass();
            }
        }
    }

    function handleCellClick(row, col) {
        // 処理中は操作を受け付けない
        if (isProcessing || currentPlayer === 'white') return;

        if (isValidMove(board, row, col, currentPlayer)) {
            isProcessing = true;  // 処理開始
            lastMove = [row, col];  // Add this line to update lastMove
            const flippedStones = applyMove(board, row, col, currentPlayer);
            
            const animationPromises = flippedStones.map(([x, y]) => {
                return new Promise(resolve => {
                    const index = x * 8 + y;
                    const cell = document.querySelector(`.board .cell:nth-child(${index + 1})`);
                    const stone = cell.querySelector('.stone');
                    if (stone) {
                        // 現在の色に基づいて適切なアニメーションクラスを追加
                        const fromColor = stone.classList.contains('black') ? 'black-to-white' : 'white-to-black';
                        stone.classList.add(fromColor);
                        setTimeout(() => {
                            stone.classList.remove(fromColor);
                            // クラスを更新して新しい色を設定
                            stone.classList.remove('black', 'white');
                            stone.classList.add(currentPlayer);
                            resolve();
                        }, 500);
                    } else {
                        resolve();
                    }
                });
            });

            Promise.all(animationPromises).then(() => {
                currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
                renderBoard();
                
                // AIの手番の処理
                if (currentPlayer === 'white') {
                    setTimeout(() => {
                        handleAITurn();
                    }, 500);
                } else {
                    isProcessing = false;  // プレイヤーの手番に戻る時に処理完了
                }
            });
        }
    }

    // パス処理を行う関数を修正
    function handlePass() {
        const passingPlayer = currentPlayer;
        const opponent = currentPlayer === 'black' ? 'white' : 'black';
        messageElement.textContent = `${currentPlayer === 'black' ? '黒' : '白'}のパスです`;
        
        setTimeout(() => {
            currentPlayer = opponent;
            messageElement.textContent = '';
            
            // 相手もパスの場合はゲーム終了
            if (!hasValidMove(board, currentPlayer)) {
                if (!hasValidMove(board, passingPlayer)) {
                    endGame();
                    return;
                }
                // 相手もパスの場合は元のプレイヤーに戻す
                currentPlayer = passingPlayer;
            }

            // AIの手番の場合
            if (currentPlayer === 'white') {
                handleAITurn();
            } else {
                // プレイヤーの手番の場合
                isProcessing = false;
                renderBoard(); // ボードを再描画して有効な手を表示
            }
        }, 1000);
    }

    // AIの手を別関数として実装
    function handleAIMove(row, col) {
        const flippedStones = applyMove(board, row, col, 'white');
        lastMove = [row, col];  // Add this line to update lastMove when AI moves
        const animationPromises = flippedStones.map(([x, y]) => {
            return new Promise(resolve => {
                const index = x * 8 + y;
                const cell = document.querySelector(`.board .cell:nth-child(${index + 1})`);
                const stone = cell.querySelector('.stone');
                if (stone) {
                    const fromColor = stone.classList.contains('black') ? 'black-to-white' : 'white-to-black';
                    stone.classList.add(fromColor);
                    setTimeout(() => {
                        stone.classList.remove(fromColor);
                        stone.classList.remove('black', 'white');
                        stone.classList.add('white');
                        resolve();
                    }, 500);
                } else {
                    resolve();
                }
            });
        });

        Promise.all(animationPromises).then(() => {
            currentPlayer = 'black';
            
            // パスが必要か確認
            if (!hasValidMove(board, currentPlayer)) {
                if (!hasValidMove(board, 'white')) {
                    endGame();
                } else {
                    handlePass();
                    return; // パスの場合は以降の処理をスキップ
                }
            }

            renderBoard();
            isProcessing = false;
        });
    }

    function getFlippableCells(board, row, col, player) {
        const directions = [
            [-1, 0], [1, 0], [0, -1], [0, 1],
            [-1, -1], [-1, 1], [1, -1], [1, 1]
        ];
        const opponent = player === 'black' ? 'white' : 'black';
        const flippableCells = [];

        for (const [dx, dy] of directions) {
            let x = row + dx, y = col + dy;
            let cellsInDirection = [];

            while (x >= 0 && x < 8 && y >= 0 && y < 8 && board[x][y] === opponent) {
                cellsInDirection.push([x, y]);
                x += dx;
                y += dy;
            }

            if (x >= 0 && x < 8 && y >= 0 && y < 8 && board[x][y] === player) {
                flippableCells.push(...cellsInDirection);
            }
        }

        return flippableCells;
    }

    function animateFlips(cells) {
        const stones = document.querySelectorAll('.stone');
        cells.forEach(([row, col]) => {
            const index = row * 8 + col;
            const stone = stones[index];
            if (stone) {
                stone.classList.add('flipping');
                setTimeout(() => {
                    stone.classList.remove('flipping');
                }, 500);
            }
        });
    }

    function endGame() {
        const { black, white } = countStones(board);
        let result = `ゲーム終了！ 黒: ${black}, 白: ${white}`;
        let winner = black > white ? 'black' : white > black ? 'white' : 'draw';
        
        // サーバーに結果を送信
        fetch(`${baseUrl}/game_end`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ winner })
        });

        if (black > white) {
            result += ' - 黒の勝ち！';
        } else if (white > black) {
            result += ' - 白の勝ち！';
        } else {
            result += ' - 引き分け';
        }
        messageElement.textContent = result;
    }

    function initAIWorker() {
        aiWorker = new Worker('js/ai.worker.js');
        aiWorker.onmessage = function(e) {
            const { move } = e.data;
            if (move) {
                handleAIMove(move[0], move[1]);
            } else {
                handlePass();
            }
            hideLoading();
        };
    }

    function showLoading() {
        messageElement.textContent = "AIが考え中...（最大10秒）";
        messageElement.classList.add('thinking');
    }

    function hideLoading() {
        messageElement.classList.remove('thinking');
        messageElement.textContent = "";
    }

    // AIのターン処理を修正
    function handleAITurn() {
        if (!aiWorker) {
            initAIWorker();
        }
        
        isProcessing = true;
        showLoading();
        
        // ゲームの状態を含めてAIに送信
        const gameState = {
            board: board,
            color: 'white',
            counts: countStones(board),
            lastMove: lastMove || null  // Update this line to handle null case
        };
        
        const timeoutId = setTimeout(() => {
            if (aiWorker) {
                aiWorker.terminate();
                aiWorker = null;
                hideLoading();
                handleFallbackAI();
            }
        }, 10000);

        aiWorker.onmessage = function(e) {
            clearTimeout(timeoutId);
            const { move, evaluation } = e.data;
            
            if (move) {
                // 評価値に基づいてメッセージを表示
                updateAIThinkingMessage(evaluation);
                handleAIMove(move[0], move[1]);
            } else {
                handlePass();
            }
            hideLoading();
        };

        aiWorker.postMessage(gameState);
    }

    function handleFallbackAI() {
        // フォールバックAIロジック（サーバーがタイムアウトした場合）
        const moves = findPossibleMoves(board, 'white');
        if (moves.length > 0) {
            const priorityMove = findPriorityMove(board, 'white');
            handleAIMove(priorityMove[0], priorityMove[1]);
        } else {
            handlePass();
        }
    }

    function updateAIThinkingMessage(evaluation) {
        let message = "AIの評価: ";
        if (evaluation > 500) {
            message += "優勢です";
        } else if (evaluation > 200) {
            message += "やや優勢です";
        } else if (evaluation < -500) {
            message += "劣勢です";
        } else if (evaluation < -200) {
            message += "やや劣勢です";
        } else {
            message += "互角です";
        }
        messageElement.textContent = message;
    }

    // 優先順位に基づく手の選択（タイムアウト時用）
    function findPriorityMove(board, color) {
        const moves = findPossibleMoves(board, color);
        if (moves.length === 0) return null;

        // 角を優先
        for (const [cornerX, cornerY] of [[0,0], [0,7], [7,0], [7,7]]) {
            const cornerMove = moves.find(([x, y]) => x === cornerX && y === cornerY);
            if (cornerMove) return cornerMove;
        }

        // 危険な手を避ける
        const safeMoves = moves.filter(([x, y]) => {
            // 角の隣を避ける
            if ((x === 0 || x === 7) && (y === 1 || y === 6)) return false;
            if ((x === 1 || x === 6) && (y === 0 || y === 7)) return false;
            // X打ちを避け���
            if ((x === 1 || x === 6) && (y === 1 || y === 6)) return false;
            return true;
        });

        return safeMoves.length > 0 ? safeMoves[0] : moves[0];
    }

    renderBoard();
});
