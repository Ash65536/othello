document.addEventListener('DOMContentLoaded', () => {
    const boardElement = document.querySelector('.board');
    const messageElement = document.querySelector('.message');
    const restartButton = document.getElementById('restartButton');
    let board = createBoard();
    let currentPlayer = 'black';
    let isProcessing = false;  // 追加: 処理中フラグ
    let aiWorker = null;
    let lastMove = null;  // Add this line to track the last move
    let currentDifficulty = 'normal';  // デフォルトの難易度
    const difficultyButtons = document.querySelectorAll('.difficulty-btn');
    let playerColor = 'black';  // プレイヤーの色
    let setupComplete = false;  // セットアップ完了フラグ
    
    // セットアップ画面の要素
    const gameSetup = document.getElementById('gameSetup');
    const gameBoard = document.getElementById('gameBoard');
    const startGameBtn = document.getElementById('startGameBtn');
    const turnButtons = document.querySelectorAll('.turn-btn');

    // 手番選択の処理
    turnButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            turnButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (btn.dataset.turn === 'random') {
                playerColor = Math.random() < 0.5 ? 'black' : 'white';
            } else {
                playerColor = btn.dataset.turn;
            }
        });
    });

    // ゲーム開始ボタンの処理
    startGameBtn.addEventListener('click', () => {
        if (!document.querySelector('.difficulty-btn.active')) {
            alert('AIの強さを選択してください');
            return;
        }
        if (!document.querySelector('.turn-btn.active')) {
            alert('手番を選択してください');
            return;
        }

        setupComplete = true;
        gameSetup.style.display = 'none';
        gameBoard.style.display = 'flex';
        startGame();
    });

    function startGame() {
        // ボードの初期化
        board = createBoard();
        board[3][3] = board[4][4] = 'white';
        board[3][4] = board[4][3] = 'black';
        currentPlayer = 'black';
        isProcessing = false;
        
        renderBoard();

        // AIが先手の場合（プレイヤーが後手の場合）
        if (currentPlayer !== playerColor) {
            setTimeout(() => {
                handleAITurn();
            }, 500);
        }
    }

    // 難易度選択の処理
    difficultyButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            difficultyButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentDifficulty = btn.dataset.level;
            if (aiWorker) {
                aiWorker.terminate();
                aiWorker = null;
            }
        });
    });

    // デフォルトの難易度をアクティブに
    document.querySelector(`[data-level="normal"]`).classList.add('active');

    // タッチデバイス検出
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    let touchStartTime = 0;
    let touchStartPosition = null;

    // 初期配置
    board[3][3] = 'white';
    board[3][4] = 'black';
    board[4][3] = 'black';
    board[4][4] = 'white';

    function renderBoard() {
        const fragment = document.createDocumentFragment();
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
                fragment.appendChild(cell);
            }
        }
        boardElement.innerHTML = '';
        boardElement.appendChild(fragment);

        // 現在のプレイヤーが置ける場所がない場合、パス処理を実行
        if (!hasValidMove(board, currentPlayer)) {
            if (!isProcessing) {
                handlePass();
            }
        }
    }

    function handleCellClick(row, col) {
        if (!setupComplete) return;
        if (isTouchDevice) {
            const now = Date.now();
            if (now - touchStartTime < 300) {
                return;
            }
            touchStartTime = now;
        }

        // プレイヤーの手番でない場合や処理中は操作を受け付けない
        if (isProcessing || currentPlayer !== playerColor) return;

        if (isValidMove(board, row, col, currentPlayer)) {
            isProcessing = true;
            lastMove = [row, col];
            const flippedStones = applyMove(board, row, col, currentPlayer);
            
            // アニメーション処理
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
                if (currentPlayer !== playerColor) {
                    setTimeout(() => {
                        handleAITurn();
                    }, 500);
                } else {
                    isProcessing = false;
                }
            });
        }
    }

    // タッチイベントの最適化
    if (isTouchDevice) {
        boardElement.addEventListener('touchstart', (e) => {
            e.preventDefault();
            touchStartPosition = {
                x: e.touches[0].clientX,
                y: e.touches[0].clientY
            };
        }, { passive: false });

        boardElement.addEventListener('touchend', (e) => {
            if (!touchStartPosition) return;
            
            const touch = e.changedTouches[0];
            const dx = touch.clientX - touchStartPosition.x;
            const dy = touch.clientY - touchStartPosition.y;
            
            // スワイプ検出（15px以上の移動でキャンセル）
            if (Math.abs(dx) > 15 || Math.abs(dy) > 15) {
                return;
            }
            
            touchStartPosition = null;
        });
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
                currentPlayer = passingPlayer;
            }

            // 現在のプレイヤーがAIの場合
            if (currentPlayer !== playerColor) {
                handleAITurn();
            } else {
                isProcessing = false;
                renderBoard();
            }
        }, 1000);
    }

    // AIの���を別関数として実装
    function handleAIMove(row, col) {
        const flippedStones = applyMove(board, row, col, currentPlayer);
        lastMove = [row, col];

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
            
            if (!hasValidMove(board, currentPlayer)) {
                if (!hasValidMove(board, currentPlayer === 'black' ? 'white' : 'black')) {
                    endGame();
                } else {
                    handlePass();
                }
                return;
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
        
        // 処理中フラグをリセット
        isProcessing = false;
        
        // AIワーカーをクリーンアップ
        if (aiWorker) {
            aiWorker.terminate();
            aiWorker = null;
        }
        
        // サーバーに結果を送信
        try {
            fetch(`${baseUrl}/game_end`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ winner })
            });
        } catch (error) {
            console.error('Failed to send game result:', error);
        }

        if (black > white) {
            result += ' - 黒の勝ち！';
        } else if (white > black) {
            result += ' - 白の勝ち！';
        } else {
            result += ' - 引き分け';
        }
        messageElement.textContent = result;

        // リスタートボタンを表示
        restartButton.style.display = 'block';
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
        messageElement.innerHTML = "AIが思考中<span class='dots'>...</span>";
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
        
        const gameState = {
            board: board,
            color: 'white',
            counts: countStones(board),
            lastMove: lastMove || null,
            remainingMoves: board.flat().filter(cell => cell === null).length,
            difficulty: currentDifficulty  // 難易度を追加
        };

        // 最小待機時間を設定（1秒）
        const minWaitTime = 1000;
        const startTime = Date.now();
        
        // AIに送信する前に最小待機時間を確保
        const timeoutId = setTimeout(() => {
            if (aiWorker) {
                aiWorker.terminate();
                aiWorker = null;
                hideLoading();
                handleFallbackAI();
            }
        }, 3000);

        aiWorker.onmessage = function(e) {
            clearTimeout(timeoutId);
            const { move, evaluation } = e.data;
            const elapsedTime = Date.now() - startTime;
            
            // 最小待機時間を確保するため、必要に応じて遅延を追加
            const remainingWait = minWaitTime - elapsedTime;
            if (remainingWait > 0) {
                setTimeout(() => {
                    if (move) {
                        updateAIThinkingMessage(evaluation);
                        handleAIMove(move[0], move[1]);
                    } else {
                        handlePass();
                    }
                    hideLoading();
                }, remainingWait);
            } else {
                if (move) {
                    updateAIThinkingMessage(evaluation);
                    handleAIMove(move[0], move[1]);
                } else {
                    handlePass();
                }
                hideLoading();
            }
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
        let message = "";  // デフォルトは空文字列
        if (evaluation > 500) {
            message = "優勢";
        } else if (evaluation > 200) {
            message = "やや優勢";
        } else if (evaluation < -500) {
            message = "劣勢";
        } else if (evaluation < -200) {
            message = "やや劣勢";
        } else if (Math.abs(evaluation) > 50) {
            message = "互角";
        }
        // 評価値が小さい場合はメッセージを表示しない
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
            // X打ちを避ける
            if ((x === 1 || x === 6) && (y === 1 || y === 6)) return false;
            return true;
        });

        return safeMoves.length > 0 ? safeMoves[0] : moves[0];
    }

    function restartGame() {
        gameBoard.style.display = 'none';
        gameSetup.style.display = 'flex';
        setupComplete = false;
        if (aiWorker) {
            aiWorker.terminate();
            aiWorker = null;
        }
    }

    // リスタートボタンのイベントリスナー
    restartButton.addEventListener('click', restartGame);

    renderBoard();
});
