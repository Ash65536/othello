document.addEventListener('DOMContentLoaded', () => {
    const boardElement = document.querySelector('.board');
    const messageElement = document.querySelector('.message');
    const board = createBoard();
    let currentPlayer = 'black';

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
                }
                cell.addEventListener('click', () => handleCellClick(i, j));
                boardElement.appendChild(cell);
            }
        }
    }

    function handleCellClick(row, col) {
        if (isValidMove(board, row, col, currentPlayer)) {
            applyMove(board, row, col, currentPlayer);
            currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
            if (!hasValidMove(board, currentPlayer)) {
                currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
                if (!hasValidMove(board, currentPlayer)) {
                    endGame();
                    return;
                }
            }
            renderBoard();
        }
    }

    function endGame() {
        const { black, white } = countStones(board);
        messageElement.textContent = `ゲーム終了！ 黒: ${black}, 白: ${white}`;
    }

    renderBoard();
});
