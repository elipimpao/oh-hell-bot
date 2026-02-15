/**
 * Shared UI components: bid picker, score table, card picker grid.
 */

/**
 * Create bid picker overlay.
 * @param {number} handSize - Maximum possible bid
 * @param {number[]} legalBids - Array of legal bid values
 * @param {function} onBid - Callback when a bid is selected
 * @returns {HTMLElement}
 */
function createBidPicker(handSize, legalBids, onBid) {
    const overlay = document.createElement('div');
    overlay.className = 'bid-overlay';

    const panel = document.createElement('div');
    panel.className = 'bid-panel';

    const title = document.createElement('div');
    title.className = 'bid-title';
    title.textContent = 'Your turn to bid';
    panel.appendChild(title);

    const circles = document.createElement('div');
    circles.className = 'bid-circles';

    for (let i = 0; i <= handSize; i++) {
        const circle = document.createElement('button');
        circle.className = 'bid-circle';
        circle.textContent = i;

        if (legalBids.includes(i)) {
            circle.classList.add('legal');
            circle.addEventListener('click', () => {
                onBid(i);
                overlay.remove();
            });
        } else {
            circle.classList.add('illegal');
            circle.title = 'Dealer restriction: total bids cannot equal hand size';
        }

        circles.appendChild(circle);
    }

    panel.appendChild(circles);
    overlay.appendChild(panel);
    return overlay;
}

/**
 * Create score table modal.
 * @param {object} data - Score data
 * @param {function} onContinue - Callback to dismiss
 * @returns {HTMLElement}
 */
function createScoreTable(data, onContinue) {
    const overlay = document.createElement('div');
    overlay.className = 'score-overlay';

    const panel = document.createElement('div');
    panel.className = 'score-panel';

    const title = document.createElement('div');
    title.className = 'score-title';
    title.textContent = data.gameOver
        ? 'Game Over!'
        : `Round ${data.roundIndex + 1} Complete (${data.handSize} cards)`;
    panel.appendChild(title);

    const table = document.createElement('table');
    table.className = 'score-table';

    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['Player', 'Bid', 'Tricks', 'Score', 'Total'].forEach(text => {
        const th = document.createElement('th');
        th.textContent = text;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    for (let i = 0; i < data.numPlayers; i++) {
        const row = document.createElement('tr');
        const hit = data.bids[i] === data.tricksWon[i];
        if (hit) row.classList.add('hit-bid');

        const cells = [
            data.seatLabels[i],
            data.bids[i] >= 0 ? data.bids[i] : '-',
            data.tricksWon[i],
            data.roundScores ? (data.roundScores[i] > 0 ? `+${data.roundScores[i]}` : '0') : '-',
            data.scores[i],
        ];

        cells.forEach((text, ci) => {
            const td = document.createElement('td');
            td.textContent = text;
            if (ci === 0) td.classList.add('player-name');
            row.appendChild(td);
        });

        tbody.appendChild(row);
    }
    table.appendChild(tbody);
    panel.appendChild(table);

    // Winner announcement for game over
    if (data.gameOver) {
        const maxScore = Math.max(...data.scores);
        const winners = data.scores
            .map((s, i) => ({ score: s, label: data.seatLabels[i] }))
            .filter(x => x.score === maxScore)
            .map(x => x.label);

        const winner = document.createElement('div');
        winner.className = 'winner-announcement';
        winner.textContent = `Winner: ${winners.join(', ')} with ${maxScore} points!`;
        panel.appendChild(winner);
    }

    const btnContainer = document.createElement('div');
    btnContainer.className = 'score-btn-container';

    const btn = document.createElement('button');
    btn.className = 'btn btn-primary';
    btn.textContent = data.gameOver ? 'Back to Setup' : 'Continue';
    btn.addEventListener('click', () => {
        overlay.remove();
        if (onContinue) onContinue();
    });
    btnContainer.appendChild(btn);
    panel.appendChild(btnContainer);

    overlay.appendChild(panel);
    return overlay;
}

/**
 * Create card picker grid (for custom deals and advisor hand selection).
 * @param {Set} selectedCards - Currently selected card integers
 * @param {function} onToggle - Callback(cardInt) when card is toggled
 * @param {object} options - { disabledCards: Set }
 * @returns {HTMLElement}
 */
function createCardPicker(selectedCards, onToggle, options = {}) {
    const container = document.createElement('div');
    container.className = 'card-picker';

    const disabledCards = options.disabledCards || new Set();

    for (let suit = 0; suit < 4; suit++) {
        const row = document.createElement('div');
        row.className = 'card-picker-row';

        const suitLabel = document.createElement('span');
        suitLabel.className = 'card-picker-suit';
        suitLabel.textContent = Card.SUIT_SYMBOLS[suit];
        suitLabel.style.color = Card.SUIT_COLORS[suit];
        row.appendChild(suitLabel);

        for (let rank = 0; rank < 13; rank++) {
            const cardInt = suit * 13 + rank;
            const btn = document.createElement('button');
            btn.className = 'card-picker-btn';
            btn.textContent = Card.RANK_DISPLAY[rank];
            btn.style.color = Card.SUIT_COLORS[suit];
            btn.dataset.card = cardInt;

            if (selectedCards.has(cardInt)) {
                btn.classList.add('selected');
            }
            if (disabledCards.has(cardInt)) {
                btn.classList.add('disabled');
                btn.disabled = true;
            }

            btn.addEventListener('click', () => {
                if (btn.disabled) return;
                onToggle(cardInt);
                btn.classList.toggle('selected');
            });

            row.appendChild(btn);
        }

        container.appendChild(row);
    }

    return container;
}

/**
 * Create recommendation display (probability bars).
 * @param {object[]} recommendations - Array of {bid/card, prob}, sorted by prob descending
 * @param {string} type - "bid" or "play"
 * @param {number} value - State value
 * @param {object} options - { onCardClick: function(cardInt), maxItems: number }
 * @returns {HTMLElement}
 */
function createRecommendationDisplay(recommendations, type, value, options = {}) {
    const maxItems = options.maxItems || 3;
    const onCardClick = options.onCardClick || null;
    // Lighter colors for dark recommendation background
    const REC_COLORS = ['#c0c0c0', '#e85454', '#e85454', '#c0c0c0'];

    const container = document.createElement('div');
    container.className = 'recommendation-display';

    const title = document.createElement('div');
    title.className = 'rec-title';
    title.textContent = type === 'bid' ? 'Bid Recommendations' : 'Play Recommendations';
    container.appendChild(title);

    const topRecs = recommendations.slice(0, maxItems);

    topRecs.forEach((rec, i) => {
        const row = document.createElement('div');
        row.className = 'rec-row';
        if (i === 0) row.classList.add('rec-best');

        const label = document.createElement('span');
        label.className = 'rec-label';
        if (type === 'bid') {
            label.textContent = `Bid ${rec.bid}`;
        } else {
            label.textContent = Card.displayName(rec.card);
            label.style.color = REC_COLORS[Card.suit(rec.card)];
        }

        const barContainer = document.createElement('div');
        barContainer.className = 'rec-bar-container';

        const bar = document.createElement('div');
        bar.className = 'rec-bar';
        bar.style.width = `${rec.prob * 100}%`;

        const pct = document.createElement('span');
        pct.className = 'rec-pct';
        pct.textContent = `${(rec.prob * 100).toFixed(1)}%`;

        barContainer.appendChild(bar);
        row.appendChild(label);
        row.appendChild(barContainer);
        row.appendChild(pct);

        // Make play recommendations clickable
        if (type === 'play' && onCardClick) {
            row.classList.add('rec-clickable');
            row.addEventListener('click', () => onCardClick(rec.card));
        }

        container.appendChild(row);
    });

    const valueDiv = document.createElement('div');
    valueDiv.className = 'rec-value';
    valueDiv.textContent = `State value: ${value.toFixed(3)}`;
    container.appendChild(valueDiv);

    return container;
}

window.Components = {
    createBidPicker,
    createScoreTable,
    createCardPicker,
    createRecommendationDisplay,
};
