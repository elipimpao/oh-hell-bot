/**
 * Game table renderer — mimics Trickster Cards layout.
 *
 * - Human at bottom, cards fanned in an arc
 * - Top players: cards fanned horizontally, overlapping
 * - Left/right players: cards rotated 90°, fanned vertically
 * - Player labels between cards and table center
 * - Trump card near the dealer
 * - Trick cards scattered in center
 */

// Player position configs: where the label sits, and where cards fan from
// handAnchor is where the card fan center is; labelAnchor is the name plate
const PLAYER_LAYOUTS = {
    2: [
        { lx: 50, ly: 87, hx: 50, hy: 96, dir: 'bottom' },
        { lx: 50, ly: 13, hx: 50, hy: 5,  dir: 'top' },
    ],
    3: [
        { lx: 50, ly: 87, hx: 50, hy: 96, dir: 'bottom' },
        { lx: 13, ly: 17, hx: 5,  hy: 17, dir: 'left' },
        { lx: 87, ly: 17, hx: 95, hy: 17, dir: 'right' },
    ],
    4: [
        { lx: 50, ly: 87, hx: 50, hy: 96, dir: 'bottom' },
        { lx: 13, ly: 50, hx: 4,  hy: 50, dir: 'left' },
        { lx: 50, ly: 13, hx: 50, hy: 5,  dir: 'top' },
        { lx: 87, ly: 50, hx: 96, hy: 50, dir: 'right' },
    ],
    5: [
        { lx: 50, ly: 87, hx: 50, hy: 96, dir: 'bottom' },
        { lx: 13, ly: 55, hx: 4,  hy: 55, dir: 'left' },
        { lx: 22, ly: 13, hx: 18, hy: 5,  dir: 'top' },
        { lx: 78, ly: 13, hx: 82, hy: 5,  dir: 'top' },
        { lx: 87, ly: 55, hx: 96, hy: 55, dir: 'right' },
    ],
};

// How far from center toward the player's label to place trick cards.
// 0 = dead center, 1 = at player label. 0.30 = 30% of the way toward player.
const TRICK_BIAS = 0.30;

class GameRenderer {
    constructor(container) {
        this.container = container;
        this.state = null;
        this.seatMapping = null;
        this.onCardClick = null;
    }

    setSeatMapping(humanSeat, numPlayers) {
        this.seatMapping = [];
        for (let i = 0; i < numPlayers; i++) {
            this.seatMapping.push((humanSeat + i) % numPlayers);
        }
    }

    displayPos(actualSeat) {
        return this.seatMapping.indexOf(actualSeat);
    }

    render(state) {
        this.state = state;
        this.container.innerHTML = '';
        this.container.className = 'game-table';

        if (!state || state.game_over) {
            if (state && state.game_over) this._renderGameOver(state);
            return;
        }

        const np = state.num_players;
        const layouts = PLAYER_LAYOUTS[np];

        this._renderRoundInfo(state);

        // Render each player: cards first, then label on top
        for (let di = 0; di < np; di++) {
            const seat = this.seatMapping[di];
            const layout = layouts[di];

            if (di === 0) {
                this._renderHumanHand(seat, state);
            } else {
                this._renderOpponentHand(seat, di, layout, state);
            }
            this._renderPlayerLabel(seat, di, layout, state);
        }

        // Trump card near dealer
        this._renderTrumpCard(state, np, layouts);

        // Center trick
        this._renderTrickArea(state, np);
    }

    // ── Player labels ──────────────────────────────────────────────────

    _renderPlayerLabel(seat, displayIdx, layout, state) {
        const label = document.createElement('div');
        label.className = 'player-label';
        label.style.left = `${layout.lx}%`;
        label.style.top = `${layout.ly}%`;

        if (seat === state.current_player) label.classList.add('active-player');

        // Name
        const name = document.createElement('span');
        name.className = 'player-name-text';
        name.textContent = state.seat_labels[seat];
        label.appendChild(name);

        // Dealer badge
        if (seat === state.dealer) {
            const badge = document.createElement('span');
            badge.className = 'dealer-badge';
            badge.textContent = 'D';
            label.appendChild(badge);
        }

        // Bid / tricks info on same line
        if (state.bids[seat] >= 0) {
            const info = document.createElement('span');
            info.className = 'player-bid-info';
            if (state.phase === 1) {
                const won = state.tricks_won[seat];
                const bid = state.bids[seat];
                info.textContent = `${won}/${bid}`;
                if (won === bid) info.classList.add('on-target');
                else if (won > bid) info.classList.add('over-target');
            } else {
                info.textContent = `bid ${state.bids[seat]}`;
            }
            label.appendChild(info);
        }

        this.container.appendChild(label);
    }

    // ── Human hand (bottom) ────────────────────────────────────────────

    _renderHumanHand(seat, state) {
        const handDiv = document.createElement('div');
        handDiv.className = 'human-hand';

        const rawCards = state.hands[String(seat)] || [];
        const trumpSuit = state.trump_suit;
        // Sort by suit in rotating order: H(2)→C(0)→D(1)→S(3), with trump rightmost.
        // Build suit order: start after trump in the cycle, so trump ends up last.
        const SUIT_CYCLE = [2, 0, 1, 3]; // Hearts, Clubs, Diamonds, Spades
        const trumpIdx = SUIT_CYCLE.indexOf(trumpSuit);
        const suitOrder = {};
        for (let i = 0; i < 4; i++) {
            const suit = SUIT_CYCLE[(trumpIdx + 1 + i) % 4];
            suitOrder[suit] = i; // 0-2 = non-trump in cycle order, 3 = trump
        }
        const cards = rawCards.slice().sort((a, b) => {
            const aSuit = Math.floor(a / 13);
            const bSuit = Math.floor(b / 13);
            const aOrder = suitOrder[aSuit];
            const bOrder = suitOrder[bSuit];
            if (aOrder !== bOrder) return aOrder - bOrder;
            return (a % 13) - (b % 13);
        });
        const legalPlays = state.legal_plays || [];
        const isMyTurn = state.current_player === seat && state.phase === 1;
        const n = cards.length;
        if (n === 0) return;

        // Fan parameters: overlap so only ~20px of each card is exposed
        const cardW = 70 * Card.Settings.scale;
        const overlap = 22 * Card.Settings.scale;
        const totalWidth = cardW + (n - 1) * overlap;
        const maxArc = Math.min(n * 2.5, 25); // total arc in degrees

        cards.forEach((cardInt, i) => {
            const cardEl = Card.create(cardInt, true, 'normal');
            cardEl.classList.add('hand-card');

            const offsetX = i * overlap - totalWidth / 2 + cardW / 2;
            const t = n > 1 ? (i / (n - 1)) - 0.5 : 0;  // -0.5 to 0.5
            const rotation = t * maxArc;
            const lift = -Math.abs(t) * 12 * Card.Settings.scale;  // arc lift

            // Use CSS custom properties so hover can add lift without losing position
            cardEl.style.setProperty('--fan-offset-x', `${offsetX}px`);
            cardEl.style.setProperty('--fan-lift', `${lift}px`);
            cardEl.style.setProperty('--fan-rotation', `${rotation}deg`);
            cardEl.style.zIndex = i + 1;

            if (isMyTurn) {
                if (legalPlays.includes(cardInt)) {
                    cardEl.classList.add('legal');
                    cardEl.addEventListener('click', () => {
                        if (this.onCardClick) this.onCardClick(cardInt);
                    });
                } else {
                    cardEl.classList.add('illegal');
                }
            }

            handDiv.appendChild(cardEl);
        });

        this.container.appendChild(handDiv);
    }

    // ── Opponent hands ─────────────────────────────────────────────────

    _renderOpponentHand(seat, displayIdx, layout, state) {
        const showCards = state.hands && state.hands[String(seat)];
        const count = state.hand_counts ? (state.hand_counts[String(seat)] || 0) : 0;
        const cards = showCards || [];
        const n = showCards ? cards.length : count;
        if (n === 0) return;

        const dir = layout.dir;
        const handDiv = document.createElement('div');
        handDiv.className = `opponent-hand opponent-hand-${dir}`;
        handDiv.style.left = `${layout.hx}%`;
        handDiv.style.top = `${layout.hy}%`;

        const isVertical = (dir === 'left' || dir === 'right');
        const s = Card.Settings.scale;
        // Adaptive overlap: shrink as card count grows to keep hands on-screen.
        // Max span: ~40% of viewport in each direction.
        const cardDim = isVertical ? 66 * s : 46 * s; // small card height/width
        const maxSpan = isVertical
            ? window.innerHeight * 0.38
            : window.innerWidth * 0.28;
        const baseOverlap = isVertical ? 16 * s : 18 * s;
        const maxOverlap = n > 1 ? Math.min(baseOverlap, (maxSpan - cardDim) / (n - 1)) : baseOverlap;
        const overlap = Math.max(6 * s, maxOverlap);
        const fanAngle = isVertical ? 2 : 2.5;

        for (let i = 0; i < n; i++) {
            const cardEl = showCards
                ? Card.create(cards[i], true, 'small')
                : Card.createBack('small');

            const t = n > 1 ? (i / (n - 1)) - 0.5 : 0;
            const offset = i * overlap;
            let rot = t * fanAngle * n * 0.5;

            if (isVertical) {
                const baseRot = dir === 'left' ? 90 : -90;
                cardEl.style.transform = `translateY(${offset - (n - 1) * overlap / 2}px) rotate(${baseRot + rot}deg)`;
            } else {
                cardEl.style.transform = `translateX(${offset - (n - 1) * overlap / 2}px) rotate(${rot}deg)`;
            }

            cardEl.style.zIndex = i + 1;
            handDiv.appendChild(cardEl);
        }

        this.container.appendChild(handDiv);
    }

    // ── Trump card ─────────────────────────────────────────────────────

    _renderTrumpCard(state, np, layouts) {
        if (state.trump_card < 0) return;

        // Place trump near the dealer's position
        const dealerDisplay = this.displayPos(state.dealer);
        const dealerLayout = layouts[dealerDisplay];

        const trumpDiv = document.createElement('div');
        trumpDiv.className = 'trump-indicator';

        // Offset toward center from dealer
        const cx = 50, cy = 50;
        const dx = dealerLayout.lx - cx;
        const dy = dealerLayout.ly - cy;
        const tx = dealerLayout.lx - dx * 0.35;
        const ty = dealerLayout.ly - dy * 0.35;

        trumpDiv.style.left = `${tx}%`;
        trumpDiv.style.top = `${ty}%`;

        const cardEl = Card.create(state.trump_card, true, 'small');
        trumpDiv.appendChild(cardEl);

        const label = document.createElement('div');
        label.className = 'trump-label';
        label.textContent = 'Trump';
        trumpDiv.appendChild(label);

        this.container.appendChild(trumpDiv);
    }

    // ── Round info ─────────────────────────────────────────────────────

    _renderRoundInfo(state) {
        const info = document.createElement('div');
        info.className = 'round-info';

        const round = document.createElement('div');
        round.textContent = `Round ${state.round_index + 1}/${state.num_rounds} \u2022 ${state.hand_size} cards`;
        info.appendChild(round);

        const scores = document.createElement('div');
        scores.className = 'scores-panel';
        const scoreTitle = document.createElement('div');
        scoreTitle.className = 'scores-title';
        scoreTitle.textContent = 'Scores';
        scores.appendChild(scoreTitle);

        for (let di = 0; di < state.num_players; di++) {
            const seat = this.seatMapping[di];
            const row = document.createElement('div');
            row.className = 'score-row';
            const name = document.createElement('span');
            name.className = 'score-name';
            name.textContent = state.seat_labels[seat];
            row.appendChild(name);
            const val = document.createElement('span');
            val.className = 'score-val';
            val.textContent = state.scores[seat];
            row.appendChild(val);
            scores.appendChild(row);
        }
        info.appendChild(scores);
        this.container.appendChild(info);
    }

    // ── Trick area ─────────────────────────────────────────────────────

    _renderTrickArea(state, np) {
        if (state.current_trick.length === 0) return;

        const layouts = PLAYER_LAYOUTS[np];

        state.current_trick.forEach(([player, cardInt]) => {
            const di = this.displayPos(player);
            const layout = layouts[di];

            const cardEl = Card.create(cardInt, true, 'normal');
            cardEl.classList.add('trick-card');

            // Place card along the line from center (50%,50%) toward player's seat.
            // position:absolute set inline to override .card's position:relative.
            const cx = 50 + TRICK_BIAS * (layout.lx - 50);
            const cy = 50 + TRICK_BIAS * (layout.ly - 50);
            cardEl.style.position = 'absolute';
            cardEl.style.left = `${cx}%`;
            cardEl.style.top = `${cy}%`;
            cardEl.style.transform = 'translate(-50%, -50%)';

            // Fixed z-index by display position — bottom card always on top
            cardEl.style.zIndex = 15 + (np - di);

            this.container.appendChild(cardEl);
        });
    }

    // ── Game over ──────────────────────────────────────────────────────

    _renderGameOver(state) {
        const msg = document.createElement('div');
        msg.className = 'game-over-msg';
        msg.textContent = 'Game Over';
        this.container.appendChild(msg);
    }
}

window.GameRenderer = GameRenderer;
