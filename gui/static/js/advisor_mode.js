/**
 * Advisor Mode — step-by-step wizard on the green felt table.
 *
 * Phases:
 * 1. select_dealer  — click a player label, confirm
 * 2. select_trump   — 4x13 card grid to pick the face-up trump card
 * 3. enter_hand     — 4x13 card grid, multi-select, confirm
 * 4. enter_bids     — one-by-one in turn order using bid picker
 * 5. playing        — card grid per player in turn order, auto-recommend for user
 */

class AdvisorMode {
    constructor(app) {
        this.app = app;
        this.ws = null;
        this.sessionId = null;
        this.state = null;
        this.numPlayers = 4;
        this.mySeat = 0;
        this.dealer = -1;
        this.phase = 'select_dealer';
        this.renderer = null;
        this._resizeHandler = null;
        this.baselineScale = 1.0;

        // Wizard state
        this.selectedDealer = -1;
        this.selectedTrump = -1;
        this.selectedHand = new Set();
        this.bidOrder = [];       // turn order for bids
        this.bidIndex = 0;        // which player we're bidding for
        this.bids = [];           // recorded bids
        this.playOrder = [];      // turn order for current trick
        this.playIndex = 0;       // which player we're recording a play for
        this.trickLeader = -1;
        this.trickNumber = 0;
        this.tricksWon = [];
        this.usedCards = new Set(); // cards already played (for disabling in picker)
        this.handSize = 0;         // initial hand size (set when hand is confirmed)
        this._savedHand = null;    // snapshot of hand at confirmation (for undo)
        this._actionLog = [];      // debug log of all actions
    }

    // ── Setup form ──────────────────────────────────────────────────────

    renderSetup(container) {
        container.innerHTML = '';

        const form = document.createElement('div');
        form.className = 'setup-form';

        const title = document.createElement('h2');
        title.textContent = 'Advisor Mode \u2014 Real Game Assistant';
        form.appendChild(title);

        const desc = document.createElement('p');
        desc.className = 'setup-desc';
        desc.textContent = 'Mirror a real Trickster Cards game and get AI move recommendations. ' +
            'A step-by-step wizard will guide you through each round.';
        form.appendChild(desc);

        // Player count
        const pcGroup = this._formGroup('Number of Players');
        const pcSelect = document.createElement('select');
        pcSelect.id = 'advisor-num-players';
        [2, 3, 4, 5].forEach(n => {
            const opt = document.createElement('option');
            opt.value = n;
            opt.textContent = `${n} Players`;
            if (n === 4) opt.selected = true;
            pcSelect.appendChild(opt);
        });
        pcGroup.appendChild(pcSelect);
        form.appendChild(pcGroup);

        // Snapshot picker
        const snapGroup = this._formGroup('AI Model (Snapshot) \u2014 optional');
        const snapSelect = document.createElement('select');
        snapSelect.id = 'advisor-snapshot';
        this._populateSnapshotPicker(snapSelect);
        snapGroup.appendChild(snapSelect);
        form.appendChild(snapGroup);

        // Start button
        const startBtn = document.createElement('button');
        startBtn.className = 'btn btn-primary btn-large';
        startBtn.textContent = 'Start Advisor';
        startBtn.addEventListener('click', () => this._startAdvisor());
        form.appendChild(startBtn);

        container.appendChild(form);
    }

    _formGroup(label) {
        const group = document.createElement('div');
        group.className = 'form-group';
        const lbl = document.createElement('label');
        lbl.textContent = label;
        group.appendChild(lbl);
        return group;
    }

    async _populateSnapshotPicker(selectEl) {
        try {
            const resp = await fetch('/api/snapshots');
            const snapshots = await resp.json();
            selectEl.innerHTML = '';
            const defaultOpt = document.createElement('option');
            defaultOpt.value = '';
            defaultOpt.textContent = '-- None (no recommendations) --';
            selectEl.appendChild(defaultOpt);
            snapshots.forEach(s => {
                const opt = document.createElement('option');
                opt.value = s.path;
                opt.textContent = `${s.name} (${s.directory})`;
                selectEl.appendChild(opt);
            });
        } catch (e) {
            console.error('Failed to load snapshots:', e);
        }
    }

    // ── Start session ───────────────────────────────────────────────────

    async _startAdvisor() {
        this.numPlayers = parseInt(document.getElementById('advisor-num-players').value);
        this.mySeat = 0;
        const snapshotPath = document.getElementById('advisor-snapshot').value;

        try {
            const body = { mode: 'advisor', num_players: this.numPlayers, human_seat: 0 };
            if (snapshotPath) body.advisor_snapshot = snapshotPath;

            const resp = await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await resp.json();
            if (data.error) { alert(`Error: ${data.error}`); return; }
            this.sessionId = data.session_id;
        } catch (e) {
            alert(`Failed to create session: ${e.message}`);
            return;
        }

        this._initRoundState();
        this._startTableView();
    }

    _initRoundState() {
        this.selectedDealer = -1;
        this.selectedTrump = -1;
        this.selectedHand = new Set();
        this.bidOrder = [];
        this.bidIndex = 0;
        this.bids = new Array(this.numPlayers).fill(-1);
        this.playOrder = [];
        this.playIndex = 0;
        this.trickLeader = -1;
        this.trickNumber = 0;
        this.tricksWon = new Array(this.numPlayers).fill(0);
        this.usedCards = new Set();
        this.phase = 'select_dealer';
    }

    // ── Table view ──────────────────────────────────────────────────────

    _startTableView() {
        const container = document.getElementById('main-content');
        container.innerHTML = '';
        container.className = 'game-view';

        const tableContainer = document.createElement('div');
        tableContainer.id = 'advisor-table';
        tableContainer.className = 'game-table';
        container.appendChild(tableContainer);

        // Back button
        const backBtn = document.createElement('button');
        backBtn.className = 'btn back-btn';
        backBtn.textContent = 'Back to Setup';
        backBtn.addEventListener('click', () => this._exitAdvisor());
        container.appendChild(backBtn);

        // Scaling
        this._computeAndApplyScale();
        this._resizeHandler = () => this._computeAndApplyScale();
        window.addEventListener('resize', this._resizeHandler);

        // Settings gear
        this._addSettingsGear(container);

        // WebSocket
        const wsUrl = `ws://${window.location.host}/ws/${this.sessionId}`;
        this.ws = new WebSocket(wsUrl);
        this.ws.onmessage = (event) => this._onMessage(JSON.parse(event.data));

        // Start wizard
        this._renderPhase();
    }

    _computeAndApplyScale() {
        const wScale = window.innerWidth / 1200;
        const hScale = window.innerHeight / 800;
        const autoScale = Math.min(wScale, hScale, 1.5);
        const finalScale = Math.max(0.5, autoScale * this.baselineScale);
        Card.Settings.scale = finalScale;
        const s = finalScale;
        document.documentElement.style.setProperty('--card-w', `${Math.round(70 * s)}px`);
        document.documentElement.style.setProperty('--card-h', `${Math.round(100 * s)}px`);
        document.documentElement.style.setProperty('--card-sm-w', `${Math.round(46 * s)}px`);
        document.documentElement.style.setProperty('--card-sm-h', `${Math.round(66 * s)}px`);
        document.documentElement.style.setProperty('--card-font-scale', Card.Settings.fontScale);
        document.documentElement.style.setProperty('--advisor-ui-scale', this.baselineScale);
    }

    _onMessage(msg) {
        this._log(`RECV: type=${msg.type}`);
        if (msg.type === 'advisor_state') {
            this.state = msg.state;
        } else if (msg.type === 'bid_recommendation') {
            this._showRecommendation(msg, 'bid');
        } else if (msg.type === 'play_recommendation') {
            this._showRecommendation(msg, 'play');
        } else if (msg.type === 'error') {
            this._log(`ERROR: ${msg.message}`);
            console.error('Advisor error:', msg.message);
        }
    }

    // ── Phase router ────────────────────────────────────────────────────

    _renderPhase() {
        const table = document.getElementById('advisor-table');
        if (!table) return;
        table.innerHTML = '';
        table.className = 'game-table';

        // Set trump suit for card highlighting
        Card.Settings.trumpSuit = this.selectedTrump >= 0
            ? Math.floor(this.selectedTrump / 13) : -1;

        // Always render player labels around the table
        this._renderPlayerLabels(table);

        switch (this.phase) {
            case 'select_dealer':  this._renderDealerPhase(table); break;
            case 'select_trump':   this._renderTrumpPhase(table); break;
            case 'enter_hand':     this._renderHandPhase(table); break;
            case 'enter_bids':     this._renderBidPhase(table); break;
            case 'playing':        this._renderPlayPhase(table); break;
        }
    }

    // ── Player labels (always visible) ──────────────────────────────────

    _renderPlayerLabels(table) {
        const np = this.numPlayers;
        const layouts = PLAYER_LAYOUTS[np];

        for (let di = 0; di < np; di++) {
            const layout = layouts[di];
            const label = document.createElement('div');
            label.className = 'player-label';
            label.style.left = `${layout.lx}%`;
            label.style.top = `${layout.ly}%`;

            const name = document.createElement('span');
            name.className = 'player-name-text';
            name.textContent = di === 0 ? 'You' : `Player ${di}`;
            label.appendChild(name);

            // Show dealer badge
            if (di === this.selectedDealer) {
                const badge = document.createElement('span');
                badge.className = 'dealer-badge';
                badge.textContent = 'D';
                label.appendChild(badge);
            }

            // Show bid info in play phase
            if (this.phase === 'playing' && this.bids[di] >= 0) {
                const info = document.createElement('span');
                info.className = 'player-bid-info';
                const won = this.tricksWon[di];
                const bid = this.bids[di];
                info.textContent = `${won}/${bid}`;
                if (won === bid) info.classList.add('on-target');
                else if (won > bid) info.classList.add('over-target');
                label.appendChild(info);
            } else if (this.phase === 'enter_bids' && this.bids[di] >= 0) {
                const info = document.createElement('span');
                info.className = 'player-bid-info';
                info.textContent = `bid ${this.bids[di]}`;
                label.appendChild(info);
            }

            // Highlight active player during bid/play phases
            if (this.phase === 'enter_bids' && this.bidIndex < this.numPlayers) {
                if (di === this.bidOrder[this.bidIndex]) {
                    label.classList.add('active-player');
                }
            }
            if (this.phase === 'playing' && this.playIndex < this.numPlayers) {
                if (di === this.playOrder[this.playIndex]) {
                    label.classList.add('active-player');
                }
            }

            label.dataset.seat = di;
            table.appendChild(label);
        }
    }

    // ── Phase 1: Select Dealer ──────────────────────────────────────────

    _renderDealerPhase(table) {
        // Make player labels clickable
        table.querySelectorAll('.player-label').forEach(label => {
            label.classList.add('dealer-selectable');
            label.style.cursor = 'pointer';

            const seat = parseInt(label.dataset.seat);
            if (seat === this.selectedDealer) {
                label.classList.add('active-player');
            }

            label.addEventListener('click', () => {
                this.selectedDealer = seat;
                this._renderPhase(); // re-render to show selection
            });
        });

        // Center instruction
        const instruction = document.createElement('div');
        instruction.className = 'dealer-select-instruction';
        if (this.selectedDealer < 0) {
            instruction.textContent = 'Click the player who is the dealer';
        } else {
            instruction.textContent = `Dealer: ${this.selectedDealer === 0 ? 'You' : 'Player ' + this.selectedDealer}`;
            // Confirm button
            const confirmBtn = document.createElement('button');
            confirmBtn.className = 'btn btn-primary wizard-confirm-btn';
            confirmBtn.textContent = 'Confirm Dealer';
            confirmBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this._send({ action: 'set_dealer', player: this.selectedDealer });
                this.dealer = this.selectedDealer;
                // Compute bid/play order: left of dealer goes first
                this.bidOrder = [];
                for (let i = 1; i <= this.numPlayers; i++) {
                    this.bidOrder.push((this.selectedDealer + i) % this.numPlayers);
                }
                this.trickLeader = this.bidOrder[0];
                this.phase = 'select_trump';
                this._renderPhase();
            });
            instruction.appendChild(document.createElement('br'));
            instruction.appendChild(confirmBtn);
        }
        table.appendChild(instruction);
    }

    // ── Phase 2: Select Trump Card ──────────────────────────────────────

    _renderTrumpPhase(table) {
        // Show trump card near dealer if already selected
        if (this.selectedTrump >= 0) {
            this._renderTrumpIndicator(table);
        }

        const overlay = document.createElement('div');
        overlay.className = 'wizard-overlay';

        const panel = document.createElement('div');
        panel.className = 'wizard-panel wizard-panel-flex';

        // Nav row at top
        panel.appendChild(this._createWizardNav({ back: true, backLabel: 'Back to Dealer' }));

        const title = document.createElement('div');
        title.className = 'wizard-title';
        title.textContent = 'Select the face-up trump card';
        panel.appendChild(title);

        const pickerWrap = document.createElement('div');
        pickerWrap.className = 'wizard-picker-scroll';
        const picker = this._createCardGrid((cardInt) => {
            this.selectedTrump = cardInt;
            // Re-render the picker to show selection
            this._renderPhase();
        }, new Set(this.selectedTrump >= 0 ? [this.selectedTrump] : []), new Set());
        pickerWrap.appendChild(picker);
        panel.appendChild(pickerWrap);

        if (this.selectedTrump >= 0) {
            const selectedInfo = document.createElement('div');
            selectedInfo.className = 'wizard-selected-info';
            selectedInfo.textContent = `Selected: ${Card.displayName(this.selectedTrump)}`;
            panel.appendChild(selectedInfo);

            const confirmBtn = document.createElement('button');
            confirmBtn.className = 'btn btn-primary wizard-confirm-btn';
            confirmBtn.textContent = 'Confirm Trump';
            confirmBtn.addEventListener('click', () => {
                this._send({ action: 'set_trump', card: this.selectedTrump });
                this.phase = 'enter_hand';
                this._renderPhase();
            });
            panel.appendChild(confirmBtn);
        }

        overlay.appendChild(panel);
        table.appendChild(overlay);
    }

    // ── Phase 3: Enter Hand ─────────────────────────────────────────────

    _renderHandPhase(table) {
        this._renderTrumpIndicator(table);

        const overlay = document.createElement('div');
        overlay.className = 'wizard-overlay';

        const panel = document.createElement('div');
        panel.className = 'wizard-panel wizard-panel-flex';

        // Nav row at top
        panel.appendChild(this._createWizardNav({ back: true, backLabel: 'Back to Trump' }));

        const title = document.createElement('div');
        title.className = 'wizard-title';
        title.textContent = 'Select your hand cards';
        panel.appendChild(title);

        const info = document.createElement('div');
        info.className = 'wizard-info';
        info.textContent = `${this.selectedHand.size} cards selected`;
        panel.appendChild(info);

        // Scrollable picker area
        const pickerWrap = document.createElement('div');
        pickerWrap.className = 'wizard-picker-scroll';

        // Disable the trump card (it's face-up, not in anyone's hand)
        const disabled = new Set();
        if (this.selectedTrump >= 0) disabled.add(this.selectedTrump);

        // Confirm button (always present, enabled when cards selected)
        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'btn btn-primary wizard-confirm-btn';
        confirmBtn.textContent = this.selectedHand.size > 0
            ? `Confirm Hand (${this.selectedHand.size} cards)`
            : 'Select cards to continue';
        confirmBtn.disabled = this.selectedHand.size === 0;
        if (this.selectedHand.size === 0) confirmBtn.classList.add('btn-disabled');

        const picker = this._createCardGrid((cardInt) => {
            if (this.selectedHand.has(cardInt)) {
                this.selectedHand.delete(cardInt);
            } else {
                this.selectedHand.add(cardInt);
            }
            info.textContent = `${this.selectedHand.size} cards selected`;
            // Toggle visual selection on the button
            const btn = picker.querySelector(`[data-card="${cardInt}"]`);
            if (btn) btn.classList.toggle('selected');
            // Update confirm button
            confirmBtn.disabled = this.selectedHand.size === 0;
            confirmBtn.classList.toggle('btn-disabled', this.selectedHand.size === 0);
            confirmBtn.textContent = this.selectedHand.size > 0
                ? `Confirm Hand (${this.selectedHand.size} cards)`
                : 'Select cards to continue';
        }, this.selectedHand, disabled);
        pickerWrap.appendChild(picker);
        panel.appendChild(pickerWrap);

        confirmBtn.addEventListener('click', () => {
            if (this.selectedHand.size === 0) return;
            this.handSize = this.selectedHand.size;
            this._savedHand = new Set(this.selectedHand);
            this._send({ action: 'set_hand', cards: Array.from(this.selectedHand) });
            this.phase = 'enter_bids';
            this.bidIndex = 0;
            this._renderPhase();
        });
        panel.appendChild(confirmBtn);

        overlay.appendChild(panel);
        table.appendChild(overlay);
    }

    // ── Phase 4: Enter Bids ─────────────────────────────────────────────

    _renderBidPhase(table) {
        this._renderTrumpIndicator(table);
        this._renderMyHand(table);

        if (this.bidIndex >= this.numPlayers) {
            // All bids entered — move to play phase
            this.phase = 'playing';
            this.playOrder = [];
            for (let i = 0; i < this.numPlayers; i++) {
                this.playOrder.push((this.trickLeader + i) % this.numPlayers);
            }
            this.playIndex = 0;
            this._renderPhase();
            return;
        }

        const currentBidder = this.bidOrder[this.bidIndex];
        const isMe = currentBidder === this.mySeat;
        const handSize = this.handSize;

        // Auto-request recommendation when it's my turn to bid
        if (isMe) {
            this._send({ action: 'get_recommendation', phase: 'bid' });
        }

        // Show bid picker overlay
        const overlay = document.createElement('div');
        overlay.className = 'bid-overlay';

        const panel = document.createElement('div');
        panel.className = 'bid-panel';

        // Nav row
        panel.appendChild(this._createWizardNav({
            back: true, backLabel: 'Edit Hand',
            undo: true, undoLabel: 'Undo Bid',
            undoDisabled: this.bidIndex === 0,
            onUndo: () => this._undoLastBid(),
            skip: true,
        }));

        const title = document.createElement('div');
        title.className = 'bid-title';
        const playerName = currentBidder === 0 ? 'Your' : `Player ${currentBidder}'s`;
        title.textContent = `${playerName} turn to bid`;
        panel.appendChild(title);

        // Recommendation display area (only for my turn)
        if (isMe) {
            const recArea = document.createElement('div');
            recArea.id = 'advisor-rec-area';
            recArea.className = 'advisor-rec-inline';
            panel.appendChild(recArea);

            const recBtn = document.createElement('button');
            recBtn.className = 'btn btn-small wizard-rec-btn';
            recBtn.textContent = 'Get Recommendation';
            recBtn.addEventListener('click', () => {
                this._send({ action: 'get_recommendation', phase: 'bid' });
            });
            panel.appendChild(recBtn);
        }

        const circles = document.createElement('div');
        circles.className = 'bid-circles';

        // Determine legal bids (dealer can't make bids sum to hand_size)
        const isLastBidder = this.bidIndex === this.numPlayers - 1;
        const sumSoFar = this.bids.filter(b => b >= 0).reduce((a, b) => a + b, 0);

        for (let i = 0; i <= handSize; i++) {
            const circle = document.createElement('button');
            circle.className = 'bid-circle';
            circle.textContent = i;

            // Dealer restriction: last bidder can't make total = hand_size
            const illegal = isLastBidder && (sumSoFar + i === handSize);
            if (illegal) {
                circle.classList.add('illegal');
                circle.title = 'Dealer restriction: total bids cannot equal hand size';
            } else {
                circle.classList.add('legal');
                circle.addEventListener('click', () => {
                    this.bids[currentBidder] = i;
                    this._send({ action: 'record_bid', player: currentBidder, value: i });
                    this.bidIndex++;
                    this._renderPhase();
                });
            }
            circles.appendChild(circle);
        }

        panel.appendChild(circles);
        overlay.appendChild(panel);
        table.appendChild(overlay);
    }

    // ── Phase 5: Play ───────────────────────────────────────────────────

    _renderPlayPhase(table) {
        this._renderTrumpIndicator(table);
        this._renderMyHand(table);
        this._renderTrickArea(table);

        // Check if all cards have been played (round over)
        const totalTricks = this.tricksWon.reduce((a, b) => a + b, 0);
        if (totalTricks >= this.handSize) {
            this._renderRoundComplete(table);
            return;
        }

        if (this.playIndex >= this.numPlayers) {
            // Trick complete — resolve it
            this._resolveTrick();
            return;
        }

        const currentPlayer = this.playOrder[this.playIndex];
        const isMe = currentPlayer === this.mySeat;

        // Auto-request recommendation when it's my turn
        if (isMe) {
            this._send({ action: 'get_recommendation', phase: 'play' });
        }

        // Show card picker for the current player
        const overlay = document.createElement('div');
        overlay.className = 'wizard-overlay wizard-overlay-bottom';

        const panel = document.createElement('div');
        panel.className = 'wizard-panel wizard-panel-flex';

        const hasPlays = (this._currentTrickCards || []).length > 0;
        // Nav row
        panel.appendChild(this._createWizardNav({
            back: true, backLabel: 'Edit Hand',
            undo: true, undoLabel: 'Undo Play',
            undoDisabled: !hasPlays,
            onUndo: () => this._undoLastPlay(),
            skip: true,
        }));

        const title = document.createElement('div');
        title.className = 'wizard-title';
        const playerName = currentPlayer === 0 ? 'Your' : `Player ${currentPlayer}'s`;
        title.textContent = `${playerName} turn to play`;
        panel.appendChild(title);

        // Recommendation display (for my turn)
        if (isMe) {
            const recArea = document.createElement('div');
            recArea.id = 'advisor-rec-area';
            recArea.className = 'advisor-rec-inline';
            panel.appendChild(recArea);

            const recBtn = document.createElement('button');
            recBtn.className = 'btn btn-small wizard-rec-btn';
            recBtn.textContent = 'Get Recommendation';
            recBtn.addEventListener('click', () => {
                this._send({ action: 'get_recommendation', phase: 'play' });
            });
            panel.appendChild(recBtn);
        }

        // Card picker — disable cards not available to this player
        const disabled = new Set(this.usedCards);
        if (this.selectedTrump >= 0) disabled.add(this.selectedTrump);
        if (isMe) {
            // My turn: only allow cards in my hand
            for (let c = 0; c < 52; c++) {
                if (!this.selectedHand.has(c)) disabled.add(c);
            }
        } else {
            // Other player's turn: also disable cards still in my hand
            for (const c of this.selectedHand) {
                disabled.add(c);
            }
        }

        const pickerWrap = document.createElement('div');
        pickerWrap.className = 'wizard-picker-scroll';
        const picker = this._createCardGrid((cardInt) => {
            this._recordPlay(currentPlayer, cardInt);
        }, new Set(), disabled);
        pickerWrap.appendChild(picker);
        panel.appendChild(pickerWrap);

        overlay.appendChild(panel);
        table.appendChild(overlay);
    }

    _recordPlay(player, cardInt) {
        this._send({ action: 'record_play', player: player, card: cardInt });
        this.usedCards.add(cardInt);
        if (this.selectedHand.has(cardInt)) {
            this.selectedHand.delete(cardInt);
        }
        // Store trick card for rendering
        if (!this._currentTrickCards) this._currentTrickCards = [];
        this._currentTrickCards.push([player, cardInt]);
        this.playIndex++;
        this._renderPhase();
    }

    _resolveTrick() {
        // Determine winner from current trick cards
        const cards = this._currentTrickCards || [];
        if (cards.length === 0) return;

        const trumpSuit = this.selectedTrump >= 0 ? Math.floor(this.selectedTrump / 13) : -1;
        let bestPlayer = cards[0][0];
        let bestCard = cards[0][1];

        for (let i = 1; i < cards.length; i++) {
            const [player, card] = cards[i];
            const cardSuit = Math.floor(card / 13);
            const bestSuit = Math.floor(bestCard / 13);

            if (cardSuit === trumpSuit && bestSuit !== trumpSuit) {
                bestPlayer = player;
                bestCard = card;
            } else if (cardSuit === bestSuit && (card % 13) > (bestCard % 13)) {
                bestPlayer = player;
                bestCard = card;
            }
        }

        this.tricksWon[bestPlayer]++;
        this.trickNumber++;
        this.trickLeader = bestPlayer;
        this._currentTrickCards = [];

        // Set up next trick play order
        this.playOrder = [];
        for (let i = 0; i < this.numPlayers; i++) {
            this.playOrder.push((this.trickLeader + i) % this.numPlayers);
        }
        this.playIndex = 0;

        // Brief delay to show who won, then re-render
        const table = document.getElementById('advisor-table');
        if (table) {
            const winMsg = document.createElement('div');
            winMsg.className = 'trick-winner-msg';
            const winnerName = bestPlayer === 0 ? 'You' : `Player ${bestPlayer}`;
            winMsg.textContent = `${winnerName} wins the trick!`;
            table.appendChild(winMsg);
            setTimeout(() => {
                this._renderPhase();
            }, 1200);
        }
    }

    _renderTrickArea(table) {
        const cards = this._currentTrickCards || [];
        if (cards.length === 0) return;

        const np = this.numPlayers;
        const layouts = PLAYER_LAYOUTS[np];

        cards.forEach(([player, cardInt]) => {
            const di = player; // In advisor, seat === displayIdx
            const layout = layouts[di];
            const cardEl = Card.create(cardInt, true, 'normal');
            cardEl.classList.add('trick-card');

            // Place card along the line from center (50%,50%) toward player's seat
            const cx = 50 + TRICK_BIAS * (layout.lx - 50);
            const cy = 50 + TRICK_BIAS * (layout.ly - 50);
            cardEl.style.position = 'absolute';
            cardEl.style.left = `${cx}%`;
            cardEl.style.top = `${cy}%`;
            cardEl.style.transform = 'translate(-50%, -50%)';
            cardEl.style.zIndex = 15 + (np - di);

            table.appendChild(cardEl);
        });
    }

    _renderRoundComplete(table) {
        const overlay = document.createElement('div');
        overlay.className = 'wizard-overlay';

        const panel = document.createElement('div');
        panel.className = 'wizard-panel';

        const title = document.createElement('div');
        title.className = 'wizard-title';
        title.textContent = 'Round Complete';
        panel.appendChild(title);

        // Score summary
        const scoreDiv = document.createElement('div');
        scoreDiv.className = 'wizard-score-summary';

        for (let i = 0; i < this.numPlayers; i++) {
            const row = document.createElement('div');
            row.className = 'wizard-score-row';
            const hit = this.bids[i] === this.tricksWon[i];
            const roundScore = hit ? 10 + this.bids[i] : 0;

            const name = document.createElement('span');
            name.textContent = i === 0 ? 'You' : `Player ${i}`;
            name.style.fontWeight = '600';
            row.appendChild(name);

            const detail = document.createElement('span');
            detail.textContent = `Bid ${this.bids[i]}, Won ${this.tricksWon[i]} \u2192 ${hit ? '+' + roundScore : '+0'}`;
            detail.style.color = hit ? '#81c784' : '#ef9a9a';
            row.appendChild(detail);

            scoreDiv.appendChild(row);
        }
        panel.appendChild(scoreDiv);

        // Score round and start new round button
        const btnRow = document.createElement('div');
        btnRow.style.cssText = 'display: flex; gap: 10px; margin-top: 14px; justify-content: center;';

        const scoreBtn = document.createElement('button');
        scoreBtn.className = 'btn btn-primary';
        scoreBtn.textContent = 'Score & Next Round';
        scoreBtn.addEventListener('click', () => {
            this._send({ action: 'score_round' });
            this._send({ action: 'new_round' });
            // Advance dealer
            const newDealer = (this.selectedDealer + 1) % this.numPlayers;
            this._initRoundState();
            this.selectedDealer = newDealer;
            this.dealer = newDealer;
            this._send({ action: 'set_dealer', player: newDealer });
            this.bidOrder = [];
            for (let i = 1; i <= this.numPlayers; i++) {
                this.bidOrder.push((newDealer + i) % this.numPlayers);
            }
            this.trickLeader = this.bidOrder[0];
            this.phase = 'select_trump';
            this._renderPhase();
        });
        btnRow.appendChild(scoreBtn);

        const exitBtn = document.createElement('button');
        exitBtn.className = 'btn';
        exitBtn.textContent = 'Back to Setup';
        exitBtn.addEventListener('click', () => this._exitAdvisor());
        btnRow.appendChild(exitBtn);

        panel.appendChild(btnRow);
        overlay.appendChild(panel);
        table.appendChild(overlay);
    }

    // ── Shared rendering helpers ────────────────────────────────────────

    _renderTrumpIndicator(table) {
        if (this.selectedTrump < 0) return;

        const np = this.numPlayers;
        const layouts = PLAYER_LAYOUTS[np];
        const dealerLayout = layouts[this.selectedDealer >= 0 ? this.selectedDealer : 0];

        const trumpDiv = document.createElement('div');
        trumpDiv.className = 'trump-indicator';

        const cx = 50, cy = 50;
        const dx = dealerLayout.lx - cx;
        const dy = dealerLayout.ly - cy;
        const tx = dealerLayout.lx - dx * 0.35;
        const ty = dealerLayout.ly - dy * 0.35;

        trumpDiv.style.left = `${tx}%`;
        trumpDiv.style.top = `${ty}%`;

        trumpDiv.appendChild(Card.create(this.selectedTrump, true, 'small'));

        const label = document.createElement('div');
        label.className = 'trump-label';
        label.textContent = 'Trump';
        trumpDiv.appendChild(label);

        table.appendChild(trumpDiv);
    }

    _renderMyHand(table) {
        if (this.selectedHand.size === 0) return;

        const handDiv = document.createElement('div');
        handDiv.className = 'human-hand';

        const trumpSuit = this.selectedTrump >= 0 ? Math.floor(this.selectedTrump / 13) : -1;
        const cards = Array.from(this.selectedHand).sort((a, b) => {
            const aSuit = Math.floor(a / 13);
            const bSuit = Math.floor(b / 13);
            const aIsTrump = aSuit === trumpSuit ? 1 : 0;
            const bIsTrump = bSuit === trumpSuit ? 1 : 0;
            if (aIsTrump !== bIsTrump) return aIsTrump - bIsTrump;
            if (aSuit !== bSuit) return aSuit - bSuit;
            return (a % 13) - (b % 13);
        });

        const n = cards.length;
        if (n === 0) return;

        const cardW = 70 * Card.Settings.scale;
        const overlap = 22 * Card.Settings.scale;
        const totalWidth = cardW + (n - 1) * overlap;
        const maxArc = Math.min(n * 2.5, 25);

        cards.forEach((cardInt, i) => {
            const cardEl = Card.create(cardInt, true, 'normal');
            cardEl.classList.add('hand-card');

            const offsetX = i * overlap - totalWidth / 2 + cardW / 2;
            const t = n > 1 ? (i / (n - 1)) - 0.5 : 0;
            const rotation = t * maxArc;
            const lift = -Math.abs(t) * 12 * Card.Settings.scale;

            cardEl.style.setProperty('--fan-offset-x', `${offsetX}px`);
            cardEl.style.setProperty('--fan-lift', `${lift}px`);
            cardEl.style.setProperty('--fan-rotation', `${rotation}deg`);
            cardEl.style.zIndex = i + 1;

            handDiv.appendChild(cardEl);
        });

        table.appendChild(handDiv);
    }

    _createCardGrid(onClick, selected, disabled) {
        // Lighter suit colors for dark picker background
        const PICKER_COLORS = ['#c0c0c0', '#e85454', '#e85454', '#c0c0c0'];
        const container = document.createElement('div');
        container.className = 'card-picker';

        for (let suit = 0; suit < 4; suit++) {
            const row = document.createElement('div');
            row.className = 'card-picker-row';

            const suitLabel = document.createElement('span');
            suitLabel.className = 'card-picker-suit';
            suitLabel.textContent = Card.SUIT_SYMBOLS[suit];
            suitLabel.style.color = PICKER_COLORS[suit];
            row.appendChild(suitLabel);

            for (let rank = 0; rank < 13; rank++) {
                const cardInt = suit * 13 + rank;
                const btn = document.createElement('button');
                btn.className = 'card-picker-btn';
                btn.textContent = Card.RANK_DISPLAY[rank];
                btn.style.color = PICKER_COLORS[suit];
                btn.dataset.card = cardInt;

                if (selected.has(cardInt)) btn.classList.add('selected');
                if (disabled.has(cardInt)) {
                    btn.classList.add('disabled');
                    btn.disabled = true;
                }

                btn.addEventListener('click', () => {
                    if (!btn.disabled) onClick(cardInt);
                });

                row.appendChild(btn);
            }
            container.appendChild(row);
        }
        return container;
    }

    // ── Recommendation display ──────────────────────────────────────────

    _showRecommendation(msg, type) {
        const area = document.getElementById('advisor-rec-area');
        if (!area) return;
        area.innerHTML = '';

        const options = { maxItems: 3 };

        // For play recommendations, make cards clickable to play them
        if (type === 'play' && this.phase === 'playing') {
            const currentPlayer = this.playOrder[this.playIndex];
            if (currentPlayer === this.mySeat) {
                options.onCardClick = (cardInt) => {
                    this._recordPlay(currentPlayer, cardInt);
                };
            }
        }

        const recEl = Components.createRecommendationDisplay(
            msg.recommendations, type, msg.value, options
        );
        area.appendChild(recEl);
    }

    // ── WebSocket send ──────────────────────────────────────────────────

    _send(msg) {
        this._log(`SEND: ${JSON.stringify(msg)}`);
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(msg));
        }
    }

    _log(text) {
        const ts = new Date().toLocaleTimeString();
        this._actionLog.push(`[${ts}] ${text}`);
    }

    // ── Navigation (back / undo / skip) ────────────────────────────────

    _goBack() {
        this._log(`GO_BACK from ${this.phase}`);
        switch (this.phase) {
            case 'select_trump':
                this.selectedTrump = -1;
                this.phase = 'select_dealer';
                break;
            case 'enter_hand':
                this.selectedHand = new Set();
                this.phase = 'select_trump';
                break;
            case 'enter_bids':
            case 'playing':
                // Restore hand from saved copy and restart from hand selection
                if (this._savedHand) {
                    this.selectedHand = new Set(this._savedHand);
                }
                this.bids = new Array(this.numPlayers).fill(-1);
                this.bidIndex = 0;
                this.tricksWon = new Array(this.numPlayers).fill(0);
                this.usedCards = new Set();
                this._currentTrickCards = [];
                this.playIndex = 0;
                this.trickNumber = 0;
                this.phase = 'enter_hand';
                break;
        }
        this._renderPhase();
    }

    _skipRound() {
        this._log('SKIP_ROUND');
        const newDealer = (this.selectedDealer + 1) % this.numPlayers;
        this._initRoundState();
        this.selectedDealer = newDealer;
        this.dealer = newDealer;
        this._send({ action: 'new_round' });
        this._send({ action: 'set_dealer', player: newDealer });
        this.bidOrder = [];
        for (let i = 1; i <= this.numPlayers; i++) {
            this.bidOrder.push((newDealer + i) % this.numPlayers);
        }
        this.trickLeader = this.bidOrder[0];
        this.phase = 'select_trump';
        this._renderPhase();
    }

    _undoLastBid() {
        if (this.bidIndex <= 0) return;
        this.bidIndex--;
        const prevBidder = this.bidOrder[this.bidIndex];
        this._log(`UNDO_BID player=${prevBidder} was=${this.bids[prevBidder]}`);
        this.bids[prevBidder] = -1;
        this._renderPhase();
    }

    _undoLastPlay() {
        const cards = this._currentTrickCards || [];
        if (cards.length === 0) return;
        const [player, cardInt] = cards.pop();
        this._log(`UNDO_PLAY player=${player} card=${Card.displayName(cardInt)}`);
        this.usedCards.delete(cardInt);
        // Restore to player's hand if it was my card
        if (player === this.mySeat && this._savedHand && this._savedHand.has(cardInt)) {
            this.selectedHand.add(cardInt);
        }
        this.playIndex--;
        this._renderPhase();
    }

    _createWizardNav(options = {}) {
        const nav = document.createElement('div');
        nav.className = 'wizard-nav';

        if (options.back) {
            const backBtn = document.createElement('button');
            backBtn.className = 'btn btn-small wizard-nav-btn';
            backBtn.textContent = options.backLabel || 'Back';
            backBtn.addEventListener('click', () => this._goBack());
            nav.appendChild(backBtn);
        }

        if (options.undo) {
            const undoBtn = document.createElement('button');
            undoBtn.className = 'btn btn-small wizard-nav-btn';
            undoBtn.textContent = options.undoLabel || 'Undo';
            if (options.undoDisabled) {
                undoBtn.disabled = true;
                undoBtn.classList.add('btn-disabled');
            }
            undoBtn.addEventListener('click', () => {
                if (options.onUndo) options.onUndo();
            });
            nav.appendChild(undoBtn);
        }

        if (options.skip) {
            const skipBtn = document.createElement('button');
            skipBtn.className = 'btn btn-small wizard-nav-btn wizard-nav-skip';
            skipBtn.textContent = 'Skip Round';
            skipBtn.addEventListener('click', () => this._skipRound());
            nav.appendChild(skipBtn);
        }

        return nav;
    }

    // ── Settings gear ──────────────────────────────────────────────────

    _addSettingsGear(container) {
        const gear = document.createElement('button');
        gear.className = 'settings-gear';
        gear.innerHTML = '\u2699';
        gear.title = 'Settings';

        let panel = null;

        gear.addEventListener('click', () => {
            if (panel) {
                panel.remove();
                panel = null;
                return;
            }

            panel = document.createElement('div');
            panel.className = 'settings-panel';

            const title = document.createElement('h4');
            title.textContent = 'Display Settings';
            panel.appendChild(title);

            // Card style
            const styleRow = document.createElement('div');
            styleRow.className = 'settings-row';
            const styleLabel = document.createElement('label');
            styleLabel.textContent = 'Card Style';
            const styleSelect = document.createElement('select');
            [['css-large', 'Large (clear)'], ['css', 'Small (clear)'], ['svg', 'Classic (artwork)']].forEach(([v, label]) => {
                const opt = document.createElement('option');
                opt.value = v;
                opt.textContent = label;
                if (Card.Settings.cardStyle === v) opt.selected = true;
                styleSelect.appendChild(opt);
            });
            styleSelect.addEventListener('change', () => {
                Card.Settings.cardStyle = styleSelect.value;
                this._renderPhase();
            });
            styleRow.appendChild(styleLabel);
            styleRow.appendChild(styleSelect);
            panel.appendChild(styleRow);

            // Font scale
            const fontRow = document.createElement('div');
            fontRow.className = 'settings-row';
            const fontLabel = document.createElement('label');
            fontLabel.textContent = 'Text Size';
            const fontSlider = document.createElement('input');
            fontSlider.type = 'range';
            fontSlider.min = '0.6';
            fontSlider.max = '1.6';
            fontSlider.step = '0.1';
            fontSlider.value = String(Card.Settings.fontScale);
            const fontVal = document.createElement('span');
            fontVal.className = 'settings-value';
            fontVal.textContent = `${Card.Settings.fontScale.toFixed(1)}x`;
            fontSlider.addEventListener('input', () => {
                Card.Settings.fontScale = parseFloat(fontSlider.value);
                fontVal.textContent = `${Card.Settings.fontScale.toFixed(1)}x`;
                document.documentElement.style.setProperty('--card-font-scale', Card.Settings.fontScale);
                this._renderPhase();
            });
            fontRow.appendChild(fontLabel);
            fontRow.appendChild(fontSlider);
            fontRow.appendChild(fontVal);
            panel.appendChild(fontRow);

            // GUI scale
            const scaleRow = document.createElement('div');
            scaleRow.className = 'settings-row';
            const scaleLabel = document.createElement('label');
            scaleLabel.textContent = 'GUI Scale';
            const scaleSlider = document.createElement('input');
            scaleSlider.type = 'range';
            scaleSlider.min = '0.5';
            scaleSlider.max = '2.0';
            scaleSlider.step = '0.1';
            scaleSlider.value = String(this.baselineScale);
            const scaleVal = document.createElement('span');
            scaleVal.className = 'settings-value';
            scaleVal.textContent = `${this.baselineScale.toFixed(1)}x`;
            scaleSlider.addEventListener('input', () => {
                this.baselineScale = parseFloat(scaleSlider.value);
                scaleVal.textContent = `${this.baselineScale.toFixed(1)}x`;
                this._computeAndApplyScale();
                this._renderPhase();
            });
            scaleRow.appendChild(scaleLabel);
            scaleRow.appendChild(scaleSlider);
            scaleRow.appendChild(scaleVal);
            panel.appendChild(scaleRow);

            // Debug log
            const logBtn = document.createElement('button');
            logBtn.className = 'btn btn-small';
            logBtn.textContent = 'Copy Debug Log';
            logBtn.style.cssText = 'margin-top: 10px; width: 100%; font-size: 11px;';
            logBtn.addEventListener('click', () => {
                const logText = this._actionLog.join('\n');
                navigator.clipboard.writeText(logText).then(() => {
                    logBtn.textContent = 'Copied!';
                    setTimeout(() => { logBtn.textContent = 'Copy Debug Log'; }, 1500);
                }).catch(() => {
                    // Fallback: show in a prompt
                    prompt('Debug log (copy from here):', logText);
                });
            });
            panel.appendChild(logBtn);

            container.appendChild(panel);
        });

        container.appendChild(gear);
    }

    // ── Cleanup ─────────────────────────────────────────────────────────

    _exitAdvisor() {
        if (this._resizeHandler) {
            window.removeEventListener('resize', this._resizeHandler);
            this._resizeHandler = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        if (this.sessionId) {
            fetch(`/api/sessions/${this.sessionId}`, { method: 'DELETE' });
            this.sessionId = null;
        }
        this.selectedHand.clear();
        this.app.showSetup();
    }
}

window.AdvisorMode = AdvisorMode;
