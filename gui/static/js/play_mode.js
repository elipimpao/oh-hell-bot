/**
 * Play Mode — setup form and in-game interaction for Mode 1.
 */

class PlayMode {
    constructor(app) {
        this.app = app;
        this.ws = null;
        this.renderer = null;
        this.state = null;
        this.sessionId = null;
        this.animationSpeed = 500; // ms between bot actions
        this.trickDisplayDuration = 1500; // ms to show completed trick
        this.baselineScale = 1.0; // user-set baseline multiplier
        this.eventQueue = [];
        this.animating = false;
        this.spectatorInterval = null;
        this.isSpectatorMode = false;
        this.spectatorPaused = false;
        this._resizeHandler = null;
    }

    /**
     * Render the setup form.
     */
    renderSetup(container) {
        container.innerHTML = '';

        const form = document.createElement('div');
        form.className = 'setup-form';

        const title = document.createElement('h2');
        title.textContent = 'Play Mode — Game Setup';
        form.appendChild(title);

        // Player count
        const pcGroup = this._formGroup('Number of Players');
        const pcSelect = document.createElement('select');
        pcSelect.id = 'num-players';
        [2, 3, 4, 5].forEach(n => {
            const opt = document.createElement('option');
            opt.value = n;
            opt.textContent = `${n} Players`;
            if (n === 4) opt.selected = true;
            pcSelect.appendChild(opt);
        });
        pcSelect.addEventListener('change', () => this._updateSeatConfig(form));
        pcGroup.appendChild(pcSelect);
        form.appendChild(pcGroup);

        // Seat configuration
        const seatSection = document.createElement('div');
        seatSection.id = 'seat-config';
        form.appendChild(seatSection);

        // Max cards setting
        const cardsGroup = this._formGroup('Max Cards Per Hand');
        const cardsSelect = document.createElement('select');
        cardsSelect.id = 'max-cards';
        const defaultOpt = document.createElement('option');
        defaultOpt.value = '';
        defaultOpt.textContent = 'Default (auto)';
        cardsSelect.appendChild(defaultOpt);
        for (let i = 1; i <= 12; i++) {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = `${i} card${i > 1 ? 's' : ''}`;
            cardsSelect.appendChild(opt);
        }
        cardsGroup.appendChild(cardsSelect);
        form.appendChild(cardsGroup);

        // Options
        const optGroup = document.createElement('div');
        optGroup.className = 'options-group';

        const devLabel = document.createElement('label');
        devLabel.className = 'checkbox-label';
        const devCheck = document.createElement('input');
        devCheck.type = 'checkbox';
        devCheck.id = 'dev-mode';
        devLabel.appendChild(devCheck);
        devLabel.appendChild(document.createTextNode(' Dev Mode (all cards face-up)'));
        optGroup.appendChild(devLabel);

        // Speed control
        const speedGroup = this._formGroup('Animation Speed');
        const speedSlider = document.createElement('input');
        speedSlider.type = 'range';
        speedSlider.id = 'anim-speed';
        speedSlider.min = 0;
        speedSlider.max = 2000;
        speedSlider.value = 500;
        speedSlider.step = 100;
        const speedLabel = document.createElement('span');
        speedLabel.id = 'speed-label';
        speedLabel.textContent = '500ms';
        speedSlider.addEventListener('input', () => {
            speedLabel.textContent = `${speedSlider.value}ms`;
            this.animationSpeed = parseInt(speedSlider.value);
        });
        speedGroup.appendChild(speedSlider);
        speedGroup.appendChild(speedLabel);
        optGroup.appendChild(speedGroup);

        form.appendChild(optGroup);

        // Start button
        const startBtn = document.createElement('button');
        startBtn.className = 'btn btn-primary btn-large';
        startBtn.textContent = 'Start Game';
        startBtn.addEventListener('click', () => this._startGame());
        form.appendChild(startBtn);

        container.appendChild(form);

        // Initialize seat config
        this._updateSeatConfig(form);
    }

    _formGroup(label) {
        const group = document.createElement('div');
        group.className = 'form-group';
        const lbl = document.createElement('label');
        lbl.textContent = label;
        group.appendChild(lbl);
        return group;
    }

    _updateSeatConfig(form) {
        const numPlayers = parseInt(document.getElementById('num-players').value);
        const seatConfig = document.getElementById('seat-config');
        seatConfig.innerHTML = '';

        const title = document.createElement('h3');
        title.textContent = 'Seat Configuration';
        seatConfig.appendChild(title);

        for (let i = 0; i < numPlayers; i++) {
            const row = document.createElement('div');
            row.className = 'seat-row';

            const label = document.createElement('span');
            label.className = 'seat-label';
            label.textContent = `Seat ${i}:`;
            row.appendChild(label);

            const select = document.createElement('select');
            select.className = 'seat-type';
            select.dataset.seat = i;

            const types = [
                { value: 'human', text: 'Human' },
                { value: 'smart', text: 'SmartBot' },
                { value: 'heuristic', text: 'HeuristicBot' },
                { value: 'random', text: 'RandomBot' },
                { value: 'nn', text: 'NN Snapshot' },
            ];

            types.forEach(t => {
                const opt = document.createElement('option');
                opt.value = t.value;
                opt.textContent = t.text;
                if (i === 0 && t.value === 'human') opt.selected = true;
                if (i > 0 && t.value === 'smart') opt.selected = true;
                select.appendChild(opt);
            });

            select.addEventListener('change', () => {
                const snapshotPicker = row.querySelector('.snapshot-picker');
                if (select.value === 'nn') {
                    if (snapshotPicker) snapshotPicker.style.display = 'block';
                } else {
                    if (snapshotPicker) snapshotPicker.style.display = 'none';
                }
            });

            row.appendChild(select);

            // Snapshot picker (hidden by default)
            const snapshotPicker = document.createElement('select');
            snapshotPicker.className = 'snapshot-picker';
            snapshotPicker.dataset.seat = i;
            snapshotPicker.style.display = 'none';
            this._populateSnapshotPicker(snapshotPicker);
            row.appendChild(snapshotPicker);

            seatConfig.appendChild(row);
        }
    }

    async _populateSnapshotPicker(selectEl) {
        try {
            const resp = await fetch('/api/snapshots');
            const snapshots = await resp.json();
            selectEl.innerHTML = '';
            const defaultOpt = document.createElement('option');
            defaultOpt.value = '';
            defaultOpt.textContent = '-- Select Snapshot --';
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

    async _startGame() {
        const numPlayers = parseInt(document.getElementById('num-players').value);
        const devMode = document.getElementById('dev-mode').checked;
        this.animationSpeed = parseInt(document.getElementById('anim-speed').value);

        const seats = [];
        let humanSeat = 0;
        let hasHuman = false;

        for (let i = 0; i < numPlayers; i++) {
            const typeSelect = document.querySelector(`.seat-type[data-seat="${i}"]`);
            const type = typeSelect.value;
            const seat = { type };

            if (type === 'human') {
                humanSeat = i;
                hasHuman = true;
            } else if (type === 'nn') {
                const snapshotSelect = document.querySelector(`.snapshot-picker[data-seat="${i}"]`);
                seat.snapshot_path = snapshotSelect.value;
                if (!seat.snapshot_path) {
                    alert(`Please select a snapshot for Seat ${i}`);
                    return;
                }
            }

            seats.push(seat);
        }

        this.isSpectatorMode = !hasHuman;

        // Create session
        const maxCardsVal = document.getElementById('max-cards').value;
        const sessionBody = {
            mode: 'play',
            num_players: numPlayers,
            seats,
            human_seat: humanSeat,
            dev_mode: devMode,
        };
        if (maxCardsVal) sessionBody.max_cards = parseInt(maxCardsVal);

        try {
            const resp = await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(sessionBody),
            });
            const data = await resp.json();
            if (data.error) {
                alert(`Error: ${data.error}`);
                return;
            }
            this.sessionId = data.session_id;
        } catch (e) {
            alert(`Failed to create session: ${e.message}`);
            return;
        }

        // Switch to game view
        this._startGameView(humanSeat, numPlayers);
    }

    _startGameView(humanSeat, numPlayers) {
        const container = document.getElementById('main-content');
        container.innerHTML = '';
        container.className = 'game-view';

        // Game table container
        const tableContainer = document.createElement('div');
        tableContainer.id = 'game-table';
        container.appendChild(tableContainer);

        // Settings gear
        this._addSettingsGear(container);

        // Spectator controls
        if (this.isSpectatorMode) {
            const controls = document.createElement('div');
            controls.className = 'spectator-controls';

            const playBtn = document.createElement('button');
            playBtn.className = 'btn';
            playBtn.id = 'spectator-play';
            playBtn.textContent = 'Play';
            playBtn.addEventListener('click', () => this._toggleSpectator());
            controls.appendChild(playBtn);

            const stepBtn = document.createElement('button');
            stepBtn.className = 'btn';
            stepBtn.textContent = 'Step';
            stepBtn.addEventListener('click', () => this._spectatorStep());
            controls.appendChild(stepBtn);

            container.appendChild(controls);
        }

        // Back button
        const backBtn = document.createElement('button');
        backBtn.className = 'btn back-btn';
        backBtn.textContent = 'Back to Setup';
        backBtn.addEventListener('click', () => this._exitGame());
        container.appendChild(backBtn);

        // Create renderer
        this.renderer = new GameRenderer(tableContainer);
        this.renderer.setSeatMapping(humanSeat, numPlayers);
        this.renderer.onCardClick = (cardInt) => this._onPlayCard(cardInt);

        // Apply auto-scaling
        this._computeAndApplyScale();
        this._resizeHandler = () => this._computeAndApplyScale();
        window.addEventListener('resize', this._resizeHandler);

        // Connect WebSocket
        const wsUrl = `ws://${window.location.host}/ws/${this.sessionId}`;
        this.ws = new WebSocket(wsUrl);
        this.ws.onmessage = (event) => this._onMessage(JSON.parse(event.data));
        this.ws.onclose = () => console.log('WebSocket closed');
        this.ws.onerror = (e) => console.error('WebSocket error:', e);
    }

    // ── Auto-scaling ──────────────────────────────────────────────────

    _computeAndApplyScale() {
        // Base: reference design at 1200x800
        const wScale = window.innerWidth / 1200;
        const hScale = window.innerHeight / 800;
        const autoScale = Math.min(wScale, hScale, 1.5); // cap at 1.5x
        const finalScale = Math.max(0.5, autoScale * this.baselineScale);
        Card.Settings.scale = finalScale;
        this._applyScale();
        this._rerender();
    }

    _applyScale() {
        const s = Card.Settings.scale;
        document.documentElement.style.setProperty('--card-w', `${Math.round(70 * s)}px`);
        document.documentElement.style.setProperty('--card-h', `${Math.round(100 * s)}px`);
        document.documentElement.style.setProperty('--card-sm-w', `${Math.round(46 * s)}px`);
        document.documentElement.style.setProperty('--card-sm-h', `${Math.round(66 * s)}px`);
        document.documentElement.style.setProperty('--card-font-scale', Card.Settings.fontScale);
    }

    // ── Message handling ──────────────────────────────────────────────

    _onMessage(msg) {
        if (msg.type === 'state_update') {
            const events = msg.events || [];

            if (events.length > 0) {
                this._animateEvents(events, msg.state);
            } else {
                this._renderState(msg.state);
            }
        } else if (msg.type === 'error') {
            console.error('Server error:', msg.message);
        }
    }

    async _animateEvents(events, finalState) {
        if (this.animating) {
            this.eventQueue.push({ events, finalState });
            return;
        }
        this.animating = true;

        for (const event of events) {
            if (event.type === 'bid') {
                // Render intermediate state showing bids recorded so far
                this.state = event.state || finalState;
                this.renderer.render(this.state);
                this._showBidBubble(event.player, event.value);
                await this._delay(this.animationSpeed);
            } else if (event.type === 'play') {
                const ge = event.game_events || {};

                if (ge.trick_complete && event.trick_cards) {
                    // Show the completed trick (all cards visible in center)
                    const trickState = Object.assign({}, event.state, {
                        current_trick: event.trick_cards
                    });
                    this.state = trickState;
                    this.renderer.render(trickState);
                    await this._delay(this.animationSpeed + this.trickDisplayDuration);
                } else {
                    // Normal play — render state showing card in trick area
                    this.state = event.state || finalState;
                    this.renderer.render(this.state);
                    await this._delay(this.animationSpeed);
                }

                if (ge.round_complete) {
                    // Render the post-round state and show score modal
                    this.state = event.state;
                    this.renderer.render(this.state);
                    await this._showRoundScores(ge, event.state);
                    this.animating = false;

                    if (ge.game_over) {
                        this._exitGame();
                        return;
                    }

                    // Process queued events first
                    if (this.eventQueue.length > 0) {
                        const next = this.eventQueue.shift();
                        this._animateEvents(next.events, next.finalState);
                        return;
                    }

                    // Request next round (auto-advances bots)
                    this._sendMessage({ action: 'auto_play' });
                    return;
                }
            }
        }

        // Render final state after all events
        this._renderState(finalState);
        this.animating = false;

        // Process queued events
        if (this.eventQueue.length > 0) {
            const next = this.eventQueue.shift();
            this._animateEvents(next.events, next.finalState);
            return;
        }

        // Show bid picker if it's human's turn to bid
        if (finalState.phase === 0 && finalState.legal_bids) {
            this._showBidPicker(finalState);
        }

        // Check for game over (fallback)
        if (finalState.game_over) {
            this._showGameOver(finalState);
        }
    }

    _renderState(state) {
        this.state = state;
        // Update trump suit for card highlighting
        Card.Settings.trumpSuit = state.trump_suit != null ? state.trump_suit : -1;
        this.renderer.render(state);

        // Show bid picker if needed
        if (state.phase === 0 && state.legal_bids && !this.animating) {
            this._showBidPicker(state);
        }
    }

    _showBidPicker(state) {
        const table = document.getElementById('game-table');
        const existing = table.querySelector('.bid-overlay');
        if (existing) existing.remove();

        const picker = Components.createBidPicker(
            state.hand_size,
            state.legal_bids,
            (bid) => this._onBid(bid)
        );
        table.appendChild(picker);
    }

    _showBidBubble(player, value) {
        const table = document.getElementById('game-table');
        const np = this.state.num_players;
        const displayIdx = this.renderer.displayPos(player);
        const layout = PLAYER_LAYOUTS[np][displayIdx];

        const bubble = document.createElement('div');
        bubble.className = 'bid-bubble';
        bubble.textContent = `Bid ${value}`;
        bubble.style.left = `${layout.lx}%`;
        bubble.style.top = `${layout.ly - 5}%`;

        table.appendChild(bubble);
        setTimeout(() => bubble.remove(), this.animationSpeed * 2);
    }

    // ── Settings gear ─────────────────────────────────────────────────

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
                this._rerender();
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
                this._rerender();
            });
            fontRow.appendChild(fontLabel);
            fontRow.appendChild(fontSlider);
            fontRow.appendChild(fontVal);
            panel.appendChild(fontRow);

            // Baseline scale
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
            });
            scaleRow.appendChild(scaleLabel);
            scaleRow.appendChild(scaleSlider);
            scaleRow.appendChild(scaleVal);
            panel.appendChild(scaleRow);

            // Trick display duration
            const trickRow = document.createElement('div');
            trickRow.className = 'settings-row';
            const trickLabel = document.createElement('label');
            trickLabel.textContent = 'Trick Pause';
            const trickSlider = document.createElement('input');
            trickSlider.type = 'range';
            trickSlider.min = '200';
            trickSlider.max = '5000';
            trickSlider.step = '100';
            trickSlider.value = String(this.trickDisplayDuration);
            const trickVal = document.createElement('span');
            trickVal.className = 'settings-value';
            trickVal.textContent = `${(this.trickDisplayDuration / 1000).toFixed(1)}s`;
            trickSlider.addEventListener('input', () => {
                this.trickDisplayDuration = parseInt(trickSlider.value);
                trickVal.textContent = `${(this.trickDisplayDuration / 1000).toFixed(1)}s`;
            });
            trickRow.appendChild(trickLabel);
            trickRow.appendChild(trickSlider);
            trickRow.appendChild(trickVal);
            panel.appendChild(trickRow);

            // Animation speed (live adjustment)
            const animRow = document.createElement('div');
            animRow.className = 'settings-row';
            const animLabel = document.createElement('label');
            animLabel.textContent = 'Bot Speed';
            const animSlider = document.createElement('input');
            animSlider.type = 'range';
            animSlider.min = '0';
            animSlider.max = '2000';
            animSlider.step = '100';
            animSlider.value = String(this.animationSpeed);
            const animVal = document.createElement('span');
            animVal.className = 'settings-value';
            animVal.textContent = `${this.animationSpeed}ms`;
            animSlider.addEventListener('input', () => {
                this.animationSpeed = parseInt(animSlider.value);
                animVal.textContent = `${this.animationSpeed}ms`;
            });
            animRow.appendChild(animLabel);
            animRow.appendChild(animSlider);
            animRow.appendChild(animVal);
            panel.appendChild(animRow);

            container.appendChild(panel);
        });

        container.appendChild(gear);
    }

    _rerender() {
        if (this.state) {
            this.renderer.render(this.state);
            if (this.state.phase === 0 && this.state.legal_bids) {
                this._showBidPicker(this.state);
            }
        }
    }

    async _showRoundScores(gameEvents, state) {
        return new Promise(resolve => {
            const table = document.getElementById('game-table');
            const scoreData = {
                roundIndex: state.round_index - 1,
                handSize: gameEvents.round_hand_size,
                numPlayers: state.num_players,
                seatLabels: state.seat_labels,
                bids: gameEvents.round_bids,
                tricksWon: gameEvents.round_tricks_won,
                roundScores: gameEvents.round_scores,
                scores: state.scores,
                gameOver: gameEvents.game_over,
            };

            const modal = Components.createScoreTable(scoreData, resolve);
            table.appendChild(modal);
        });
    }

    _showGameOver(state) {
        const table = document.getElementById('game-table');
        const scoreData = {
            roundIndex: state.round_index,
            handSize: 0,
            numPlayers: state.num_players,
            seatLabels: state.seat_labels,
            bids: [],
            tricksWon: [],
            roundScores: null,
            scores: state.scores,
            gameOver: true,
        };

        const modal = Components.createScoreTable(scoreData, () => {
            this._exitGame();
        });
        table.appendChild(modal);
    }

    _onBid(value) {
        this._sendMessage({ action: 'bid', value });
    }

    _onPlayCard(cardInt) {
        this._sendMessage({ action: 'play_card', card: cardInt });
    }

    _sendMessage(msg) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(msg));
        }
    }

    _toggleSpectator() {
        const btn = document.getElementById('spectator-play');
        if (this.spectatorPaused || !this.spectatorInterval) {
            this.spectatorPaused = false;
            btn.textContent = 'Pause';
            this.spectatorInterval = setInterval(() => {
                if (!this.animating) {
                    this._sendMessage({ action: 'advance' });
                }
            }, this.animationSpeed + 200);
        } else {
            this.spectatorPaused = true;
            btn.textContent = 'Play';
            clearInterval(this.spectatorInterval);
            this.spectatorInterval = null;
        }
    }

    _spectatorStep() {
        if (!this.animating) {
            this._sendMessage({ action: 'advance' });
        }
    }

    _exitGame() {
        // Clean up resize listener
        if (this._resizeHandler) {
            window.removeEventListener('resize', this._resizeHandler);
            this._resizeHandler = null;
        }
        if (this.spectatorInterval) {
            clearInterval(this.spectatorInterval);
            this.spectatorInterval = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        if (this.sessionId) {
            fetch(`/api/sessions/${this.sessionId}`, { method: 'DELETE' });
            this.sessionId = null;
        }
        this.app.showSetup();
    }

    _delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

window.PlayMode = PlayMode;
