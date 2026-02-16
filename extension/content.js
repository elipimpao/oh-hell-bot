/**
 * content.js — Chrome Extension content script for Trickster Cards.
 *
 * Responsibilities:
 *  1. Listen for OHB_EVENT messages from inject.js (page world)
 *  2. Manage game state machine
 *  3. Communicate with the oh-hell-bot backend via background service worker
 *  4. Render a recommendation overlay on the game page
 */
(function () {
  'use strict';

  const LOG = (...args) => console.log('[OHB]', ...args);
  LOG('content.js loaded');

  // Detect invalidated extension context (happens when extension is reloaded
  // but the game page is not refreshed)
  if (typeof chrome === 'undefined' || !chrome.runtime || !chrome.runtime.id) {
    console.warn('[OHB] Extension context invalidated! Please REFRESH this page (F5) after reloading the extension.');
    // Try to show a visible warning
    const warnEl = document.createElement('div');
    warnEl.style.cssText = 'position:fixed;top:10px;right:10px;z-index:999999;background:#d32f2f;color:#fff;padding:12px 16px;border-radius:8px;font:14px sans-serif;box-shadow:0 4px 12px rgba(0,0,0,0.4);cursor:pointer;';
    warnEl.textContent = 'Oh Hell Advisor: Extension reloaded — please refresh this page (F5)';
    warnEl.onclick = () => { warnEl.remove(); location.reload(); };
    if (document.body) document.body.appendChild(warnEl);
    else document.addEventListener('DOMContentLoaded', () => document.body.appendChild(warnEl));
    return; // Stop — nothing will work without chrome.runtime
  }

  // ─── Settings & defaults ───────────────────────────────────────────
  const DEFAULT_BACKEND = 'http://localhost:8000';
  let backendUrl = DEFAULT_BACKEND;
  let snapshotPath = '';
  let autoplayEnabled = false;
  let autoplayDelay = 750;   // ms delay before autoplay acts (50-2000)
  let autoplayTimer = null;  // pending autoplay timeout

  // ─── Card display helpers ──────────────────────────────────────────
  const SUIT_SYMBOLS = ['\u2663', '\u2666', '\u2665', '\u2660'];
  const RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
  const SUIT_COLORS = ['#90caf9', '#ff6b6b', '#ff6b6b', '#90caf9']; // light blue for ♣♠, red for ♦♥
  const SUIT_WORD_MAP = { 'Clubs': 0, 'Diamonds': 1, 'Hearts': 2, 'Spades': 3 };
  const RANK_WORD_MAP = {
    'Two': 0, 'Three': 1, 'Four': 2, 'Five': 3, 'Six': 4, 'Seven': 5,
    'Eight': 6, 'Nine': 7, 'Ten': 8, 'Jack': 9, 'Queen': 10, 'King': 11, 'Ace': 12
  };

  function cardName(c) {
    return RANK_NAMES[c % 13] + SUIT_SYMBOLS[Math.floor(c / 13)];
  }

  function cardColor(c) {
    return SUIT_COLORS[Math.floor(c / 13)];
  }

  // ─── Game state ────────────────────────────────────────────────────
  let localPlayerName = null;     // The logged-in user's display name
  let players = {};               // name -> trickster seat index
  let numPlayers = 0;
  let seatOrder = [];             // advisor seat 0 = user, then clockwise
  let tricksterToAdvisor = {};    // trickster seat -> advisor seat

  let bgPort = null;              // Port to background service worker
  let sessionId = null;
  let sessionCreating = false;    // true while waiting for session creation response
  let wsReady = false;            // true when backend WebSocket is open
  let phase = 'idle';             // idle | setup | bidding | playing
  let roundNumber = 0;
  let bidOrderNames = [];         // names in bid order for current round
  let firstBidderName = null;     // first player to bid (= dealer+1)
  let dealerDetected = false;
  let bidsRecorded = 0;
  let playsInTrick = 0;
  let localTurnSignaled = false;  // true after player_turn/lead_prompt fires for local player
  let lastDealtCards = [];         // saved for session recreation if player count changes
  let inferredBidOrder = [];       // names in bid order (for rejoin player inference)
  let inferredBids = {};           // name → bidValue (for rejoin state replay)
  let lastTrumpCard = -1;          // saved from DOM detection for rejoin replay
  let pendingRecommendation = null; // 'bid' or 'play' if awaiting retry (e.g. trump not set yet)

  // ─── Seat mapping helpers ──────────────────────────────────────────

  /** Map a Trickster player name to the advisor seat index (user = 0, clockwise). */
  function advisorSeat(name) {
    if (!seatOrder.length) return -1;
    const idx = seatOrder.indexOf(name);
    if (idx >= 0) return idx;
    LOG('advisorSeat: unknown player', name, '- known:', JSON.stringify(seatOrder));
    return -1;
  }

  /**
   * Build seatOrder from the players map.
   * Trickster seats may be non-contiguous (e.g. [0, 2, 3, 4] for 4 players).
   * We sort actual seat numbers, then rotate so the local player is advisor seat 0.
   */
  function buildSeatOrder() {
    if (!localPlayerName || !Object.keys(players).length) return;
    const localSeat = players[localPlayerName];
    if (localSeat === undefined) {
      LOG('buildSeatOrder: localPlayerName not found in players:', localPlayerName, players);
      return;
    }

    // Sort actual Trickster seats — player count = number of known players
    const allSeats = Object.values(players).sort((a, b) => a - b);
    numPlayers = allSeats.length;

    // Map: local player's Trickster seat -> advisor 0, then clockwise through sorted seats
    const localIdx = allSeats.indexOf(localSeat);
    tricksterToAdvisor = {};
    for (let i = 0; i < numPlayers; i++) {
      const tricksterSeat = allSeats[(localIdx + i) % numPlayers];
      tricksterToAdvisor[tricksterSeat] = i;
    }

    // Build seatOrder (names in advisor seat order)
    seatOrder = new Array(numPlayers).fill(null);
    for (const [name, seat] of Object.entries(players)) {
      const advSeat = tricksterToAdvisor[seat];
      if (advSeat !== undefined) {
        seatOrder[advSeat] = name;
      }
    }

    LOG('buildSeatOrder:', seatOrder, '(numPlayers:', numPlayers, ') tricksterToAdvisor:', JSON.stringify(tricksterToAdvisor));
  }

  // ─── Trump detection from DOM ──────────────────────────────────────

  function readTrumpFromDOM() {
    // Try aria-label first: "King of Hearts (Trump)"
    const trumpEl = document.querySelector('.card.trump.face-up');
    LOG('readTrumpFromDOM: trumpEl =', trumpEl ? 'found' : 'NOT found');
    if (trumpEl) {
      const label = trumpEl.getAttribute('aria-label') || '';
      LOG('readTrumpFromDOM: aria-label =', JSON.stringify(label));
      const match = label.match(/(\w+)\s+of\s+(\w+)/);
      if (match) {
        const rank = RANK_WORD_MAP[match[1]];
        const suit = SUIT_WORD_MAP[match[2]];
        LOG('readTrumpFromDOM: parsed rank =', match[1], rank, 'suit =', match[2], suit);
        if (rank !== undefined && suit !== undefined) {
          const card = suit * 13 + rank;
          LOG('readTrumpFromDOM: trump card =', card, cardName(card));
          return card;
        }
      }
      // Fallback: parse CSS classes like "face King Hearts"
      const faceEl = trumpEl.querySelector('.face');
      if (faceEl) {
        LOG('readTrumpFromDOM: fallback classes =', faceEl.className);
        const classes = faceEl.className.split(/\s+/);
        let rank = -1, suit = -1;
        for (const cls of classes) {
          if (RANK_WORD_MAP[cls] !== undefined) rank = RANK_WORD_MAP[cls];
          if (SUIT_WORD_MAP[cls] !== undefined) suit = SUIT_WORD_MAP[cls];
        }
        if (rank >= 0 && suit >= 0) return suit * 13 + rank;
      }
    }
    LOG('readTrumpFromDOM: returning -1 (not found)');
    return -1;
  }

  // ─── Backend communication (via background service worker) ─────────

  let sendQueue = [];  // messages queued before WebSocket is ready

  function connectBackground() {
    if (bgPort) {
      LOG('connectBackground: already connected, skipping');
      return;
    }
    LOG('connectBackground: connecting to background service worker...');
    try {
      bgPort = chrome.runtime.connect({ name: 'ohb' });
    } catch (e) {
      LOG('connectBackground: chrome.runtime.connect() FAILED:', e.message || e);
      bgPort = null;
      return;
    }
    bgPort.onMessage.addListener((msg) => {
      LOG('From background:', msg.type, msg.type === 'ws_message' ? JSON.stringify(msg.data).substring(0, 200) : (msg.message || msg.sessionId || ''));
      switch (msg.type) {
        case 'session_created':
          sessionCreating = false;
          sessionId = msg.sessionId;
          LOG('Session created, id:', sessionId);
          setStatus('Session created, opening WebSocket...');
          break;
        case 'session_error':
          sessionCreating = false;
          LOG('Session error:', msg.message);
          setStatus('Error: ' + msg.message, true);
          break;
        case 'ws_open':
          if (wsReady) {
            LOG('ws_open: already ready, ignoring duplicate');
            break;
          }
          wsReady = true;
          LOG('WebSocket ready! Queue has', sendQueue.length, 'messages');
          setStatus('Connected');
          flushQueue();
          break;
        case 'ws_close':
          wsReady = false;
          sessionId = null;  // Allow new session creation (e.g. backend restarted)
          setStatus('Disconnected', true);
          break;
        case 'ws_error':
          setStatus('WebSocket error', true);
          break;
        case 'ws_message':
          handleBackendMessage(msg.data);
          break;
        case 'error':
          setStatus('Background: ' + msg.message, true);
          break;
      }
    });
    bgPort.onDisconnect.addListener(() => {
      LOG('bgPort disconnected');
      bgPort = null;
      wsReady = false;
      sessionId = null;  // Allow new session creation
    });
  }

  /** Try to connect + create session. Retries once after 1s if connection fails. */
  function ensureConnected(retryCount) {
    retryCount = retryCount || 0;
    if (sessionId || sessionCreating) return;
    if (numPlayers < 2) return;
    connectBackground();
    if (!bgPort && retryCount < 2) {
      LOG('ensureConnected: bgPort is null, will retry in 1s (attempt', retryCount + 1, ')');
      setTimeout(() => ensureConnected(retryCount + 1), 1000);
      return;
    }
    createSession();
  }

  function createSession() {
    LOG('createSession: numPlayers =', numPlayers, 'bgPort =', !!bgPort, 'backendUrl =', backendUrl, 'snapshotPath =', snapshotPath, 'sessionId =', sessionId, 'sessionCreating =', sessionCreating);
    if (sessionId || sessionCreating) {
      LOG('createSession: session already exists or being created, skipping');
      return;
    }
    if (numPlayers < 2 || !bgPort) {
      LOG('createSession: ABORTED (numPlayers < 2 or no bgPort)');
      return;
    }
    sessionCreating = true;
    const body = {
      mode: 'advisor',
      num_players: numPlayers,
      advisor_snapshot: snapshotPath || null
    };
    LOG('createSession: sending create_session with body:', JSON.stringify(body));
    bgPort.postMessage({
      type: 'create_session',
      backendUrl,
      body
    });
  }

  /** Send a message to the backend. Queues if WebSocket isn't ready yet. */
  function send(msg) {
    if (bgPort && wsReady) {
      LOG('send:', msg.action, JSON.stringify(msg).substring(0, 150));
      bgPort.postMessage({ type: 'ws_send', data: msg });
    } else {
      LOG('send QUEUED (wsReady=' + wsReady + ' bgPort=' + !!bgPort + '):', msg.action);
      sendQueue.push(msg);
      // Try to reconnect if session is lost (backend restart, extension reload, etc.)
      if (!sessionId && !sessionCreating && numPlayers >= 2) {
        ensureConnected();
      }
    }
  }

  /** Flush queued messages once the WebSocket connects. */
  function flushQueue() {
    LOG('flushQueue: flushing', sendQueue.length, 'queued messages, phase =', phase);
    while (sendQueue.length > 0) {
      const msg = sendQueue.shift();
      LOG('flushQueue send:', msg.action, JSON.stringify(msg).substring(0, 150));
      if (bgPort && wsReady) {
        bgPort.postMessage({ type: 'ws_send', data: msg });
      }
    }
    // Don't auto-request recommendations here — wait for explicit turn signals
    // (bid_prompt, player_turn, lead_prompt) to ensure dealer/bids are set first
  }

  function handleBackendMessage(msg) {
    LOG('handleBackendMessage:', msg.type, msg.type === 'error' ? msg.message : '');
    if (msg.type === 'bid_recommendation') {
      pendingRecommendation = null;
      setStatus('Connected');
      LOG('Bid recommendation received:', msg.recommendations?.length, 'options, value:', msg.value);
      showBidRecommendation(msg.recommendations, msg.value);
    } else if (msg.type === 'play_recommendation') {
      pendingRecommendation = null;
      setStatus('Connected');
      LOG('Play recommendation received:', msg.recommendations?.length, 'options, value:', msg.value);
      showPlayRecommendation(msg.recommendations, msg.value);
    } else if (msg.type === 'advisor_state') {
      // State acknowledgment from backend — useful for debugging
      LOG('Advisor state:', JSON.stringify(msg.state).substring(0, 300));
    } else if (msg.type === 'error') {
      LOG('Backend error:', msg.message);
      if (msg.message) setStatus('Backend: ' + msg.message, true);
    }
  }

  // ─── Event handlers from inject.js ─────────────────────────────────

  function onEvent(event, data) {
    LOG('EVENT:', event, JSON.stringify(data).substring(0, 200));
    switch (event) {
      case 'create_game':
      case 'rejoin_game':
        resetAll();
        phase = 'setup';
        setStatus('Game detected, waiting for players...');
        break;

      case 'player_joined': {
        localPlayerName = data.localName;
        players = data.players;
        const prevNumPlayers = numPlayers;
        LOG('player_joined: localPlayer =', localPlayerName, 'players =', JSON.stringify(players));
        buildSeatOrder();
        if (numPlayers >= 2) {
          setStatus(`${numPlayers} players detected`);
        }
        // If session was already created with wrong player count, recreate it
        if ((sessionId || sessionCreating) && numPlayers !== prevNumPlayers && numPlayers >= 2) {
          LOG('player_joined: numPlayers changed from', prevNumPlayers, 'to', numPlayers, '— recreating session');
          if (bgPort) bgPort.postMessage({ type: 'close_session' });
          sessionId = null;
          sessionCreating = false;
          wsReady = false;
          // Re-queue essential state for the new session
          sendQueue = [];
          if (lastDealtCards.length) {
            if (roundNumber > 0) sendQueue.push({ action: 'new_round' });
            sendQueue.push({ action: 'set_hand', cards: lastDealtCards });
          }
          createSession();
          // Re-trigger trump detection (DOM should still have the trump card)
          if (lastDealtCards.length) {
            attemptTrumpDetection(lastDealtCards.length, 0);
          }
        }
        break;
      }

      case 'cards_dealt':
        LOG('cards_dealt:', data.cards.length, 'cards:', data.cards, 'localName:', data.localName);
        // Set local player name from inject.js if not already known (rejoin case)
        if (data.localName && !localPlayerName) {
          localPlayerName = data.localName;
          LOG('localPlayerName set from cards_dealt:', localPlayerName);
        }
        // Always track game state first
        onCardsDealt(data.cards);
        // Small delay before creating session to ensure all player_joined events
        // have been processed (they may arrive slightly after cards_dealt)
        setTimeout(() => ensureConnected(), 100);
        break;

      case 'begin_bid':
        LOG('begin_bid:', data.name, '-> advisor seat', advisorSeat(data.name));
        onBeginBid(data.name);
        break;

      case 'player_bid':
        LOG('player_bid:', data.name, 'bid', data.bidValue, '-> advisor seat', advisorSeat(data.name));
        onPlayerBid(data.name, data.bidValue);
        break;

      case 'bid_prompt':
        // Identify local player from preceding begin_bid (rejoin fallback)
        if (!localPlayerName && bidOrderNames.length > 0) {
          localPlayerName = bidOrderNames[bidOrderNames.length - 1];
          LOG('localPlayerName inferred from bid_prompt:', localPlayerName);
        }
        LOG('bid_prompt: requesting bid recommendation, dealerDetected =', dealerDetected, 'bidsRecorded =', bidsRecorded, 'numPlayers =', numPlayers);
        if (!dealerDetected && numPlayers > 0) {
          // User (seat 0) is the (bidsRecorded)-th bidder (0-indexed).
          // First bidder = (dealer + 1) % N, so:
          // dealer = (N - 1 - bidsRecorded) % N
          const inferredDealer = (numPlayers - 1 - bidsRecorded) % numPlayers;
          LOG('Inferring dealer from bid_prompt: seat', inferredDealer);
          send({ action: 'set_dealer', player: inferredDealer });
          dealerDetected = true;
        }
        // Only request recommendation if we have a session (skip on rejoin first round)
        if (seatOrder.length) {
          if (lastTrumpCard >= 0) {
            requestRecommendation('bid');
          } else {
            // Trump not detected yet — defer until it is
            pendingRecommendation = 'bid';
            LOG('bid_prompt: deferring recommendation until trump is detected');
          }
        }
        break;

      case 'hand_started':
        phase = 'playing';
        playsInTrick = 0;
        localTurnSignaled = false;
        updateOverlayPhase();
        // Rejoin recovery: if no player_joined events, build player map from bid order
        if (!seatOrder.length && localPlayerName && inferredBidOrder.length >= 2) {
          LOG('hand_started: rejoin recovery — building player map from bid order');
          buildPlayersFromBidOrder();
          // Clear stale queued messages from unknown-player phase
          sendQueue = [];
          ensureConnected();
          replayRoundState();
        }
        break;

      case 'card_played':
        LOG('card_played:', data.name, 'played card', data.card, '-> advisor seat', advisorSeat(data.name),
            'localTurnSignaled =', localTurnSignaled);
        // Skip replayed/auto plays for the local player that arrive before player_turn
        if (data.name === localPlayerName && !localTurnSignaled) {
          LOG('card_played: SKIPPING local player replay (no player_turn yet)');
          break;
        }
        onCardPlayed(data.card, data.name);
        break;

      case 'player_turn':
        LOG('player_turn:', data.name, '(localPlayer =', localPlayerName, ')');
        if (data.name === localPlayerName) {
          localTurnSignaled = true;
          if (lastTrumpCard >= 0) {
            requestRecommendation('play');
          } else {
            pendingRecommendation = 'play';
            LOG('player_turn: deferring recommendation until trump is detected');
          }
        }
        break;

      case 'lead_prompt':
        LOG('lead_prompt: requesting play recommendation');
        localTurnSignaled = true;
        if (lastTrumpCard >= 0) {
          requestRecommendation('play');
        } else {
          pendingRecommendation = 'play';
          LOG('lead_prompt: deferring recommendation until trump is detected');
        }
        break;

      case 'trick_won':
        playsInTrick = 0;
        localTurnSignaled = false;
        clearRecommendation();
        break;

      case 'hand_ended':
        phase = 'idle';
        roundNumber++;
        clearRecommendation();
        updateOverlayPhase();
        break;
    }
  }

  function onCardsDealt(cards) {
    phase = 'bidding';
    bidsRecorded = 0;
    bidOrderNames = [];
    firstBidderName = null;
    dealerDetected = false;
    playsInTrick = 0;
    localTurnSignaled = false;
    lastDealtCards = [...cards];
    inferredBidOrder = [];
    inferredBids = {};
    lastTrumpCard = -1;
    pendingRecommendation = null;

    // New round
    if (roundNumber > 0) {
      send({ action: 'new_round' });
    }

    // Send hand immediately (don't wait for trump)
    send({ action: 'set_hand', cards });
    updateOverlayPhase();

    // Detect trump from DOM with retries (DOM may not be rendered yet)
    attemptTrumpDetection(cards.length, 0);
  }

  function attemptTrumpDetection(handSize, attempt) {
    // Dealing animation takes ~100ms per card, so large hands (12+ cards) need 2-4+ seconds.
    // Use escalating delays with enough headroom for the longest rounds.
    const delays = [100, 400, 1000, 2000, 3500, 5000];
    if (attempt >= delays.length) {
      LOG('Trump detection FAILED after', delays.length, 'attempts. Exploring DOM...');
      // Log DOM info so user can relay it for debugging
      const allTrumps = document.querySelectorAll('[class*="trump" i]');
      LOG('Elements with "trump" in class:', allTrumps.length);
      allTrumps.forEach((el, i) => {
        LOG('  trump el', i, ':', el.tagName, el.className, el.getAttribute('aria-label') || '(no aria-label)');
      });
      // Also check for any card elements with aria-label containing "Trump"
      const ariaLabels = document.querySelectorAll('[aria-label*="Trump"]');
      LOG('Elements with "Trump" in aria-label:', ariaLabels.length);
      ariaLabels.forEach((el, i) => {
        LOG('  aria el', i, ':', el.tagName, el.className, el.getAttribute('aria-label'));
      });
      return;
    }
    setTimeout(() => {
      const trumpCard = readTrumpFromDOM();
      if (trumpCard >= 0) {
        lastTrumpCard = trumpCard;
        send({ action: 'set_trump', card: trumpCard });
        updateOverlayInfo(handSize, trumpCard);
        // Retry any pending recommendation that failed due to missing trump
        if (pendingRecommendation) {
          LOG('Trump detected — retrying pending', pendingRecommendation, 'recommendation');
          requestRecommendation(pendingRecommendation);
        }
      } else {
        LOG('Trump attempt', attempt + 1, '/', delays.length, 'failed, retrying in', delays[attempt + 1] || 'N/A', 'ms');
        attemptTrumpDetection(handSize, attempt + 1);
      }
    }, delays[attempt]);
  }

  function onBeginBid(name) {
    // Just track the bid order; dealer is inferred from bid_prompt
    bidOrderNames.push(name);
  }

  /**
   * Build the player map from bid order when onPlayerJoined never fired (rejoin case).
   * Bid order IS the clockwise order (starting from dealer+1), so assigning
   * sequential Trickster seats in bid order produces correct advisor seat mapping.
   */
  function buildPlayersFromBidOrder() {
    // Ensure local player is in the list
    if (inferredBidOrder.indexOf(localPlayerName) === -1) {
      inferredBidOrder.push(localPlayerName);
    }
    // Assign sequential Trickster seats in bid order
    players = {};
    for (let i = 0; i < inferredBidOrder.length; i++) {
      players[inferredBidOrder[i]] = i;
    }
    buildSeatOrder();
    setStatus(`${numPlayers} players detected (rejoin)`);
    LOG('buildPlayersFromBidOrder:', inferredBidOrder, '→ seatOrder:', seatOrder, 'numPlayers:', numPlayers);
  }

  /** Re-send essential game state after session creation (rejoin recovery). */
  function replayRoundState() {
    // Hand
    if (lastDealtCards.length) {
      send({ action: 'set_hand', cards: lastDealtCards });
    }
    // Trump
    if (lastTrumpCard >= 0) {
      send({ action: 'set_trump', card: lastTrumpCard });
    }
    // Dealer = last bidder in the round
    if (inferredBidOrder.length > 0) {
      const dealerName = inferredBidOrder[inferredBidOrder.length - 1];
      const dealerSeat = advisorSeat(dealerName);
      if (dealerSeat >= 0) {
        send({ action: 'set_dealer', player: dealerSeat });
        dealerDetected = true;
      }
    }
    // Replay all bids
    for (const name of inferredBidOrder) {
      const seat = advisorSeat(name);
      const bid = inferredBids[name];
      if (seat >= 0 && bid !== undefined) {
        send({ action: 'record_bid', player: seat, value: bid });
      }
    }
    bidsRecorded = inferredBidOrder.length;
    LOG('replayRoundState: sent hand, trump, dealer, and', bidsRecorded, 'bids');
  }

  function onPlayerBid(name, bidValue) {
    // Always track for rejoin inference (even if seatOrder not built yet)
    if (inferredBidOrder.indexOf(name) === -1) {
      inferredBidOrder.push(name);
    }
    inferredBids[name] = bidValue;

    const seat = advisorSeat(name);
    if (seat < 0) return;

    // Infer dealer from first bid: first bidder = (dealer + 1) % N
    // So dealer = (firstBidder - 1 + N) % N
    if (!dealerDetected && bidsRecorded === 0) {
      const dealer = (seat - 1 + numPlayers) % numPlayers;
      LOG('Inferring dealer from first bid:', name, 'seat', seat, '-> dealer seat', dealer);
      send({ action: 'set_dealer', player: dealer });
      dealerDetected = true;
    }

    send({ action: 'record_bid', player: seat, value: bidValue });
    bidsRecorded++;
  }

  function onCardPlayed(card, name) {
    const seat = advisorSeat(name);
    if (seat < 0) return;
    send({ action: 'record_play', player: seat, card });
    playsInTrick++;
  }

  function requestRecommendation(recPhase) {
    pendingRecommendation = recPhase;
    send({ action: 'get_recommendation', phase: recPhase });
  }

  // ─── Full reset ────────────────────────────────────────────────────

  function resetAll() {
    if (bgPort) {
      bgPort.postMessage({ type: 'close_session' });
      bgPort.disconnect();
      bgPort = null;
    }
    wsReady = false;
    sendQueue = [];
    sessionId = null;
    sessionCreating = false;
    localPlayerName = null;
    players = {};
    numPlayers = 0;
    seatOrder = [];
    tricksterToAdvisor = {};
    phase = 'idle';
    roundNumber = 0;
    bidOrderNames = [];
    firstBidderName = null;
    dealerDetected = false;
    bidsRecorded = 0;
    playsInTrick = 0;
    localTurnSignaled = false;
    lastDealtCards = [];
    inferredBidOrder = [];
    inferredBids = {};
    lastTrumpCard = -1;
    pendingRecommendation = null;
    clearRecommendation();
  }

  // ─── Overlay UI ────────────────────────────────────────────────────

  let overlay = null;
  let statusEl = null;
  let infoEl = null;
  let phaseEl = null;
  let recEl = null;
  let minimized = false;

  function createOverlay() {
    overlay = document.createElement('div');
    overlay.id = 'ohb-overlay';
    overlay.innerHTML = `
      <div id="ohb-header">
        <span id="ohb-title">Oh Hell Advisor</span>
        <span id="ohb-auto-badge" style="display:none">AUTO</span>
        <span id="ohb-controls">
          <button id="ohb-settings-btn" title="Settings">\u2699</button>
          <button id="ohb-minimize" title="Minimize">\u2212</button>
        </span>
      </div>
      <div id="ohb-settings" style="display:none">
        <div class="ohb-setting-row">
          <span class="ohb-setting-label">Autoplay</span>
          <label class="ohb-toggle">
            <input type="checkbox" id="ohb-autoplay-toggle">
            <span class="ohb-toggle-slider"></span>
          </label>
        </div>
        <div class="ohb-setting-row ohb-delay-row">
          <span class="ohb-setting-label">Delay</span>
          <input type="range" id="ohb-delay-slider" min="50" max="2000" step="50" value="750">
          <span id="ohb-delay-value" class="ohb-delay-value">750ms</span>
        </div>
      </div>
      <div id="ohb-body">
        <div id="ohb-status">Waiting for game...</div>
        <div id="ohb-info"></div>
        <div id="ohb-phase"></div>
        <div id="ohb-recommendations"></div>
      </div>
    `;
    document.documentElement.appendChild(overlay);

    statusEl = document.getElementById('ohb-status');
    infoEl = document.getElementById('ohb-info');
    phaseEl = document.getElementById('ohb-phase');
    recEl = document.getElementById('ohb-recommendations');

    // Settings toggle
    document.getElementById('ohb-settings-btn').addEventListener('click', () => {
      const panel = document.getElementById('ohb-settings');
      panel.style.display = panel.style.display === 'none' ? '' : 'none';
    });

    // Autoplay toggle
    const autoToggle = document.getElementById('ohb-autoplay-toggle');
    autoToggle.checked = autoplayEnabled;
    autoToggle.addEventListener('change', () => {
      autoplayEnabled = autoToggle.checked;
      LOG('Autoplay toggled:', autoplayEnabled);
      updateAutoplayBadge();
      if (chrome.storage) {
        chrome.storage.local.set({ autoplayEnabled });
      }
    });

    // Delay slider
    const delaySlider = document.getElementById('ohb-delay-slider');
    const delayLabel = document.getElementById('ohb-delay-value');
    delaySlider.value = autoplayDelay;
    delayLabel.textContent = autoplayDelay + 'ms';
    delaySlider.addEventListener('input', () => {
      autoplayDelay = parseInt(delaySlider.value);
      delayLabel.textContent = autoplayDelay + 'ms';
    });
    delaySlider.addEventListener('change', () => {
      autoplayDelay = parseInt(delaySlider.value);
      delayLabel.textContent = autoplayDelay + 'ms';
      LOG('Autoplay delay set to:', autoplayDelay);
      if (chrome.storage) {
        chrome.storage.local.set({ autoplayDelay });
      }
    });

    // Minimize toggle
    document.getElementById('ohb-minimize').addEventListener('click', () => {
      minimized = !minimized;
      document.getElementById('ohb-body').style.display = minimized ? 'none' : '';
      document.getElementById('ohb-settings').style.display = 'none';
      document.getElementById('ohb-minimize').textContent = minimized ? '+' : '\u2212';
    });

    // Make draggable
    makeDraggable(overlay, document.getElementById('ohb-header'));
    updateAutoplayBadge();
  }

  function updateAutoplayBadge() {
    const badge = document.getElementById('ohb-auto-badge');
    if (badge) badge.style.display = autoplayEnabled ? '' : 'none';
  }

  function makeDraggable(el, handle) {
    let offsetX = 0, offsetY = 0, dragging = false;
    handle.style.cursor = 'grab';
    handle.addEventListener('mousedown', (e) => {
      if (e.target.tagName === 'BUTTON') return;
      dragging = true;
      offsetX = e.clientX - el.getBoundingClientRect().left;
      offsetY = e.clientY - el.getBoundingClientRect().top;
      handle.style.cursor = 'grabbing';
      e.preventDefault();
    });
    document.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      el.style.left = (e.clientX - offsetX) + 'px';
      el.style.top = (e.clientY - offsetY) + 'px';
      el.style.right = 'auto';
      el.style.bottom = 'auto';
    });
    document.addEventListener('mouseup', () => {
      dragging = false;
      handle.style.cursor = 'grab';
    });
  }

  function setStatus(text, isError) {
    if (statusEl) {
      statusEl.textContent = text;
      statusEl.className = isError ? 'ohb-error' : '';
    }
  }

  function updateOverlayInfo(handSize, trumpCard) {
    if (!infoEl) return;
    const trumpSuit = Math.floor(trumpCard / 13);
    const trumpRank = RANK_NAMES[trumpCard % 13];
    const suitSym = SUIT_SYMBOLS[trumpSuit];
    const color = SUIT_COLORS[trumpSuit];
    infoEl.innerHTML = `Round ${roundNumber + 1} &middot; ${handSize} cards &middot; Trump <span style="color:${color}">${trumpRank}${suitSym}</span>`;
  }

  function updateOverlayPhase() {
    if (!phaseEl) return;
    if (phase === 'bidding') {
      phaseEl.textContent = 'Phase: Bidding';
    } else if (phase === 'playing') {
      phaseEl.textContent = 'Phase: Playing';
    } else {
      phaseEl.textContent = '';
    }
  }

  function clearRecommendation() {
    if (recEl) recEl.innerHTML = '';
    pendingRecommendation = null;
    if (autoplayTimer) {
      clearTimeout(autoplayTimer);
      autoplayTimer = null;
    }
  }

  /** Schedule an autoplay action after a configurable delay so the user can see the recommendation. */
  function scheduleAutoplay(action, value) {
    if (autoplayTimer) clearTimeout(autoplayTimer);
    const label = action === 'bid' ? `Bidding ${value}` : `Playing ${cardName(value)}`;
    setStatus('Autoplay: ' + label + '...');
    LOG('Autoplay scheduled:', action, value, '(' + autoplayDelay + 'ms delay)');
    autoplayTimer = setTimeout(() => {
      autoplayTimer = null;
      LOG('Autoplay executing:', action, value);
      window.postMessage({ type: 'OHB_AUTOPLAY', action, value }, '*');
    }, autoplayDelay);
  }

  function showBidRecommendation(recs, value) {
    if (!recEl) return;
    let html = '<div class="ohb-rec-title">BID RECOMMENDATION</div>';
    for (let i = 0; i < recs.length && i < 5; i++) {
      const r = recs[i];
      const pct = (r.prob * 100).toFixed(1);
      const barWidth = Math.max(2, r.prob * 100);
      const isTop = i === 0;
      html += `
        <div class="ohb-rec-row ${isTop ? 'ohb-rec-top' : ''}">
          <span class="ohb-rec-label">${isTop ? '\u2605 ' : ''}Bid ${r.bid}</span>
          <span class="ohb-rec-bar"><span class="ohb-rec-fill" style="width:${barWidth}%"></span></span>
          <span class="ohb-rec-pct">${pct}%</span>
        </div>`;
    }
    html += `<div class="ohb-rec-value">Value: ${value.toFixed(3)}</div>`;
    recEl.innerHTML = html;
    // Autoplay: auto-submit the top bid
    if (autoplayEnabled && recs.length > 0) {
      scheduleAutoplay('bid', recs[0].bid);
    }
  }

  function showPlayRecommendation(recs, value) {
    if (!recEl) return;
    let html = '<div class="ohb-rec-title">PLAY RECOMMENDATION</div>';
    for (let i = 0; i < recs.length && i < 6; i++) {
      const r = recs[i];
      const pct = (r.prob * 100).toFixed(1);
      const barWidth = Math.max(2, r.prob * 100);
      const isTop = i === 0;
      const name = cardName(r.card);
      const color = cardColor(r.card);
      html += `
        <div class="ohb-rec-row ${isTop ? 'ohb-rec-top' : ''}">
          <span class="ohb-rec-label" style="color:${color}">${isTop ? '\u2605 ' : ''}${name}</span>
          <span class="ohb-rec-bar"><span class="ohb-rec-fill" style="width:${barWidth}%"></span></span>
          <span class="ohb-rec-pct">${pct}%</span>
        </div>`;
    }
    html += `<div class="ohb-rec-value">Value: ${value.toFixed(3)}</div>`;
    recEl.innerHTML = html;
    // Autoplay: auto-play the top card
    if (autoplayEnabled && recs.length > 0) {
      scheduleAutoplay('play', recs[0].card);
    }
  }

  // ─── Initialization ────────────────────────────────────────────────

  // Load settings immediately
  if (typeof chrome !== 'undefined' && chrome.storage) {
    chrome.storage.local.get(['backendUrl', 'snapshotPath', 'autoplayEnabled', 'autoplayDelay'], (items) => {
      if (items.backendUrl) backendUrl = items.backendUrl;
      if (items.snapshotPath) snapshotPath = items.snapshotPath;
      if (items.autoplayEnabled) autoplayEnabled = true;
      if (items.autoplayDelay !== undefined) autoplayDelay = items.autoplayDelay;
      LOG('Settings loaded: backendUrl =', backendUrl, 'snapshotPath =', snapshotPath, 'autoplay =', autoplayEnabled, 'delay =', autoplayDelay);
      // Sync UI if overlay already created
      const toggle = document.getElementById('ohb-autoplay-toggle');
      if (toggle) toggle.checked = autoplayEnabled;
      const slider = document.getElementById('ohb-delay-slider');
      if (slider) { slider.value = autoplayDelay; }
      const delayLabel = document.getElementById('ohb-delay-value');
      if (delayLabel) delayLabel.textContent = autoplayDelay + 'ms';
      updateAutoplayBadge();
    });
  }

  // Start listening for events from inject.js right away (inject.js is loaded
  // via manifest "world": "MAIN", no manual injection needed)
  window.addEventListener('message', (evt) => {
    if (evt.data && evt.data.type === 'OHB_EVENT') {
      onEvent(evt.data.event, evt.data.data);
    }
  });
  LOG('Listening for OHB_EVENT messages from inject.js');

  // Create the overlay once the DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => createOverlay());
  } else {
    createOverlay();
  }
})();
