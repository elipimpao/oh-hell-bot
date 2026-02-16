/**
 * inject.js — Runs in the PAGE's main world (not the content script's isolated world).
 * Hooks console.log to intercept Trickster Cards game events and forwards them
 * to the content script via window.postMessage.
 */
(function () {
  'use strict';

  // _origLog is set below after hooking console.log; use this early-stage ref
  const _earlyLog = console.log.bind(console);
  _earlyLog('[OHB-inject] inject.js loaded in page world');

  const SUIT_MAP = { '\u2663': 0, '\u2666': 1, '\u2665': 2, '\u2660': 3 };
  const RANK_MAP = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
    '8': 6, '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
  };

  /** Parse a Trickster card symbol like "\u26635" or "\u2660A" or "\u266510" into card integer. */
  function parseCard(sym) {
    sym = sym.trim();
    if (!sym) return -1;
    const suit = SUIT_MAP[sym[0]];
    if (suit === undefined) return -1;
    const rankStr = sym.slice(1);
    const rank = RANK_MAP[rankStr];
    if (rank === undefined) return -1;
    return suit * 13 + rank;
  }

  /** Parse a comma-separated list of card symbols. */
  function parseCardList(str) {
    return str.split(',').map(s => parseCard(s)).filter(c => c >= 0);
  }

  /** Extract player name. Strips rating badge (👤1390) and action tags like (auto). */
  function cleanPlayerName(raw) {
    return raw.replace(/\s*\ud83d\udc64\d+/g, '').replace(/\s*\(auto\)/gi, '').trim();
  }

  function post(event, data) {
    _origLog('[OHB-inject] POST:', event, JSON.stringify(data).substring(0, 200));
    window.postMessage({ type: 'OHB_EVENT', event, data }, '*');
  }

  // Hook console.log
  const _origLog = console.log;
  console.log = function (...args) {
    _origLog.apply(console, args);
    try {
      const msg = args.map(a => (typeof a === 'string' ? a : String(a))).join(' ');
      processMessage(msg);
    } catch (_) { /* never break the page */ }
  };

  function processMessage(msg) {
    // --- Game lifecycle ---
    if (msg.includes('CreateGame()') || msg.includes('SYNC: CreateGame()')) {
      post('create_game', {});
      return;
    }
    if (msg.includes('RejoinGame()') || msg.includes('SYNC: RejoinGame()')) {
      post('rejoin_game', {});
      return;
    }

    // --- Player joined (contains seat map) ---
    // Format: "Name <- onPlayerJoined:\n\tPlayer2 (seat = 1, id = ...)\n\tplayers =\n\tName (seat = 0, id = ...)"
    if (msg.includes('<- onPlayerJoined:')) {
      const localMatch = msg.match(/^(.+?)\s*<-\s*onPlayerJoined:/);
      const localName = localMatch ? cleanPlayerName(localMatch[1]) : null;
      // Extract all "Name (seat = N" patterns
      const seatRegex = /\t(.+?)\s*\(seat\s*=\s*(\d+)/g;
      const players = {};
      let m;
      while ((m = seatRegex.exec(msg)) !== null) {
        players[cleanPlayerName(m[1])] = parseInt(m[2]);
      }
      post('player_joined', { localName, players });
      return;
    }

    // --- Cards dealt ---
    // "Name <- onCardsDealt: \u26635,\u2663J,..."
    if (msg.includes('<- onCardsDealt:')) {
      const localMatch = msg.match(/^(.+?)\s*<-\s*onCardsDealt:/);
      const localName = localMatch ? cleanPlayerName(localMatch[1]) : null;
      const match = msg.match(/onCardsDealt:\s*(.+)/);
      if (match) {
        const cards = parseCardList(match[1]);
        post('cards_dealt', { cards, localName });
      }
      return;
    }

    // --- Player bid ---
    // "Name <- onPlayerBid: PlayerName bid 1202"
    if (msg.includes('<- onPlayerBid:')) {
      const match = msg.match(/onPlayerBid:\s*(.+?)\s+bid\s+(\d+)/);
      if (match) {
        const name = cleanPlayerName(match[1]);
        const bidCode = parseInt(match[2]);
        const bidValue = bidCode - 1200;
        post('player_bid', { name, bidValue });
      }
      return;
    }

    // --- Begin bid (whose turn) ---
    // "Name <- onBeginBid: PlayerName" or "Name <- onBeginBid: PlayerName (auto)"
    if (msg.includes('<- onBeginBid:')) {
      const match = msg.match(/onBeginBid:\s*(.+)/);
      if (match) {
        let name = match[1].replace(/\(auto\)/g, '').trim();
        name = cleanPlayerName(name);
        post('begin_bid', { name });
      }
      return;
    }

    // --- Hand started (play phase) ---
    if (msg.includes('<- onHandStarted')) {
      post('hand_started', {});
      return;
    }

    // --- Card played ---
    // "Name <- onCardPlayed: \u26635 by PlayerName, claiming=false"
    if (msg.includes('<- onCardPlayed:')) {
      const match = msg.match(/onCardPlayed:\s*(\S+)\s+by\s+(.+?),\s*claiming/);
      if (match) {
        const card = parseCard(match[1]);
        const name = cleanPlayerName(match[2]);
        if (card >= 0) {
          post('card_played', { card, name });
        }
      }
      return;
    }

    // --- User's turn to play ---
    // "Name is up. Legal cards: \u26635,\u2663J,..."
    if (msg.includes('is up. Legal cards:')) {
      const match = msg.match(/(.+?)\s+is up\.\s+Legal cards:\s*(.*)/);
      if (match) {
        const name = cleanPlayerName(match[1]);
        const legalCards = match[2] ? parseCardList(match[2]) : [];
        post('player_turn', { name, legalCards });
      }
      return;
    }

    // --- Trick resolved ---
    // "Name <- trick \u26635,\u266610 won by PlayerName with \u266610; scored to PlayerName"
    if (msg.includes('<- trick') && msg.includes('won by')) {
      const match = msg.match(/trick\s+(.+?)\s+won by\s+(.+?)\s+with/);
      if (match) {
        const trickCards = parseCardList(match[1]);
        const winner = cleanPlayerName(match[2]);
        post('trick_won', { trickCards, winner });
      }
      return;
    }

    // --- Hand ended ---
    // "Name <- handEnded; took 7; score is 0"
    if (msg.includes('<- handEnded')) {
      const match = msg.match(/handEnded;\s*took\s+(\d+);\s*score is\s+(\d+)/);
      post('hand_ended', {
        tricksTaken: match ? parseInt(match[1]) : 0,
        score: match ? parseInt(match[2]) : 0
      });
      return;
    }

    // --- Bid instructions shown (user's turn to bid) ---
    if (msg.includes('show #bid-instructions')) {
      post('bid_prompt', {});
      return;
    }

    // --- Lead message (user leads the trick) ---
    if (msg.includes('show #lead-message')) {
      post('lead_prompt', {});
      return;
    }
  }

  // ─── Autoplay: DOM interaction for automatic bid/play ──────────────

  const RANK_WORDS = ['Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace'];
  const SUIT_WORDS = ['Clubs', 'Diamonds', 'Hearts', 'Spades'];

  /** Convert card integer to aria-label search string, e.g. 11 → "King of Clubs" */
  function cardToAriaLabel(cardInt) {
    const rank = cardInt % 13;
    const suit = Math.floor(cardInt / 13);
    return RANK_WORDS[rank] + ' of ' + SUIT_WORDS[suit];
  }

  /** Try to submit a bid value by interacting with the bid UI. */
  function autoplayBid(bidValue) {
    _origLog('[OHB-inject] autoplayBid:', bidValue);

    const bidPanel = document.getElementById('bid-instructions');
    if (!bidPanel) {
      _origLog('[OHB-inject] autoplayBid: #bid-instructions not found');
    }

    const container = bidPanel || document;
    const buttons = container.querySelectorAll('button');
    _origLog('[OHB-inject] autoplayBid: found', buttons.length, 'buttons');

    // Strategy 1: Trickster uses numbered buttons ("0", "1", "2", ...).
    // Click the button whose text matches the bid value exactly.
    for (const btn of buttons) {
      const text = (btn.textContent || '').trim();
      if (text === String(bidValue)) {
        _origLog('[OHB-inject] autoplayBid: clicking numbered button:', text);
        btn.click();
        return;
      }
    }

    // Strategy 2: Look for input field + submit button (other possible UI)
    const inputs = container.querySelectorAll('input[type="number"], input[type="range"], input[type="text"]');
    if (inputs.length > 0) {
      const input = inputs[0];
      _origLog('[OHB-inject] autoplayBid: setting input value to', bidValue);
      const nativeSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      nativeSetter.call(input, String(bidValue));
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
      // Click submit button after value propagates
      setTimeout(() => {
        for (const btn of buttons) {
          const t = (btn.textContent || '').trim().toLowerCase();
          if (t.includes('bid') || t.includes('ok') || t.includes('submit')) {
            _origLog('[OHB-inject] autoplayBid: clicking submit button:', t);
            btn.click();
            return;
          }
        }
      }, 100);
      return;
    }

    _origLog('[OHB-inject] autoplayBid: no matching button or input found for bid', bidValue);
  }

  /** Try to play a card by clicking it in the hand. */
  function autoplayCard(cardInt) {
    const label = cardToAriaLabel(cardInt);
    _origLog('[OHB-inject] autoplayCard:', cardInt, '→', label);

    // Strategy 1: Find by aria-label containing the card name (most reliable)
    // Cards in hand that are legal to play typically have class "legal"
    const candidates = document.querySelectorAll('[aria-label]');
    let clicked = false;

    for (const el of candidates) {
      const ariaLabel = el.getAttribute('aria-label') || '';
      if (ariaLabel.includes(label)) {
        // Prefer elements with "legal" class (playable cards)
        const classes = el.className || '';
        _origLog('[OHB-inject] autoplayCard: found match:', el.tagName, 'class="' + classes + '"', 'aria-label="' + ariaLabel + '"');
        if (classes.includes('legal') || classes.includes('hand')) {
          _origLog('[OHB-inject] autoplayCard: clicking legal/hand card');
          el.click();
          clicked = true;
          break;
        }
      }
    }

    // Strategy 2: If no legal-class match, click any card with matching aria-label
    if (!clicked) {
      for (const el of candidates) {
        const ariaLabel = el.getAttribute('aria-label') || '';
        if (ariaLabel.includes(label) && (el.className || '').includes('card')) {
          _origLog('[OHB-inject] autoplayCard: clicking card element (fallback)');
          el.click();
          clicked = true;
          break;
        }
      }
    }

    // Strategy 3: broadest fallback — any element with matching aria-label
    if (!clicked) {
      for (const el of candidates) {
        const ariaLabel = el.getAttribute('aria-label') || '';
        if (ariaLabel.includes(label)) {
          _origLog('[OHB-inject] autoplayCard: clicking any matching element (last resort)');
          el.click();
          clicked = true;
          break;
        }
      }
    }

    if (!clicked) {
      _origLog('[OHB-inject] autoplayCard: NO matching element found for', label);
      // Debug: log all card elements
      const cards = document.querySelectorAll('.card[aria-label]');
      _origLog('[OHB-inject] autoplayCard: all card elements with aria-label:');
      cards.forEach((c, i) => {
        _origLog('[OHB-inject]   card', i, ':', c.className, '|', c.getAttribute('aria-label'));
      });
    }
  }

  // Listen for autoplay commands from the content script
  window.addEventListener('message', (evt) => {
    if (evt.data && evt.data.type === 'OHB_AUTOPLAY') {
      _origLog('[OHB-inject] Autoplay command received:', evt.data.action, evt.data.value);
      try {
        if (evt.data.action === 'bid') {
          autoplayBid(evt.data.value);
        } else if (evt.data.action === 'play') {
          autoplayCard(evt.data.value);
        }
      } catch (e) {
        _origLog('[OHB-inject] Autoplay error:', e);
      }
    }
  });
})();
