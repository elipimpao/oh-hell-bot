/**
 * Card rendering utilities.
 * Cards are integers 0-51: suit = card // 13, rank = card % 13
 * Suits: 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades
 * Ranks: 0=2, 1=3, ..., 12=Ace
 */

const SUITS = ['club', 'diamond', 'heart', 'spade'];
const SUIT_SYMBOLS = ['\u2663', '\u2666', '\u2665', '\u2660'];
const SUIT_COLORS = ['#1a1a1a', '#d32f2f', '#d32f2f', '#1a1a1a'];
const RANKS_SVG = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', '1'];
const RANK_DISPLAY = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];

const SVG_NS = 'http://www.w3.org/2000/svg';
const XLINK_NS = 'http://www.w3.org/1999/xlink';
const CARD_VIEWBOX = '0 0 169.075 244.640';

// Settings (mutable, updated by settings panel)
const CardSettings = {
    cardStyle: 'css-large',  // 'css', 'css-large', or 'svg'
    scale: 1.0,
    fontScale: 1.0,      // text size multiplier (adjustable via slider)
    trumpSuit: -1,        // current trump suit for highlighting (-1 = none)
};

function cardSuit(cardInt) { return Math.floor(cardInt / 13); }
function cardRank(cardInt) { return cardInt % 13; }

function cardSvgId(cardInt) {
    return `${SUITS[cardSuit(cardInt)]}_${RANKS_SVG[cardRank(cardInt)]}`;
}

function cardDisplayName(cardInt) {
    return `${RANK_DISPLAY[cardRank(cardInt)]}${SUIT_SYMBOLS[cardSuit(cardInt)]}`;
}

function cardSuitColor(cardInt) {
    return SUIT_COLORS[cardSuit(cardInt)];
}

/**
 * Create a card element (CSS-rendered with clear rank/suit).
 */
function createCardElement(cardInt, faceUp = true, size = 'normal') {
    const wrapper = document.createElement('div');
    wrapper.className = `card card-${size}`;
    wrapper.dataset.card = cardInt;

    if (!faceUp) {
        wrapper.classList.add('card-back');
        const backInner = document.createElement('div');
        backInner.className = 'card-back-pattern';
        wrapper.appendChild(backInner);
        return wrapper;
    }

    // Trump highlight
    if (CardSettings.trumpSuit >= 0 && cardSuit(cardInt) === CardSettings.trumpSuit) {
        wrapper.classList.add('card-trump');
    }

    wrapper.title = cardDisplayName(cardInt);
    const suit = cardSuit(cardInt);
    const rank = cardRank(cardInt);
    const color = SUIT_COLORS[suit];
    const rankStr = RANK_DISPLAY[rank];
    const suitStr = SUIT_SYMBOLS[suit];

    if (CardSettings.cardStyle === 'svg') {
        // SVG card artwork
        const svg = document.createElementNS(SVG_NS, 'svg');
        svg.setAttribute('viewBox', CARD_VIEWBOX);
        svg.classList.add('card-svg');
        const use = document.createElementNS(SVG_NS, 'use');
        use.setAttributeNS(XLINK_NS, 'href', `/static/cards/svg-cards.svg#${cardSvgId(cardInt)}`);
        svg.appendChild(use);
        wrapper.appendChild(svg);
    } else {
        // CSS card: white background, prominent rank + suit
        wrapper.classList.add('card-css');

        // Large style gets extra class for bigger fonts
        if (CardSettings.cardStyle === 'css-large') {
            wrapper.classList.add('card-style-large');
        }

        // Top-left index
        const topIdx = document.createElement('div');
        topIdx.className = 'card-index card-index-tl';
        topIdx.style.color = color;
        topIdx.innerHTML = `<span class="card-rank">${rankStr}</span><span class="card-suit">${suitStr}</span>`;
        wrapper.appendChild(topIdx);

        // Center suit (large)
        const centerSuit = document.createElement('div');
        centerSuit.className = 'card-center-suit';
        centerSuit.style.color = color;
        centerSuit.textContent = suitStr;
        wrapper.appendChild(centerSuit);

        // Bottom-right index (upside down)
        const botIdx = document.createElement('div');
        botIdx.className = 'card-index card-index-br';
        botIdx.style.color = color;
        botIdx.innerHTML = `<span class="card-rank">${rankStr}</span><span class="card-suit">${suitStr}</span>`;
        wrapper.appendChild(botIdx);
    }

    return wrapper;
}

/**
 * Create a card back element.
 */
function createCardBack(size = 'normal') {
    return createCardElement(0, false, size);
}

window.Card = {
    SUITS, SUIT_SYMBOLS, SUIT_COLORS, RANKS_SVG, RANK_DISPLAY,
    Settings: CardSettings,
    suit: cardSuit,
    rank: cardRank,
    svgId: cardSvgId,
    displayName: cardDisplayName,
    suitColor: cardSuitColor,
    create: createCardElement,
    createBack: createCardBack,
};
