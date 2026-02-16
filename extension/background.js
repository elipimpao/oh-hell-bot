/**
 * background.js — MV3 Service Worker.
 *
 * Handles all network communication (fetch + WebSocket) on behalf of the
 * content script, since content scripts on HTTPS pages cannot reach
 * HTTP localhost due to mixed-content / CORS restrictions.
 *
 * Communication: content script <-> background via chrome.runtime.connect (Port).
 */

const LOG = (...args) => console.log('[OHB-bg]', ...args);

let ws = null;
let port = null;  // active Port to content script

chrome.runtime.onConnect.addListener((p) => {
  LOG('Port connected:', p.name);
  port = p;

  port.onMessage.addListener(async (msg) => {
    LOG('Received from content:', msg.type, msg.type === 'ws_send' ? msg.data?.action : '');
    try {
      if (msg.type === 'create_session') {
        await handleCreateSession(msg);
      } else if (msg.type === 'ws_send') {
        wsSend(msg.data);
      } else if (msg.type === 'close_session') {
        closeWebSocket();
      }
    } catch (e) {
      LOG('Error handling message:', e);
      port.postMessage({ type: 'error', message: String(e) });
    }
  });

  port.onDisconnect.addListener(() => {
    LOG('Port disconnected');
    port = null;
    closeWebSocket();
  });
});

async function handleCreateSession(msg) {
  const { backendUrl, body } = msg;
  LOG('Creating session:', backendUrl, JSON.stringify(body));
  try {
    const resp = await fetch(`${backendUrl}/api/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    LOG('Session response status:', resp.status);
    const data = await resp.json();
    LOG('Session response data:', JSON.stringify(data));
    if (data.error) {
      port?.postMessage({ type: 'session_error', message: data.error });
      return;
    }
    // Session created — open WebSocket
    const wsUrl = backendUrl.replace(/^http/, 'ws') + '/ws/' + data.session_id;
    LOG('Opening WebSocket:', wsUrl);
    openWebSocket(wsUrl);
    port?.postMessage({ type: 'session_created', sessionId: data.session_id });
  } catch (e) {
    LOG('Session creation failed:', e);
    port?.postMessage({ type: 'session_error', message: 'Cannot reach backend at ' + backendUrl });
  }
}

function openWebSocket(url) {
  closeWebSocket();
  ws = new WebSocket(url);
  ws.onopen = () => {
    LOG('WebSocket opened');
    port?.postMessage({ type: 'ws_open' });
  };
  ws.onclose = (evt) => {
    LOG('WebSocket closed:', evt.code, evt.reason);
    ws = null;
    port?.postMessage({ type: 'ws_close' });
  };
  ws.onerror = (evt) => {
    LOG('WebSocket error:', evt);
    port?.postMessage({ type: 'ws_error' });
  };
  ws.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      LOG('WebSocket received:', data.type);
      port?.postMessage({ type: 'ws_message', data });
    } catch (_) { }
  };
}

function wsSend(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    LOG('WebSocket send:', data.action);
    ws.send(JSON.stringify(data));
  } else {
    LOG('WebSocket not open, dropping:', data.action, 'readyState:', ws?.readyState);
  }
}

function closeWebSocket() {
  if (ws) {
    LOG('Closing WebSocket');
    ws.close();
    ws = null;
  }
}
