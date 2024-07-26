import WebSocket from 'ws';

class WebSocketServer {
  constructor() {
    this.wss = new WebSocket.Server({ port: 8080 });
    this.clients = {};
  }

  async start() {
    this.wss.on('connection', (ws) => {
      const clientId = ws.upgradeReq.headers['sec-websocket-key'];
      this.clients[clientId] = ws;

      console.log(`Client connected: ${clientId}`);

      ws.on('message', (message) => {
        console.log(`Received message from client ${clientId}: ${message}`);
        // Handle incoming message from client
      });

      ws.on('close', () => {
        delete this.clients[clientId];
        console.log(`Client disconnected: ${clientId}`);
      });
    });
  }

  async broadcast(message) {
    Object.values(this.clients).forEach((ws) => {
      ws.send(message);
    });
  }
}

export { WebSocketServer };
