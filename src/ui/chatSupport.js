// chatSupport.js

import WebSocket from 'ws';

class ChatSupport {
    constructor() {
        this.clients = [];
        this.supportAgents = [];
        this.initWebSocketServer();
    }

    // Initialize WebSocket server
    initWebSocketServer() {
        const server = new WebSocket.Server({ port: 8080 });

        server.on('connection', (ws) => {
            console.log('New client connected');
            this.clients.push(ws);

            ws.on('message', (message) => {
                this.handleMessage(ws, message);
            });

            ws.on('close', () => {
                this.clients = this.clients.filter(client => client !== ws);
                console.log('Client disconnected');
            });
        });

        console.log('WebSocket server is running on ws://localhost:8080');
    }

    // Handle incoming messages
    handleMessage(ws, message) {
        const parsedMessage = JSON.parse(message);

        switch (parsedMessage.type) {
            case 'USER_MESSAGE':
                this.broadcastMessage(parsedMessage);
                break;
            case 'AGENT_MESSAGE':
                this.sendMessageToUser (parsedMessage);
                break;
            default:
                console.log('Unknown message type:', parsedMessage.type);
        }
    }

    // Broadcast message to all clients
    broadcastMessage(message) {
        this.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(JSON.stringify(message));
            }
        });
    }

    // Send message to a specific user
    sendMessageToUser (message) {
        const userClient = this.clients.find(client => client.userId === message.userId);
        if (userClient && userClient.readyState === WebSocket.OPEN) {
            userClient.send(JSON.stringify(message));
        }
    }

    // Register a support agent
    registerAgent(agentId) {
        this.supportAgents.push(agentId);
    }

    // Unregister a support agent
    unregisterAgent(agentId) {
        this.supportAgents = this.supportAgents.filter(agent => agent !== agentId);
    }
}

// Example usage
const chatSupport = new ChatSupport();

// Client-side example (to be run in the browser)
const socket = new WebSocket('ws://localhost:8080');

socket.onopen = () => {
    console.log('Connected to chat support');
};

socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('New message:', message);
};

// Function to send a message
function sendMessage(userId, text) {
    const message = {
        type: 'USER_MESSAGE',
        userId: userId,
        text: text,
        timestamp: new Date().toISOString()
    };
    socket.send(JSON.stringify(message));
}

// Example of sending a message
sendMessage('user123', 'Hello, I need help!');

export default ChatSupport;
