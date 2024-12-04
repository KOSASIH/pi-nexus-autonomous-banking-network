// realTimeAnalytics.js

import WebSocket from 'ws';

class RealTimeAnalytics {
    constructor(port) {
        this.port = port;
        this.wss = new WebSocket.Server({ port: this.port });
        this.metrics = {}; // Object to store key metrics
        this.clients = new Set(); // Set to store connected clients

        this.initializeWebSocket();
    }

    // Initialize WebSocket server
    initializeWebSocket() {
        this.wss.on('connection', (ws) => {
            console.log('New client connected');
            this.clients.add(ws);

            // Send current metrics to the newly connected client
            ws.send(JSON.stringify(this.metrics));

            // Handle client disconnection
            ws.on('close', () => {
                console.log('Client disconnected');
                this.clients.delete(ws);
            });
        });

        console.log(`WebSocket server is running on ws://localhost:${this.port}`);
    }

    // Update a specific metric
    updateMetric(key, value) {
        this.metrics[key] = value;
        console.log(`Updated metric: ${key} = ${value}`);
        this.broadcastMetrics();
    }

    // Broadcast updated metrics to all connected clients
    broadcastMetrics() {
        const metricsData = JSON.stringify(this.metrics);
        this.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(metricsData);
            }
        });
    }

    // Example usage
    exampleUsage() {
        // Simulate metric updates
        setInterval(() => {
            const randomUsers = Math.floor(Math.random() * 100);
            const randomSales = (Math.random() * 1000).toFixed(2);
            this.updateMetric('activeUsers', randomUsers);
            this.updateMetric('totalSales', randomSales);
        }, 5000); // Update every 5 seconds
    }
}

// Example usage
const analyticsPort = 8080; // Port for WebSocket server
const realTimeAnalytics = new RealTimeAnalytics(analyticsPort);
realTimeAnalytics.exampleUsage();

export default RealTimeAnalytics;
