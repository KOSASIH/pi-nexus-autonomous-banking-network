// src/agents/NotificationAgent.js
import Agent from './Agent';

class NotificationAgent extends Agent {
    constructor() {
        super('NotificationAgent');
    }

    sendNotification(user, message) {
        // Logic to send a notification to the user
        this.log(`Sent notification to ${user.username}: ${message}`);
    }
}

export default NotificationAgent;
