// notifications.js

class Notifications {
    constructor() {
        this.notifications = [];
        this.init();
    }

    // Initialize the notification system
    init() {
        this.render();
    }

    // Add a new notification
    addNotification(message) {
        const notification = {
            id: this.notifications.length + 1,
            message,
            timestamp: new Date(),
        };
        this.notifications.push(notification);
        this.renderNotification(notification);
    }

    // Render the notification area
    render() {
        const notificationContainer = document.createElement('div');
        notificationContainer.id = 'notifications';
        notificationContainer.innerHTML = `
            <h2>Notifications</h2>
            <ul id="notification-list"></ul>
        `;
        document.body.appendChild(notificationContainer);
    }

    // Render a single notification
    renderNotification(notification) {
        const notificationList = document.getElementById('notification-list');
        const notificationElement = document.createElement('li');
        notificationElement.innerText = `${notification.timestamp.toLocaleTimeString()}: ${notification.message}`;
        notificationList.appendChild(notificationElement);
    }
}

// Example usage
const notifications = new Notifications();
notifications.addNotification("Transaction of $200 completed.");
notifications.addNotification("New alert: Your account balance is low.");

export default Notifications;
