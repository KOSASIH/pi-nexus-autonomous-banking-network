// analytics/analytics.js
class Analytics {
    constructor() {
        this.events = []; // Store collected events
    }

    trackEvent(eventName, data) {
        const event = {
            name: eventName,
            data: data,
            timestamp: new Date(),
        };
        this.events.push(event);
        console.log(`Event tracked: ${eventName}`, data);
    }

    getEvents() {
        return this.events;
    }

    clearEvents() {
        this.events = [];
        console.log('All events cleared.');
    }
}

module.exports = Analytics;
