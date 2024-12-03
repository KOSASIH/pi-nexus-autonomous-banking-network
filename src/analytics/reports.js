// analytics/reports.js
const Analytics = require('./analytics');

class Reports {
    constructor(analytics) {
        this.analytics = analytics;
    }

    generateEventReport() {
        const events = this.analytics.getEvents();
        const report = events.map(event => ({
            name: event.name,
            data: event.data,
            timestamp: event.timestamp.toISOString(),
        }));

        return {
            totalEvents: events.length,
            events: report,
        };
    }

    generateSummaryReport() {
        const events = this.analytics.getEvents();
        const summary = {};

        events.forEach(event => {
            if (!summary[event.name]) {
                summary[event.name] = 0;
            }
            summary[event.name]++;
        });

        return {
            totalEvents: events.length,
            summary: summary,
        };
    }
}

module.exports = Reports;
