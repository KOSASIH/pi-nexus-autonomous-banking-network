// tests/analytics.test.js
const Analytics = require('../analytics'); // Assuming you have an analytics module

describe('Analytics Module', () => {
    test('should track user activity correctly', () => {
        const activity = { userId: 'user123', action: 'login' };
        Analytics.trackActivity(activity);
        expect(Analytics.getActivityLog()).toContainEqual(activity);
    });

    test('should return correct analytics data', () => {
        const data = Analytics.getAnalyticsData();
        expect(data).toHaveProperty('totalUsers');
        expect(data.totalUsers).toBeGreaterThan(0);
    });
});
