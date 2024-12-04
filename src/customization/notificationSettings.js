// notificationSettings.js

class NotificationSettings {
    constructor(userId) {
        this.userId = userId;
        this.preferences = {
            email: true,
            sms: false,
            push: true,
            newsletter: true,
        };
    }

    // Get current notification preferences
    getPreferences() {
        return this.preferences;
    }

    // Update notification preferences
    updatePreferences(newPreferences) {
        this.preferences = { ...this.preferences, ...newPreferences };
        console.log(`Notification preferences updated for user ${this.userId}:`, this.preferences);
    }

    // Reset to default preferences
    resetToDefault() {
        this.preferences = {
            email: true,
            sms: false,
            push: true,
            newsletter: true,
        };
        console.log(`Notification preferences reset to default for user ${this.userId}.`);
    }
}

// Example usage
const userNotificationSettings = new NotificationSettings('user123');
console.log('Current Preferences:', userNotificationSettings.getPreferences());

userNotificationSettings.updatePreferences({ sms: true, newsletter: false });
console.log('Updated Preferences:', userNotificationSettings.getPreferences());

userNotificationSettings.resetToDefault();
console.log('Preferences after reset:', userNotificationSettings.getPreferences());
