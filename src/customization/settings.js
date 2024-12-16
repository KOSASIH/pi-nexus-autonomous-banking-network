// customization/settings.js
class UserSettings {
    constructor(userId) {
        this.userId = userId;
        this.settings = {
            language: 'en',
            notifications: true,
            privacy: 'public',
        };
    }

    updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        console.log(`Settings updated for user ${this.userId}:`, this.settings);
        return this.settings;
    }

    getSettings() {
        return this.settings;
    }

    resetSettings() {
        this.settings = {
            language: 'en',
            notifications: true,
            privacy: 'public',
        };
        console.log(`Settings reset for user ${this.userId}.`);
        return this.settings;
    }
}

class SettingsManager {
    constructor() {
        this.usersSettings = {}; // Store settings for multiple users
    }

    createUser Settings(userId) {
        if (this.usersSettings[userId]) {
            throw new Error('User  settings already exist.');
        }
        this.usersSettings[userId] = new UserSettings(userId);
        console.log(`User  settings created for user ${userId}.`);
        return this.usersSettings[userId];
    }

    getUser Settings(userId) {
        const userSettings = this.usersSettings[userId];
        if (!userSettings) {
            throw new Error('User  settings not found.');
        }
        return userSettings.getSettings();
    }

    updateUser Settings(userId, newSettings) {
        const userSettings = this.usersSettings[userId];
        if (!userSettings) {
            throw new Error('User  settings not found.');
        }
        return userSettings.updateSettings(newSettings);
    }

    resetUser Settings(userId) {
        const userSettings = this.usersSettings[userId];
        if (!userSettings) {
            throw new Error('User  settings not found.');
        }
        return userSettings.resetSettings();
    }
}

module.exports = SettingsManager;
