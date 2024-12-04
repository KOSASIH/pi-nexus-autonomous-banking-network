// userPreferences.js

class UserPreferences {
    constructor() {
        this.preferences = {}; // Object to store user preferences
    }

    // Set user preference
    setPreference(userId, key, value) {
        if (!this.preferences[userId]) {
            this.preferences[userId] = {};
        }
        this.preferences[userId][key] = value;
        console.log(`Preference set for User ${userId}: ${key} = ${value}`);
    }

    // Get user preference
    getPreference(userId, key) {
        if (this.preferences[userId] && this.preferences[userId][key] !== undefined) {
            return this.preferences[userId][key];
        } else {
            console.log(`Preference ${key} not found for User ${userId}.`);
            return null; // Return null if preference not found
        }
    }

    // Get all preferences for a user
    getAllPreferences(userId) {
        if (this.preferences[userId]) {
            return this.preferences[userId];
        } else {
            console.log(`No preferences found for User ${userId}.`);
            return {}; // Return an empty object if no preferences found
        }
    }

    // Example usage
    exampleUsage() {
        // Setting preferences for users
        this.setPreference('user1', 'theme', 'dark');
        this.setPreference('user1', 'language', 'en');
        this.setPreference('user1', 'notifications', true);

        this.setPreference('user2', 'theme', 'light');
        this.setPreference('user2', 'language', 'fr');

        // Getting preferences
        console.log(`User  1 Theme: ${this.getPreference('user1', 'theme')}`); // Should return 'dark'
        console.log(`User  2 Language: ${this.getPreference('user2', 'language')}`); // Should return 'fr'
        console.log(`User  1 Notifications: ${this.getPreference('user1', 'notifications')}`); // Should return true

        // Getting all preferences for a user
        console.log('User  1 Preferences:', this.getAllPreferences('user1')); // Should return all preferences for user 1
        console.log('User  2 Preferences:', this.getAllPreferences('user2')); // Should return all preferences for user 2
    }
}

// Example usage
const userPreferences = new UserPreferences();
userPreferences.exampleUsage();

export default UserPreferences;
