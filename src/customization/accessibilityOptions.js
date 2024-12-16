// accessibilityOptions.js

class AccessibilityOptions {
    constructor() {
        this.userAccessibilitySettings = {}; // Store accessibility settings by user ID
    }

    // Set accessibility options for a user
    setAccessibilityOptions(userId, options) {
        this.userAccessibilitySettings[userId] = options;
        console.log(`Accessibility options set for user ${userId}:`, options);
        return options;
    }

    // Get accessibility options for a user
    getAccessibilityOptions(userId) {
        return this.userAccessibilitySettings[userId] || this.getDefaultAccessibilityOptions();
    }

    // Reset accessibility options to default
    resetAccessibilityOptions(userId) {
        this.userAccessibilitySettings[userId] = this.getDefaultAccessibilityOptions();
        console.log(`Accessibility options reset to default for user ${userId}`);
        return this.userAccessibilitySettings[userId];
    }

    // Get default accessibility options
    getDefaultAccessibilityOptions() {
        return {
            textToSpeech: false,
            highContrast: false,
            keyboardNavigation: true,
            fontSize: 'medium',
        };
    }
}

// Example usage
const accessibilityManager = new AccessibilityOptions();
accessibilityManager.setAccessibilityOptions('user123', { textToSpeech: true, highContrast: true, keyboardNavigation: true, fontSize: 'large' });
const userAccessibility = accessibilityManager.getAccessibilityOptions('user123');
console.log('User Accessibility Options for user123:', userAccessibility);

accessibilityManager.resetAccessibilityOptions('user123');
const defaultAccessibility = accessibilityManager.getAccessibilityOptions('user123');
console.log('Default Accessibility Options for user123:', defaultAccessibility);

export default AccessibilityOptions;
