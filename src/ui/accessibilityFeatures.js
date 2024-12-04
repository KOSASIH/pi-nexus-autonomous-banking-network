// accessibilityFeatures.js

class AccessibilityFeatures {
    constructor() {
        this.highContrastMode = false;
        this.init();
    }

    // Initialize accessibility features
    init() {
        this.setupKeyboardNavigation();
        this.setupHighContrastToggle();
    }

    // Setup keyboard navigation for interactive elements
    setupKeyboardNavigation() {
        const focusableElements = 'a, button, input, select, textarea';
        const elements = document.querySelectorAll(focusableElements);
        elements.forEach(element => {
            element.setAttribute('tabindex', '0'); // Make elements focusable
            element.addEventListener('focus', () => {
                element.classList.add('focus-outline');
            });
            element.addEventListener('blur', () => {
                element.classList.remove('focus-outline');
            });
        });
    }

    // Toggle high contrast mode
    setupHighContrastToggle() {
        const toggleButton = document.getElementById('high-contrast-toggle');
        toggleButton.addEventListener('click', () => this.toggleHighContrast());
    }

    // Function to toggle high contrast mode
    toggleHighContrast() {
        this.highContrastMode = !this.highContrastMode;
        document.body.classList.toggle('high-contrast', this.highContrastMode);
        localStorage.setItem('highContrast', this.highContrastMode);
    }

    // Load high contrast preference from local storage
    loadHighContrastPreference() {
        const highContrast = localStorage.getItem('highContrast');
        if (highContrast === 'true') {
            this.highContrastMode = true;
            document.body.classList.add('high-contrast');
        }
    }
}

// Example usage
document.addEventListener('DOMContentLoaded', () => {
    const accessibility = new AccessibilityFeatures();
    accessibility.loadHighContrastPreference();
});
