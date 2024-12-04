// darkModeToggle.js

class DarkModeToggle {
    constructor() {
        this.toggleButton = document.getElementById('dark-mode-toggle');
        this.init();
    }

    // Initialize the dark mode toggle
    init() {
        this.loadTheme();
        this.toggleButton.addEventListener('click', () => this.toggleTheme());
    }

    // Load the user's theme preference from local storage
    loadTheme() {
        const currentTheme = localStorage.getItem('theme');
        if (currentTheme) {
            document.body.classList.toggle('dark-mode', currentTheme === 'dark');
        } else {
            // Default to light mode
            document.body.classList.remove('dark-mode');
        }
    }

    // Toggle between light and dark themes
    toggleTheme() {
        const isDarkMode = document.body.classList.toggle('dark-mode');
        localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
    }
}

// Example usage
document.addEventListener('DOMContentLoaded', () => {
    new DarkModeToggle();
});
