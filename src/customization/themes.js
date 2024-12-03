// customization/themes.js
class Theme {
    constructor(name, properties) {
        this.id = Theme.incrementId();
        this.name = name;
        this.properties = properties; // e.g., { backgroundColor: '#fff', textColor: '#000' }
        this.createdAt = new Date();
    }

    static incrementId() {
        if (!this.currentId) {
            this.currentId = 1;
        } else {
            this.currentId++;
        }
        return this.currentId;
    }
}

class ThemeManager {
    constructor() {
        this.themes = []; // Store themes
    }

    addTheme(name, properties) {
        const theme = new Theme(name, properties);
        this.themes.push(theme);
        console.log(`Theme added: ${name}`);
        return theme;
    }

    getThemes() {
        return this.themes;
    }

    getThemeById(themeId) {
        const theme = this.themes.find(t => t.id === themeId);
        if (!theme) {
            throw new Error('Theme not found.');
        }
        return theme;
    }

    updateTheme(themeId, updatedProperties) {
        const theme = this.getThemeById(themeId);
        Object.assign(theme.properties, updatedProperties);
        console.log(`Theme ${themeId} updated.`);
        return theme;
    }

    deleteTheme(themeId) {
        const index = this.themes.findIndex(t => t.id === themeId);
        if (index === -1) {
            throw new Error('Theme not found.');
        }
        this.themes.splice(index, 1);
        console.log(`Theme ${themeId} deleted.`);
    }
}

module.exports = ThemeManager;
