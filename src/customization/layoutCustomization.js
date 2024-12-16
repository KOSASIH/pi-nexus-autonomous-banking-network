// layoutCustomization.js

class LayoutCustomization {
    constructor() {
        this.userLayouts = {}; // Store layout preferences by user ID
    }

    // Set layout preferences for a user
    setLayout(userId, layoutOptions) {
        this.userLayouts[userId] = layoutOptions;
        console.log(`Layout set for user ${userId}:`, layoutOptions);
        return layoutOptions;
    }

    // Get layout preferences for a user
    getLayout(userId) {
        return this.userLayouts[userId] || this.getDefaultLayout();
    }

    // Reset layout to default
    resetLayout(userId) {
        this.userLayouts[userId] = this.getDefaultLayout();
        console.log(`Layout reset to default for user ${userId}`);
        return this.userLayouts[userId];
    }

    // Get default layout options
    getDefaultLayout() {
        return {
            theme: 'light',
            fontSize: 'medium',
            layoutStyle: 'grid',
            showSidebar: true,
        };
    }
}

// Example usage
const layoutManager = new LayoutCustomization();
layoutManager.setLayout('user123', { theme: 'dark', fontSize: 'large', layoutStyle: 'list', showSidebar: false });
const userLayout = layoutManager.getLayout('user123');
console.log('User  Layout for user123:', userLayout);

layoutManager.resetLayout('user123');
const defaultLayout = layoutManager.getLayout('user123');
console.log('Default Layout for user123:', defaultLayout);

export default LayoutCustomization;
