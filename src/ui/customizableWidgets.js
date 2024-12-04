// customizableWidgets.js

class CustomizableWidgets {
    constructor() {
        this.widgets = this.loadWidgets();
    }

    // Load widgets from local storage
    loadWidgets() {
        const savedWidgets = localStorage.getItem('customWidgets');
        return savedWidgets ? JSON.parse(savedWidgets) : this.defaultWidgets();
    }

    // Default widgets configuration
    defaultWidgets() {
        return [
            { id: 'balance', name: 'Account Balance', visible: true },
            { id: 'transactions', name: 'Recent Transactions', visible: true },
            { id: 'loyalty', name: 'Loyalty Points', visible: true },
            { id: 'analytics', name: 'Analytics Overview', visible: false }
        ];
    }

    // Get the current widget configuration
    getWidgets() {
        return this.widgets;
    }

    // Add a new widget
    addWidget(widget) {
        this.widgets.push(widget);
        this.saveWidgets();
    }

    // Remove a widget
    removeWidget(widgetId) {
        this.widgets = this.widgets.filter(widget => widget.id !== widgetId);
        this.saveWidgets();
    }

    // Toggle widget visibility
    toggleWidgetVisibility(widgetId) {
        const widget = this.widgets.find(widget => widget.id === widgetId);
        if (widget) {
            widget.visible = !widget.visible;
            this.saveWidgets();
        }
    }

    // Save widgets to local storage
    saveWidgets() {
        localStorage.setItem('customWidgets', JSON.stringify(this.widgets));
    }

// Example usage
    const customizableWidgets = new CustomizableWidgets();
    console.log('Current Widgets:', customizableWidgets.getWidgets());

    // Adding a new widget
    customizableWidgets.addWidget({ id: 'notifications', name: 'Notifications', visible: true });
    console.log('Widgets after adding:', customizableWidgets.getWidgets());

    // Removing a widget
    customizableWidgets.removeWidget('analytics');
    console.log('Widgets after removing:', customizableWidgets.getWidgets());

    // Toggling widget visibility
    customizableWidgets.toggleWidgetVisibility('balance');
    console.log('Widgets after toggling balance visibility:', customizableWidgets.getWidgets());

export default CustomizableWidgets;
