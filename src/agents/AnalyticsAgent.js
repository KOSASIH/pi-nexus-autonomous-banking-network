import Agent from './Agent';

class AnalyticsAgent extends Agent {
    constructor() {
        super('AnalyticsAgent');
        this.userActions = [];
    }

    logUser Action(action) {
        this.userActions.push(action);
        this.log(`User  action logged: ${action}`);
    }

    generateUser Report() {
        // Logic to generate a report based on user actions
        return this.userActions;
    }
}

export default AnalyticsAgent;
