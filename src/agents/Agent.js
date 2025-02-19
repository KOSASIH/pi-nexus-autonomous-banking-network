// src/agents/Agent.js
class Agent {
    constructor(name) {
        this.name = name;
    }

    log(message) {
        console.log(`[${this.name}] ${message}`);
    }
}

export default Agent;
