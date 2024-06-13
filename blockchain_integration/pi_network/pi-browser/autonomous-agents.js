import { AutonomousAgent } from 'autonomous-agent-sdk';

class AutonomousAgents {
  constructor() {
    this.autonomousAgent = new AutonomousAgent();
  }

  async createAgent(agentConfig) {
    const agent = await this.autonomousAgent.create(agentConfig);
    return agent;
  }

  async executeAgentTask(agent, task) {
    const result = await this.autonomousAgent.executeTask(agent, task);
    return result;
  }
}

export default AutonomousAgents;
