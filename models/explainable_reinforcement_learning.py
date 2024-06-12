import explainable_rl

# Define an explainable RL agent
def explainable_rl_agent(env, policy):
    agent = explainable_rl.ExplainableRLAgent(env, policy)
    return agent

# Use the explainable RL agent to make decisions
def make_decision(agent, state):
    action, explanation = agent.act(state)
    return action, explanation
