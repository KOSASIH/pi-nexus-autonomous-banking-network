import pytest
import agents.agent_a
import agents.agent_b

@pytest.fixture
def agent_a():
    return agents.agent_a.AgentA()

@pytest.fixture
def agent_b():
    return agents.agent_b.AgentB()
