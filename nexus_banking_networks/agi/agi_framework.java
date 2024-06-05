// agi_framework.java
import java.util.ArrayList;
import java.util.List;

public class AGIFramework {
    private List<Agent> agents;

    public AGIFramework() {
        this.agents = new ArrayList<>();
    }

    public void addAgent(Agent agent) {
        agents.add(agent);
    }

    public void processInput(String input) {
        for (Agent agent : agents) {
            agent.processInput(input);
        }
    }

    public String getResponse() {
        String response = "";
        for (Agent agent : agents) {
            response += agent.getResponse() + "\n";
        }
        return response;
    }
}

class Agent {
    private String name;
    private String response;

    public Agent(String name) {
        this.name = name;
    }

    public void processInput(String input) {
        // Implement agent logic here
        response = "Agent " + name + " processed input: " + input;
    }

    public String getResponse() {
        return response;
    }
}

// Example usage:
AGIFramework agi = new AGIFramework();
Agent agent1 = new Agent("Agent 1");
Agent agent2 = new Agent("Agent 2");
agi.addAgent(agent1);
agi.addAgent(agent2);
agi.processInput("Hello, AGI!");
System.out.println(agi.getResponse());
