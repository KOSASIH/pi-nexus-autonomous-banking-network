import java.util.ArrayList;
import java.util.List;

public class ArtificialLife {
    private List<Agent> agents;

    public ArtificialLife() {
        agents = new ArrayList<>();
    }

    public void addAgent(Agent agent) {
        agents.add(agent);
    }

    public void simulateBankingScenario() {
        // Implement artificial life simulation logic for banking scenario
    }

    public static void main(String[] args) {
        ArtificialLife al = new ArtificialLife();
        al.addAgent(new CustomerAgent());
        al.addAgent(new BankAgent());

        al.simulateBankingScenario();
    }
}
