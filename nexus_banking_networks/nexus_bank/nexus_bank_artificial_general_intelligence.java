import java.util.ArrayList;
import java.util.List;

public class ArtificialGeneralIntelligence {
    private List<String> knowledgeBase;

    public ArtificialGeneralIntelligence() {
        knowledgeBase = new ArrayList<>();
    }

    public void learn(String concept) {
        knowledgeBase.add(concept);
    }

    public String reason(String question) {
        // Reason about the knowledge base to answer the question
        return "Answer: " + question;
    }

    public static void main(String[] args) {
        ArtificialGeneralIntelligence agi = new ArtificialGeneralIntelligence();
        agi.learn("The capital of France is Paris.");
        agi.learn("The capital of Germany is Berlin.");
        System.out.println(agi.reason("What is the capital of France?"));
    }
}
