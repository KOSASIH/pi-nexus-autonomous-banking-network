#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

class ArtificialIntelligence {
public:
    ArtificialIntelligence() {}

    void learn(std::vector<std::string> data) {
        // Learn from the data
        for (const auto& item : data) {
            // Process the item
            processItem(item);
        }
    }

    void reason(std::string question) {
        // Reason about the knowledge base to answer the question
        std::string answer = getAnswer(question);
        std::cout << "Answer: " << answer << std::endl;
    }

private:
    void processItem(std::string item) {
        // Process the item and add it to the knowledge base
        knowledgeBase_.push_back(item);
    }

    std::string getAnswer(std::string question) {
        // Get the answer from the knowledge base
        for (const auto& item : knowledgeBase_) {
            if (item.find(question)!= std::string::npos) {
                return item;
            }
        }
        return "Unknown";
    }

    std::vector<std::string> knowledgeBase_;
};

int main() {
    ArtificialIntelligence ai;
    std::vector<std::string> data = {"The capital of France is Paris.", "The capital of Germany is Berlin."};
    ai.learn(data);
    ai.reason("What is the capital of France?");
    return 0;
}
