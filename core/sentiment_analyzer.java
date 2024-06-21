import java.util.ArrayList;
import java.util.List;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;

public class SentimentAnalyzer {
  private Tokenizer tokenizer;

  public SentimentAnalyzer() {
    TokenizerModel model = new TokenizerModel("en-token.bin");
    tokenizer = new TokenizerME(model);
  }

  public String analyzeSentiment(String text) {
    List<String> tokens = tokenizer.tokenize(text);
    int positiveTokens = 0;
    int negativeTokens = 0;

    for (String token : tokens) {
      if (token.contains("good") || token.contains("great")) {
        positiveTokens++;
      } else if (token.contains("bad") || token.contains("terrible")) {
        negativeTokens++;
      }
    }

    if (positiveTokens > negativeTokens) {
      return "Positive";
    } else if (negativeTokens > positiveTokens) {
      return "Negative";
    } else {
      return "Neutral";
    }
  }

  public static void main(String[] args) {
    SentimentAnalyzer analyzer = new SentimentAnalyzer();
    String text = "I love this bank!";
    String sentiment = analyzer.analyzeSentiment(text);
    System.out.println("Sentiment: " + sentiment);
  }
}
