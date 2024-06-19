const nlp = require('compromise');

class NaturalLanguageProcessingController {
  async analyzeFeedback(req, res) {
    const { text } = req.body;
    const doc = nlp(text);
    const analysis = doc.sentiment().terms().out('array');
    res.json({ analysis });
  }
}
