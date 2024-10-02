const natural = require("natural");

const tokenizer = new natural.WordTokenizer();
const text = "Hello, how are you?";
const tokens = tokenizer.tokenize(text);
console.log(tokens);

const classifier = new natural.BayesClassifier();
classifier.addDocument("I am happy.", "positive");
classifier.addDocument("I am sad.", "negative");
classifier.train();
const sentiment = classifier.classify("I am happy today.");
console.log(sentiment);
