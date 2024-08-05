// Eonix NLP
const EonixNLP = {
  // Engine
  engine: 'NLTK',
  // Models
  models: [
    {
      name: 'Word2Vec',
      architecture: 'Word2Vec',
      dimensions: 128,
    },
    {
      name: 'BERT',
      architecture: 'BERT',
      dimensions: 768,
    },
  ],
  // Text Processing
  textProcessing: {
    tokenization: 'WordPiece',
    stopWords: ['the', 'and', 'a'],
  },
};
