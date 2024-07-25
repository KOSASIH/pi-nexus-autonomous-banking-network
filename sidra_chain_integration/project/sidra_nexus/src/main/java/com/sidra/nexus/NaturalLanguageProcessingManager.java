package com.sidra.nexus;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

public class NaturalLanguageProcessingManager {
    private StanfordCoreNLP pipeline;

    public NaturalLanguageProcessingManager() {
        pipeline = new StanfordCoreNLP();
    }

    public void analyzeText(String text) {
        // Analyze text using natural language processing
        Annotation annotation = pipeline.process(text);

        // Perform sentiment analysis
        performSentimentAnalysis(annotation);

        // Perform entity recognition
        performEntityRecognition(annotation);

        // Perform part-of-speech tagging
        performPartOfSpeechTagging(annotation);

        // Perform named entity recognition
        performNamedEntityRecognition(annotation);

        // Perform dependency parsing
        performDependencyParsing(annotation);

        // Perform coreference resolution
        performCoreferenceResolution(annotation);
    }

    private void performSentimentAnalysis(Annotation annotation) {
        for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
            Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
            int sentimentType = RNNCoreAnnotations.getPredictedClass(tree);
            String sentiment = getSentiment(sentimentType);
            System.out.println("Sentiment: " + sentiment);
        }
    }

    private String getSentiment(int sentimentType) {
        switch (sentimentType) {
            case 0:
            case 1:
                return "Very Negative";
            case 2:
                return "Negative";
            case 3:
                return "Neutral";
            case 4:
                return "Positive";
            case 5:
                return "Very Positive";
            default:
                return "Unknown";
        }
    }

    private void performEntityRecognition(Annotation annotation) {
        for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                String entity = token.get(CoreAnnotations.NamedEntityTagAnnotation.class);
                System.out.println("Entity: " + entity);
            }
        }
    }

    private void performPartOfSpeechTagging(Annotation annotation) {
        for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                System.out.println("Part of Speech: " + pos);
            }
        }
    }

    private void performNamedEntityRecognition(Annotation annotation) {
        for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                String ner = token.get(CoreAnnotations.NamedEntityTagAnnotation.class);
                System.out.println("Named Entity: " + ner);
            }
        }
    }

    private void performDependencyParsing(Annotation annotation) {
        for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
            Tree tree = sentence.get(CoreAnnotations.TreeAnnotation.class);
            System.out.println("Dependency Parsing: " + tree);
        }
    }

    private void performCoreferenceResolution(Annotation annotation) {
        for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
            for (CoreferenceChain chain : annotation.get(CoreAnnotations.CorefChainAnnotation.class).getChains().values()) {
                System.out.println("Coreference Resolution: " + chain);
            }
        }
    }
          }
