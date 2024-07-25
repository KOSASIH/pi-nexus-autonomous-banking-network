package com.sidra.nexus;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

public class NaturalLanguageProcessor {
    private StanfordCoreNLP pipeline;

    public NaturalLanguageProcessor() {
        pipeline = new StanfordCoreNLP();
    }

    public String sentimentAnalysis(String text) {
        Annotation annotation = pipeline.process(text);
        for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
            Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
            int sentimentType = RNNCoreAnnotations.getPredictedClass(tree);
            return sentimentType == 0 ? "Very Negative" : sentimentType == 1 ? "Negative" : sentimentType == 2 ? "Neutral" : sentimentType == 3 ? "Positive" : "Very Positive";
        }
        return null;
    }
             }
