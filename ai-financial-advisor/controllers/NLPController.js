const express = require("express");
const router = express.Router();
const nlp = require("nlp.js");
const compromise = require("compromise");
const sentiment = require("sentiment");
const languageDetector = require("language-detector");
const Dialogflow = require("@google-cloud/dialogflow");
const { v4: uuidv4 } = require("uuid");

const dialogflowClient = new Dialogflow.SessionsClient();
const projectId = "your-project-id";
const sessionId = uuidv4();

const financialAdvisorModel = require("../models/FinancialAdvisor");

router.post("/analyzeText", async (req, res) => {
  const text = req.body.text;
  const language = languageDetector.detect(text);
  const nlpDoc = nlp(text);
  const sentimentAnalysis = sentiment(text);
  const entities = compromise(text).entities();

  // Extract relevant financial information from text
  const financialEntities = entities.filter(
    (entity) => entity.type === "MONEY" || entity.type === "DATE",
  );
  const financialData = financialEntities.map((entity) => ({
    type: entity.type,
    value: entity.text,
  }));

  // Use Dialogflow to determine user intent
  const dialogflowRequest = {
    session: `projects/${projectId}/sessions/${sessionId}`,
    queryInput: {
      text: {
        text,
        languageCode: language,
      },
    },
  };
  const dialogflowResponse =
    await dialogflowClient.detectIntent(dialogflowRequest);
  const intent = dialogflowResponse.queryResult.intent.displayName;

  // Use machine learning model to provide personalized financial advice
  const financialAdvisor = await financialAdvisorModel.findOne({
    userId: req.user.id,
  });
  const advice = financialAdvisor.generateAdvice(
    financialData,
    intent,
    sentimentAnalysis,
  );

  res.json({ advice });
});

router.post("/getFinancialGoal", async (req, res) => {
  const text = req.body.text;
  const nlpDoc = nlp(text);
  const entities = compromise(text).entities();

  // Extract relevant financial goal information from text
  const goalEntities = entities.filter(
    (entity) => entity.type === "FINANCIAL_GOAL",
  );
  const goalData = goalEntities.map((entity) => ({
    type: entity.type,
    value: entity.text,
  }));

  // Use machine learning model to generate financial goal
  const financialAdvisor = await financialAdvisorModel.findOne({
    userId: req.user.id,
  });
  const goal = financialAdvisor.generateGoal(goalData);

  res.json({ goal });
});

module.exports = router;
