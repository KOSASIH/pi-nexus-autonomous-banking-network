/**
 * Copyright 2020 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

'use strict';

const functions = require('firebase-functions');
const { dialogflow } = require('actions-on-google');

const MIN_SEQUENCE_LENGTH = 10;

exports.dialogflowFirebaseFulfillment = functions.https.onRequest((request, response) => {
  let dfRequest = request.body;
  let action = dfRequest.queryResult.action;
  switch (action) {
    case 'handle-sequence':
      handleSequence(dfRequest, response);
      break;
    case 'validate-sequence':
      validateSequence(dfRequest, response);
      break;
    default:
      response.json({
        fulfillmentText: `Webhook for action "${action}" not implemented.`,
      });
  }
});

//// Helper functions

/**
 * Send an SSML response.
 * @param {Object} request Dialogflow WebhookRequest JSON with camelCase keys.
 * @param {Object} response Express JS response object
 * @param {string} ssml SSML string.
 * @example sendSSML(request, response, 'hello')
 * Will call response.json() with SSML payload 'hello'
 */
function sendSSML(request, response, ssml) {
  ssml = `${ssml}`;

  if (request.originalDetectIntentRequest.source === 'GOOGLE_TELEPHONY') {
    // Dialogflow Phone Gateway Response
    // see https://cloud.google.com/dialogflow/es/docs/reference/rpc/google.cloud.dialogflow.v2beta1#google.cloud.dialogflow.v2beta1.Intent.Message.TelephonySynthesizeSpeech
    response.json({
      fulfillmentMessages: [{
        platform: 'TELEPHONY',
        telephonySynthesizeSpeech: { ssml: ssml },
      }],
    });
  } else {
    // Some CCAI telephony partners accept SSML in a plain text response.
    // Check...
    response.json({ fulfillmentText: ssml });
  }
}

function handleSequence(dfRequest, response) {
  // TODO: add logic to handle the sequence
  sendSSML(dfRequest, response, 'Thank you. Your sequence is ...');
}

function validateSequence(dfRequest, response) {
  let parameters = dfRequest.queryResult.parameters;
  // TODO: add logic to validate the sequence
  let verbatim = `<say-as interpret-as="verbatim">${parameters.sequence}</say-as>`;
  sendSSML(dfRequest, response, `Thank you. Your sequence is ${verbatim}`);
}
