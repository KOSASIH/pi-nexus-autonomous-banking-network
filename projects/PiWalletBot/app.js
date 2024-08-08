import express from 'express';
import bodyParser from 'body-parser';
import { DialogflowApp } from 'actions-on-google';
import { getBalance } from './Dialogflow/intents/getBalance';
import { sendTransaction } from './Dialogflow/intents/sendTransaction';
import { getTransactionHistory } from './Dialogflow/intents/getTransactionHistory';
import apiService from './services/apiService';
import authService from './services/authService';
import walletService from './services/walletService';

const app = express();
app.use(bodyParser.json());

const dialogflowApp = new DialogflowApp();

dialogflowApp.intent('get balance', getBalance);
dialogflowApp.intent('send transaction', sendTransaction);
dialogflowApp.intent('get transaction history', getTransactionHistory);

app.post('/dialogflow', dialogflowApp);

app.use('/api', apiService);
app.use('/auth', authService);
app.use('/wallet', walletService);

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
