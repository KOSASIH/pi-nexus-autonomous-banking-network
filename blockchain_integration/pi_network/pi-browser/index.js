import { createServer } from 'http';
import { Server } from 'ocket.io';
import { Blockchain } from './blockchain';
import { Wallet } from './wallet';
import { TransactionPool } from './transactionPool';
import { Miner } from './miner';
import { PI_Network } from './piNetwork';
import { Browser } from './browser';

const blockchain = new Blockchain();
const wallet = new Wallet();
const transactionPool = new TransactionPool();
const miner = new Miner(blockchain, transactionPool);
const piNetwork = new PI_Network(blockchain, wallet, transactionPool, miner);
const browser = new Browser(piNetwork);

const httpServer = createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.end(`
    <html>
      <head>
        <title>PI-Nexus Autonomous Banking Network</title>
      </head>
      <body>
        <h1>PI-Nexus Autonomous Banking Network</h1>
        <p>Welcome to the PI-Nexus Autonomous Banking Network!</p>
        <button id="connect-button">Connect to Network</button>
        <script src="index.js"></script>
      </body>
    </html>
  `);
});

const io = new Server(httpServer, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
  },
});

io.on('connection', (socket) => {
  console.log('New connection established');

  socket.on('disconnect', () => {
    console.log('Connection lost');
  });

  socket.on('connect-to-network', () => {
    browser.connectToNetwork(socket);
  });

  socket.on('get-blockchain', () => {
    browser.getBlockchain(socket);
  });

  socket.on('get-wallet-balance', () => {
    browser.getWalletBalance(socket);
  });

  socket.on('send-transaction', (transaction) => {
    browser.sendTransaction(socket, transaction);
  });

  socket.on('mine-block', () => {
    browser.mineBlock(socket);
  });
});

httpServer.listen(3000, () => {
  console.log('Server listening on port 3000');
});
