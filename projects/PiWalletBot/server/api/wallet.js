import { getPiBalance, sendPiTransaction } from './piNetwork';

const walletApi = async (req, res) => {
  switch (req.body.intent) {
    case 'GET_BALANCE':
      const balance = await getPiBalance(req.body.address);
      res.json({ balance });
      break;
    case 'SEND_TRANSACTION':
      const transaction = await sendPiTransaction(req.body.from, req.body.to, req.body.amount);
      res.json({ transaction });
      break;
    default:
      res.status(400).json({ error: 'Invalid intent' });
  }
};

export default walletApi;
