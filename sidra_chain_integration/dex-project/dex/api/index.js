import express from 'express';
import bodyParser from 'body-parser';
import BlockchainAPI from './BlockchainAPI';

const app = express();
app.use(bodyParser.json());

// API endpoint to place an order
app.post('/orders', async (req, res) => {
  try {
    const { amount, price, stopLoss, takeProfit, leverage, expiration } = req.body;
    const orderId = await BlockchainAPI.placeOrder(amount, price, stopLoss, takeProfit, leverage, expiration);
    res.json({ orderId });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to place order' });
  }
});

// API endpoint to cancel an order
app.post('/orders/:orderId/cancel', async (req, res) => {
  try {
    const orderId = req.params.orderId;
    await BlockchainAPI.cancelOrder(orderId);
    res.json({ message: 'Order cancelled successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to cancel order' });
  }
});

// API endpoint to update an order
app.post('/orders/:orderId/update', async (req, res) => {
  try {
    const orderId = req.params.orderId;
    const { newPrice, newStopLoss, newTakeProfit } = req.body;
    await BlockchainAPI.updateOrder(orderId, newPrice, newStopLoss, newTakeProfit);
    res.json({ message: 'Order updated successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to update order' });
  }
});

// API endpoint to get all orders
app.get('/orders', async (req, res) => {
  try {
    const orders = await BlockchainAPI.getOrders();
    res.json(orders);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to get orders' });
  }
});

app.listen(3000, () => {
  console.log('API listening on port 3000');
});
