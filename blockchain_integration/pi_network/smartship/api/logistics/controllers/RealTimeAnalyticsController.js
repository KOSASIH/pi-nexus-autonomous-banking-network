const kafka = require('kafka-node');

class RealTimeAnalyticsController {
  async analyzeStreamingData(req, res) {
    const { topic, data } = req.body;
    const consumer = new kafka.Consumer(client, [{ topic }]);
    consumer.on('message', (message) => {
      const analysis = analyzeData(message.value);
      res.json({ analysis });
    });
  }
}
