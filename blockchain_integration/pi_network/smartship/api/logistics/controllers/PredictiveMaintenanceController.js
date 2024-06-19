const tf = require('@tensorflow/tfjs');

class PredictiveMaintenanceController {
  async predictMaintenance(req, res) {
    const { vehicleId, sensorData } = req.body;
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, inputShape: [10] }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: tf.optimizers.adam(), loss: 'meanSquaredError' });
    const prediction = model.predict(sensorData);
    res.json({ prediction });
  }
}
