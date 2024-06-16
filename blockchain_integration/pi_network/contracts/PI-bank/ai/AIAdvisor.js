const tf = require('@tensorflow/tfjs');
const { PIBank } = require('./PIBank');

class AIAdvisor {
    constructor() {
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ units: 10, inputShape: [10] }));
        this.model.add(tf.layers.dense({ units: 10 }));
        this.model.compile({ optimizer: tf.optimizers.adam(), loss: 'eanSquaredError' });
    }

    async train(data) {
        // Train the model using the provided data
        this.model.fit(data, { epochs: 10 });
    }

    async advise(user) {
        // Use the trained model to provide personalized financial advice
        const userData = await PIBank.getUserData(user);
        const advice = this.model.predict(userData);
        return advice;
    }
}

module.exports = AIAdvisor;
