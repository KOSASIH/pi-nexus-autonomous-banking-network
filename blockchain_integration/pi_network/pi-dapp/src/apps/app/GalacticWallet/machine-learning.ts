import * as tf from '@tensorflow/tfjs';

const useMachineLearning = () => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);

  useEffect(() => {
    const createModel = async () => {
      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
      model.compile({ optimizer: tf.optimizers.adam(), loss: 'eanSquaredError' });
      setModel(model);
    };
    createModel();
  }, []);

  const train = (transaction: any) => {
    if (!model) return;
    const input = tf.tensor2d([transaction.amount], [1, 1]);
    const output = tf.tensor2d([transaction.recipient], [1, 1]);
    model.fit(input, output, { epochs: 1 });
  };

  return { train };
};

export default useMachineLearning;
