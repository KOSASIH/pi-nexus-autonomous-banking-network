import { useState, useEffect } from 'react';
import { useNeuralNetwork } from '@brainjs/neural-react';

const NeuralNetworkForm = ({ onSubmit }) => {
  const [inputValues, setInputValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const { predict } = useNeuralNetwork('user-input-model');

  useEffect(() => {
    const handleInputChange = (event) => {
      const { name, value } = event.target;
      setInputValues((prevValues) => ({ ...prevValues, [name]: value }));
    };

    const handlePredict = async () => {
      const prediction = await predict(inputValues);
      setPrediction(prediction);
    };

    handlePredict();
  }, [inputValues]);

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(inputValues, prediction);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Name:
        <input type="text" name="name" onChange={handleInputChange} />
      </label>
      <label>
        Email:
        <input type="email" name="email" onChange={handleInputChange} />
      </label>
      <button type="submit">Submit</button>
      {prediction && <p>Prediction: {prediction}</p>}
    </form>
  );
};

export default NeuralNetworkForm;
