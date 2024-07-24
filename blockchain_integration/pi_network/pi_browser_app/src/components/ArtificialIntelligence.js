import React, { useState, useEffect } from 'eact';
import { useSelector, useDispatch } from 'eact-redux';
import { trainModel, predict } from '../actions';

const ArtificialIntelligence = () => {
  const dispatch = useDispatch();
  const [inputData, setInputData] = useState('');
  const [modelTrained, setModelTrained] = useState(false);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    dispatch(trainModel());
  }, []);

  const handleInputChange = (event) => {
    setInputData(event.target.value);
  };

  const handlePredict = () => {
    dispatch(predict(inputData));
  };

  return (
    <div>
      <h1>Artificial Intelligence</h1>
      <input type="text" value={inputData} onChange={handleInputChange} />
      <button onClick={handlePredict}>Predict</button>
      {prediction && <p>Prediction: {prediction}</p>}
    </div>
  );
};

export default ArtificialIntelligence;
