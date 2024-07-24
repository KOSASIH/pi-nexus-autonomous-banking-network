import React, { useState, useEffect } from 'eact';
import { connect } from 'eact-redux';
import { getVotes } from '../actions/vote.actions';
import { TensorFlow } from '@tensorflow/tfjs';

const VotePredictor = ({ getVotes, voteCount, voteAverage, voteStandardDeviation }) => {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      const model = await TensorFlow.loadLayersModel('https://example.com/vote-predictor-model.json');
      setModel(model);
    };
    loadModel();
  }, []);

  useEffect(() => {
    if (model && voteCount && voteAverage && voteStandardDeviation) {
      const inputData = [
        voteCount,
        voteAverage,
        voteStandardDeviation
      ];
      const output = model.predict(inputData);
      setPrediction(output.dataSync()[0]);
    }
  }, [model, voteCount, voteAverage, voteStandardDeviation]);

  return (
    <div>
      <h1>Vote Predictor</h1>
      {prediction? (
        <p>Predicted outcome: {prediction > 0.5? 'Yes' : 'No'}</p>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
};

const mapStateToProps = (state) => {
  return {
    voteCount: state.vote.voteCount,
    voteAverage: state.vote.voteAverage,
    voteStandardDeviation: state.vote.voteStandardDeviation
  };
};

export default connect(mapStateToProps, { getVotes })(VotePredictor);
