import React, { useState, useEffect } from 'eact';
import { connect } from 'eact-redux';
import { getVotes } from '../actions/vote.actions';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'echarts';
import { Spinner } from 'eactstrap';

const VotingDashboard = ({
  getVotes,
  voteCount,
  voteAverage,
  voteStandardDeviation,
  isVoting,
}) => {
  const [intervalId, setIntervalId] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getVotes();
    const intervalId = setInterval(() => {
      getVotes();
    }, 5000);
    setIntervalId(intervalId);
    return () => {
      clearInterval(intervalId);
    };
  }, [getVotes]);

  useEffect(() => {
    if (voteCount && voteAverage && voteStandardDeviation) {
      const data = [];
      for (let i = 0; i < 10; i++) {
        data.push({
          time: new Date(Date.now() - i * 1000).toLocaleTimeString(),
          count: voteCount - i,
          average: voteAverage - i * 0.1,
          stdDev: voteStandardDeviation - i * 0.01,
        });
      }
      setChartData(data);
      setLoading(false);
    } else {
      setLoading(true);
    }
  }, [voteCount, voteAverage, voteStandardDeviation]);

  return (
    <div>
      <h1>Voting Dashboard</h1>
      {loading ? (
        <Spinner color="primary" />
      ) : (
        <LineChart width={800} height={400} data={chartData}>
          <Line type="monotone" dataKey="count" stroke="#8884d8" />
          <Line type="monotone" dataKey="average" stroke="#82ca9d" />
          <Line type="monotone" dataKey="stdDev" stroke="#ffc658" />
          <XAxis dataKey="time" />
          <YAxis />
          <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
          <Tooltip />
        </LineChart>
      )}
    </div>
  );
};

const mapStateToProps = (state) => {
  return {
    voteCount: state.vote.voteCount,
    voteAverage: state.vote.voteAverage,
    voteStandardDeviation: state.vote.voteStandardDeviation,
    isVoting: state.vote.isVoting,
  };
};

export default connect(mapStateToProps, { getVotes })(VotingDashboard);
