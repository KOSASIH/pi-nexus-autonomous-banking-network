import React from 'react';

const Trade = ({ trade }) => {
  return (
    <div>
      <h3>Trade Details</h3>
      <p>Trader: {trade.trader}</p>
      <p>Token In: {trade.tokenIn}</p>
      <p>Token Out: {trade.tokenOut}</p>
      <p>Amount In: {trade.amountIn} {trade.tokenIn}</p>
      <p>Amount Out: {trade.amountOut} {trade.tokenOut}</p>
      <p>Timestamp: {trade.timestamp}</p>
    </div>
  );
};

export default Trade;
