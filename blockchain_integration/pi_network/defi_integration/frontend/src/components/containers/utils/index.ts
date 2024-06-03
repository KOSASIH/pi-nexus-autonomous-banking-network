// index.ts
import React from 'react';
import ReactDOM from 'react-dom';
import { OmniDeFiGateway } from './utils/omni_defi_gateway';

const App = () => {
  const [portfolio, setPortfolio] = React.useState([]);
  const [optimizedPortfolio, setOptimizedPortfolio] = React.useState({});

  const omniDeFiGateway = new OmniDeFiGateway();

  const handleOptimizePortfolio = async () => {
    try {
      const optimizedPortfolio = await omniDeFiGateway.optimizePortfolio(portfolio);
      setOptimizedPortfolio(optimizedPortfolio);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <h1>Omni DeFi Gateway</h1>
      <p>Enter your portfolio:</p>
      <input
        type="text"
        value={portfolio.join(', ')}
        onChange={(e) => setPortfolio(e.target.value.split(', '))}
      />
      <button onClick={handleOptimizePortfolio}>Optimize Portfolio</button>
      <p>Optimized Portfolio:</p>
      <pre>{JSON.stringify(optimizedPortfolio, null, 2)}</pre>
    </div>
  );
};

ReactDOM.render(<App />, document.getElementById('root'));
