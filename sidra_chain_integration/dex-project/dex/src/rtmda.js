import { SidraChain } from '../sidra-chain';
import { WebSocket } from 'ws';
import { GraphQL } from 'graphql-tag';

class RTMDA {
  constructor(sidraChain) {
    this.sidraChain = sidraChain;
    this.webSocket = new WebSocket('wss://sidra-chain.com/ws');
    this.graphQL = new GraphQL();
  }

  async start() {
    // Establish WebSocket connection
    this.webSocket.on('message', (message) => {
      // Process real-time market data and analytics
      const data = JSON.parse(message);
      this.sidraChain.updateMarketData(data);
      // GraphQL-powered data querying
      const query = `
        query {
          marketData {
            symbol
            price
            volume
          }
        }
      `;
      const result = await this.graphQL.query(query);
      console.log(result);
    });
  }
}

export { RTMDA };
