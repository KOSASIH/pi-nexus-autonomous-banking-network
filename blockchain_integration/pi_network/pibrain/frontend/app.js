// app.js

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { render } from 'react-dom';
import { BrowserRouter, Route, Switch, Link } from 'react-router-dom';
import { ApolloClient, InMemoryCache, ApolloProvider } from '@apollo/client';
import { Web3Provider } from '@ethersproject/providers';
import { Web3ReactProvider } from '@web3-react/core';
import { ThemeProvider } from 'styled-components';
import { RecoilRoot } from 'recoil';
import { ToastContainer } from 'react-toastify';

import 'react-toastify/dist/ReactToastify.css';
import 'normalize.css';
import './styles/global.css';

import AppLayout from './components/AppLayout';
import NodePerformanceChart from './components/NodePerformanceChart';
import NetworkMetricsDashboard from './components/NetworkMetricsDashboard';
import DataSharingMarketplace from './components/DataSharingMarketplace';
import AnomalyDetectionAlerts from './components/AnomalyDetectionAlerts';
import NodeOptimizerTrainer from './components/NodeOptimizerTrainer';

import { nodeOptimizerModel } from './models/nodeOptimizerModel';
import { networkPredictorModel } from './models/networkPredictorModel';
import { dataSharingContractABI } from './contracts/dataSharingContractABI';

const client = new ApolloClient({
  uri: 'https://api.example.com/graphql',
  cache: new InMemoryCache(),
});

const web3Provider = new Web3Provider(window.ethereum);

const App = () => {
  const [nodeData, setNodeData] = useState([]);
  const [networkMetrics, setNetworkMetrics] = useState({});
  const [dataSharingContract, setDataSharingContract] = useState(null);

  useEffect(() => {
    const fetchNodeData = async () => {
      const response = await client.query({
        query: gql`
          query {
            nodes {
              id
              performanceMetrics {
                cpuUsage
                memoryUsage
              }
            }
          }
        `,
      });
      setNodeData(response.data.nodes);
    };
    fetchNodeData();
  }, []);

  useEffect(() => {
    const fetchNetworkMetrics = async () => {
      const response = await client.query({
        query: gql`
          query {
            networkMetrics {
              throughput
              latency
            }
          }
        `,
      });
      setNetworkMetrics(response.data.networkMetrics);
    };
    fetchNetworkMetrics();
  }, []);

  useEffect(() => {
    const initWeb3 = async () => {
      const contract = new web3Provider.eth.Contract(dataSharingContractABI, '0x...ContractAddress...');
      setDataSharingContract(contract);
    };
    initWeb3();
  }, []);

  const handleNodeOptimizerTraining = useCallback(async () => {
    const optimizer = new NodeOptimizerTrainer(nodeOptimizerModel);
    const trainedModel = await optimizer.train();
    console.log(trainedModel);
  }, [nodeOptimizerModel]);

  const handleNetworkPredictorTraining = useCallback(async () => {
    const predictor = new NetworkPredictorTrainer(networkPredictorModel);
    const trainedModel = await predictor.train();
    console.log(trainedModel);
  }, [networkPredictorModel]);

  return (
    <ThemeProvider theme={{ primaryColor: '#333', secondaryColor: '#666' }}>
      <RecoilRoot>
        <Web3ReactProvider>
          <ApolloProvider client={client}>
            <BrowserRouter>
              <AppLayout>
                <Switch>
                  <Route path="/" exact component={NodePerformanceChart} />
                  <Route path="/network-metrics" component={NetworkMetricsDashboard} />
                  <Route path="/data-sharing-marketplace" component={DataSharingMarketplace} />
                  <Route path="/anomaly-detection-alerts" component={AnomalyDetectionAlerts} />
                </Switch>
                <button onClick={handleNodeOptimizerTraining}>Train Node Optimizer</button>
                <button onClick={handleNetworkPredictorTraining}>Train Network Predictor</button>
              </AppLayout>
            </BrowserRouter>
          </ApolloProvider>
        </Web3ReactProvider>
      </RecoilRoot>
      <ToastContainer />
    </ThemeProvider>
  );
};

render(<App />, document.getElementById('root'));
