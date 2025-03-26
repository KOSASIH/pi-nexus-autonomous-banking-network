import React from "react";
import ReactDOM from "react-dom";
import { BrowserRouter } from "react-router-dom";
import { Provider } from "react-redux";
import { createStore, combineReducers } from "redux";
import { composeWithDevTools } from "redux-devtools-extension";
import { ApolloClient, InMemoryCache } from "@apollo/client";
import { Web3Provider } from "@web3-react/providers";
import { ethers } from "ethers";
import App from "./App";
import reducers from "./reducers";

// Create a Redux store with dev tools enabled
const store = createStore(combineReducers(reducers), composeWithDevTools());

// Create an Apollo client with an in-memory cache
const client = new ApolloClient({
  uri: "http://localhost:4000/graphql",
  cache: new InMemoryCache(),
});

ReactDOM.render(
  <Provider store={store}>
    <BrowserRouter>
      <Web3Provider client={client}>
        <App />
      </Web3Provider>
    </BrowserRouter>
  </Provider>,
  document.getElementById("root"),
);
