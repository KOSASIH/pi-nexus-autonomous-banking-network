import React, { useState, useEffect, useMemo, useCallback } from "react";
import { useMachine } from "@xstate/react";
import { createMachine } from "xstate";
import { useInterval } from "react-use";
import { useWeb3 } from "@web3-react/core";
import { Web3Provider } from "@web3-react/providers";
import { ethers } from "ethers";
import { useApolloClient } from "@apollo/client";
import { gql } from "graphql-tag";
import { useAuth0 } from "@auth0/auth0-react";
import { useTranslation } from "react-i18next";

// Define a finite state machine for app navigation
const navigationMachine = createMachine({
  id: "navigation",
  initial: "home",
  states: {
    home: {
      on: {
        NAVIGATE_TO_DASHBOARD: "dashboard",
      },
    },
    dashboard: {
      on: {
        NAVIGATE_TO_SETTINGS: "settings",
      },
    },
    settings: {
      on: {
        NAVIGATE_TO_HOME: "home",
      },
    },
  },
});

// Define a React context for app settings
const AppSettingsContext = React.createContext();

// Define a custom hook for fetching data from a GraphQL API
const useFetchData = (query, variables) => {
  const client = useApolloClient();
  const { data, error, loading } = client.query({
    query: gql(query),
    variables,
  });
  return { data, error, loading };
};

// Define a custom hook for interacting with the Ethereum blockchain
const useEthereum = () => {
  const { account, library } = useWeb3();
  const { data, error, loading } = useFetchData(
    `
      query {
        user(id: "${account}") {
          balance
        }
      }
    `,
    {},
  );
  return { balance: data?.user?.balance, error, loading };
};

// Define the App component
const App = () => {
  const [state, send] = useMachine(navigationMachine);
  const { t } = useTranslation();
  const { isAuthenticated, loginWithPopup, logout } = useAuth0();
  const { balance, error, loading } = useEthereum();
  const [intervalId, setIntervalId] = useState(null);

  useEffect(() => {
    if (isAuthenticated) {
      // Start a recurring task to fetch data from the Ethereum blockchain
      const intervalId = useInterval(() => {
        console.log("Fetching data from Ethereum blockchain...");
      }, 10000);
      setIntervalId(intervalId);
    }
  }, [isAuthenticated]);

  const handleNavigate = useCallback(
    (event) => {
      send(event.type);
    },
    [send],
  );

  return (
    <Web3Provider library={ethers.providers.Web3Provider}>
      <AppSettingsContext.Provider value={{ theme: "dark" }}>
        <div className="app">
          <header>
            <nav>
              <ul>
                <li>
                  <a
                    href="#"
                    onClick={handleNavigate}
                    data-type="NAVIGATE_TO_DASHBOARD"
                  >
                    {t("dashboard")}
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    onClick={handleNavigate}
                    data-type="NAVIGATE_TO_SETTINGS"
                  >
                    {t("settings")}
                  </a>
                </li>
              </ul>
            </nav>
          </header>
          <main>
            {state.matches("home") && <Home />}
            {state.matches("dashboard") && <Dashboard balance={balance} />}
            {state.matches("settings") && <Settings />}
          </main>
        </div>
      </AppSettingsContext.Provider>
    </Web3Provider>
  );
};

export default App;
