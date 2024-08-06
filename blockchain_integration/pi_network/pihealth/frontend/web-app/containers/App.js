import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, CardBody, CardTitle, CardSubtitle, Button } from 'reactstrap';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { ApolloClient, InMemoryCache, ApolloProvider } from '@apollo/client';
import { createMuiTheme, ThemeProvider } from '@material-ui/core/styles';
import { SnackbarProvider } from 'notistack';
import { ReactQueryDevtools } from 'react-query-devtools';
import { Helmet } from 'react-helmet';

import { HealthRecordsContainer } from './HealthRecordsContainer';
import { MedicalBillsContainer } from './MedicalBillsContainer';
import { PatientsContainer } from './PatientsContainer';
import { Header } from './Header';
import { Footer } from './Footer';

const theme = createMuiTheme({
  palette: {
    primary: {
      main: '#333',
    },
    secondary: {
      main: '#666',
    },
  },
});

const client = new ApolloClient({
  uri: 'https://example.com/graphql',
  cache: new InMemoryCache(),
});

const App = () => {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const storedDarkMode = localStorage.getItem('darkMode');
    if (storedDarkMode) {
      setDarkMode(storedDarkMode === 'true');
    }
  }, []);

  const handleDarkModeToggle = () => {
    setDarkMode(!darkMode);
    localStorage.setItem('darkMode', darkMode ? 'false' : 'true');
  };

  return (
    <ApolloProvider client={client}>
      <ThemeProvider theme={theme}>
        <SnackbarProvider maxSnack={3}>
          <BrowserRouter>
            <Helmet>
              <title>Medical Billing App</title>
            </Helmet>
            <Header darkMode={darkMode} onDarkModeToggle={handleDarkModeToggle} />
            <Container fluid>
              <Row>
                <Col md={12}>
                  <Switch>
                    <Route path="/" exact component={HealthRecordsContainer} />
                    <Route path="/medical-bills" component={MedicalBillsContainer} />
                    <Route path="/patients" component={PatientsContainer} />
                  </Switch>
                </Col>
              </Row>
            </Container>
            <Footer />
          </BrowserRouter>
        </SnackbarProvider>
      </ThemeProvider>
    </ApolloProvider>
  );
};

export default App;
