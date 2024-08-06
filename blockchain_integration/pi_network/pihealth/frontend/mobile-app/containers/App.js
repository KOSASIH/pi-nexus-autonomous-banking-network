import React, { useState, useEffect } from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { connect } from 'react-redux';
import { authenticate } from '../utils/auth';
import { setAuthenticated } from '../store/authSlice';
import Header from '../components/Header';
import Footer from '../components/Footer';
import HealthRecordComponent from '../components/HealthRecordComponent';
import MedicalBillingComponent from '../components/MedicalBillingComponent';
import Dashboard from '../pages/Dashboard';
import Login from '../pages/Login';
import Register from '../pages/Register';

const App = ({ authenticated, setAuthenticated }) => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const user = await authenticate();
        setAuthenticated(true);
        setLoading(false);
      } catch (error) {
        setAuthenticated(false);
        setLoading(false);
      }
    };

    checkAuth();
  }, [setAuthenticated]);

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <BrowserRouter>
      <Header />
      <Switch>
        <Route exact path="/" component={Dashboard} />
        <Route path="/health-records" component={HealthRecordComponent} />
        <Route path="/medical-bills" component={MedicalBillingComponent} />
        <Route path="/login" component={Login} />
        <Route path="/register" component={Register} />
      </Switch>
      <Footer />
    </BrowserRouter>
  );
};

const mapStateToProps = (state) => ({
  authenticated: state.auth.authenticated,
});

const mapDispatchToProps = (dispatch) => ({
  setAuthenticated: (authenticated) => dispatch(setAuthenticated(authenticated)),
});

export default connect(mapStateToProps, mapDispatchToProps)(App);
