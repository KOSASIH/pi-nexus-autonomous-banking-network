import React, { useState, useEffect, useContext } from "react";
import { BrowserRouter, Route, Switch, Redirect } from "react-router-dom";
import { AppContext } from "../context/AppContext";
import { AuthContext } from "../context/AuthContext";
import { PropertyService } from "../services/PropertyService";
import { Spinner } from "../components/Spinner";
import { ErrorComponent } from "../components/ErrorComponent";
import { Navbar } from "../components/Navbar";
import { PropertyListing } from "../components/PropertyListing";
import { PropertyDetails } from "../components/PropertyDetails";
import { Login } from "../components/Login";
import { Register } from "../components/Register";
import { Profile } from "../components/Profile";
import { Logout } from "../components/Logout";

const App = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const { user, setUser } = useContext(AuthContext);
  const { properties, setProperties } = useContext(AppContext);

  useEffect(() => {
    const fetchProperties = async () => {
      try {
        setLoading(true);
        const response = await PropertyService.getProperties();
        setProperties(response.data);
        setLoading(false);
      } catch (error) {
        setError(error);
        setLoading(false);
      }
    };
    fetchProperties();
  }, []);

  if (loading) {
    return <Spinner />;
  }

  if (error) {
    return <ErrorComponent error={error} />;
  }

  return (
    <BrowserRouter>
      <Navbar />
      <Switch>
        <Route exact path="/" component={PropertyListing} />
        <Route path="/properties/:id" component={PropertyDetails} />
        <Route path="/login" component={Login} />
        <Route path="/register" component={Register} />
        <Route path="/profile" component={Profile} />
        <Route path="/logout" component={Logout} />
        <Redirect to="/" />
      </Switch>
    </BrowserRouter>
  );
};

export default App;
