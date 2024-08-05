// App.js
import React, { useState, useEffect, useContext } from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import { UserContext } from '../contexts/UserContext';
import { FoodContext } from '../contexts/FoodContext';
import { InventoryContext } from '../contexts/InventoryContext';
import { AppTheme } from '../styles/AppTheme';
import { Header } from '../components/Header';
import { Footer } from '../components/Footer';
import { FoodTracker } from '../components/FoodTracker';
import { InventoryManager } from '../components/InventoryManager';
import { Login } from '../components/Login';
import { Register } from '../components/Register';
import { PrivateRoute } from '../components/PrivateRoute';

const App = () => {
  const [user, setUser] = useState(null);
  const [foods, setFoods] = useState([]);
  const [inventory, setInventory] = useState({});

  useEffect(() => {
    const fetchUser = async () => {
      try {
        const user = await UserService.getCurrentUser();
        setUser(user);
      } catch (err) {
        console.error(err);
      }
    };
    fetchUser();
  }, []);

  useEffect(() => {
    const fetchFoods = async () => {
      try {
        const foods = await FoodService.getFoods();
        setFoods(foods);
      } catch (err) {
        console.error(err);
      }
    };
    fetchFoods();
  }, []);

  useEffect(() => {
    const fetchInventory = async () => {
      try {
        const inventory = await InventoryService.getInventory();
        setInventory(inventory);
      } catch (err) {
        console.error(err);
      }
    };
    fetchInventory();
  }, []);

  return (
    <ThemeProvider theme={AppTheme}>
      <BrowserRouter>
        <UserContext.Provider value={{ user, setUser }}>
          <FoodContext.Provider value={{ foods, setFoods }}>
            <InventoryContext.Provider value={{ inventory, setInventory }}>
              <Header />
              <Switch>
                <Route path="/login" component={Login} />
                <Route path="/register" component={Register} />
                <PrivateRoute path="/food-tracker" component={FoodTracker} />
                <PrivateRoute path="/inventory-manager" component={InventoryManager} />
              </Switch>
              <Footer />
            </InventoryContext.Provider>
          </FoodContext.Provider>
        </UserContext.Provider>
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default App;
