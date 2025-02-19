import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import IdentityManager from './IdentityManager';
import Home from './Home'; // Assuming you have a Home component
import NotFound from './NotFound'; // A component for handling 404 errors
import './App.css'; // Import your CSS file for styling

function App() {
  return (
    <Router>
      <div className="app-container">
        <header className="app-header">
          <h1>Welcome to Pi Nexus</h1>
        </header>
        <main>
          <Switch>
            <Route path="/" exact component={Home} />
            <Route path="/identity" component={IdentityManager} />
            <Route component={NotFound} /> {/* Fallback for 404 */}
          </Switch>
        </main>
        <footer className="app-footer">
          <p>&copy; 2025 Pi Nexus. All rights reserved.</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
