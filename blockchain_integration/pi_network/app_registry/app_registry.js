// app_registry.js

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');

const APP_REGISTRY_FILE = 'apps.json';
const APP_REGISTRY_DIR = path.join(__dirname, './');

class AppRegistry {
  constructor() {
    this.apps = this.loadApps();
  }

  loadApps() {
    try {
      const appsJson = fs.readFileSync(APP_REGISTRY_FILE, 'utf8');
      return JSON.parse(appsJson);
    } catch (error) {
      console.error(`Error loading app registry: ${error}`);
      return { apps: [] };
    }
  }

  saveApps() {
    try {
      const appsJson = JSON.stringify(this.apps, null, 2);
      fs.writeFileSync(APP_REGISTRY_FILE, appsJson);
    } catch (error) {
      console.error(`Error saving app registry: ${error}`);
    }
  }

  listApps() {
    return this.apps.apps;
  }

  getAppById(id) {
    return this.apps.apps.find(app => app.id === id);
  }

  getAppByName(name) {
    return this.apps.apps.find(app => app.name === name);
  }

  addApp(appData) {
    const appId = uuidv4();
    const app = {
      id: appId,
      name: appData.name,
      description: appData.description,
      icon: appData.icon,
      url: appData.url,
      contractAddress: appData.contractAddress,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    this.apps.apps.push(app);
    this.saveApps();
    return app;
  }

  updateApp(id, appData) {
    const app = this.getAppById(id);
    if (!app) {
      throw new Error(`App not found: ${id}`);
    }
    app.name = appData.name;
    app.description = appData.description;
    app.icon = appData.icon;
    app.url = appData.url;
    app.contractAddress = appData.contractAddress;
    app.updatedAt = new Date().toISOString();
    this.saveApps();
    return app;
  }

  removeApp(id) {
    const appIndex = this.apps.apps.findIndex(app => app.id === id);
    if (appIndex === -1) {
      throw new Error(`App not found: ${id}`);
    }
    this.apps.apps.splice(appIndex, 1);
    this.saveApps();
  }

  generateAppToken(appId) {
    const app = this.getAppById(appId);
    if (!app) {
      throw new Error(`App not found: ${appId}`);
    }
    const token = crypto.randomBytes(32).toString('hex');
    app.token = token;
    this.saveApps();
    return token;
  }

  verifyAppToken(appId, token) {
    const app = this.getAppById(appId);
    if (!app) {
      throw new Error(`App not found: ${appId}`);
    }
    return app.token === token;
  }
}

module.exports = new AppRegistry();
