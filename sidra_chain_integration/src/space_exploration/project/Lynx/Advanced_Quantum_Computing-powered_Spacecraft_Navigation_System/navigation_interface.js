import { NavigationInterface } from 'navigation-interface-library';

class NavigationInterface {
  constructor() {
    this.navigationInterface = new NavigationInterface();
  }

  getNavigationData() {
    // Return navigation data (position, velocity, trajectory)
    return this.navigationInterface.getNavigationData();
  }

  setNavigationData(navigationData) {
    // Set navigation data (position, velocity, trajectory)
    this.navigationInterface.setNavigationData(navigationData);
  }
}

export default NavigationInterface;
