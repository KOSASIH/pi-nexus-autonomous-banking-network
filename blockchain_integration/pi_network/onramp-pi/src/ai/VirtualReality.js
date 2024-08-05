// VirtualReality.js

import { HeadMountedDisplay } from 'head-mounted-display';
import { Controllers } from 'controllers';

class VirtualReality {
  constructor() {
    this.headMountedDisplay = new HeadMountedDisplay();
    this.controllers = new Controllers();
  }

  experience() {
    // Provide a virtual reality experience
    this.headMountedDisplay.render();
    this.controllers.track();
  }
}

export default VirtualReality;
