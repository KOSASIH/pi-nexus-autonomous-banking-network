import * as math from 'mathjs';

class Robotics {
  constructor() {
    this.kinematics = null;
  }

  async loadKinematics() {
    this.kinematics = await import('robot-kinematics');
  }

  async calculateForwardKinematics(jointAngles) {
    const forwardKinematics = this.kinematics.forwardKinematics(jointAngles);
    return forwardKinematics;
  }

  async calculateInverseKinematics(endEffectorPose) {
    const inverseKinematics = this.kinematics.inverseKinematics(endEffectorPose);
    return inverseKinematics;
  }

  async calculateDynamics(jointAngles, jointVelocities, jointAccelerations) {
    const dynamics = this.kinematics.dynamics(jointAngles, jointVelocities, jointAccelerations);
    return dynamics;
  }
}

export default Robotics;
