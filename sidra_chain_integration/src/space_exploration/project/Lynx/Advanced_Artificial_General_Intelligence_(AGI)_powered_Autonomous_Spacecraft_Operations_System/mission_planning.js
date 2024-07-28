import { MissionPlan } from 'mission-plan-library';

class MissionPlanning {
  constructor() {
    this.missionPlan = new MissionPlan();
  }

  update(decision) {
    // Update mission plan based on decision
    this.missionPlan.update(decision);
  }
}

export default MissionPlanning;
