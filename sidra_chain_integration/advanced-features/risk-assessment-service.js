// risk-assessment-service.js
const RiskAssessmentModel = require('./risk-assessment-model');

class RiskAssessmentService {
  constructor() {
    this.model = new RiskAssessmentModel();
  }

  async assessRisk(loanApplication) {
    const riskScore = await this.model.predict(loanApplication);
    return riskScore;
  }
}

module.exports = RiskAssessmentService;
