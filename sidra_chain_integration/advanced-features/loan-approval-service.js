// loan-approval-service.js
const LoanApprovalModel = require('./loan-approval-model');

class LoanApprovalService {
  constructor() {
    this.model = new LoanApprovalModel();
  }

  async approveLoan(loanApplication) {
    const approvalStatus = await this.model.predict(loanApplication);
    return approvalStatus;
  }
}

module.exports = LoanApprovalService;
