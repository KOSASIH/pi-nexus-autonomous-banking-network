// src/agents/index.js
import TransactionAgent from './TransactionAgent';
import IdentityAgent from './IdentityAgent';
import NotificationAgent from './NotificationAgent';
import ComplianceAgent from './ComplianceAgent';

const transactionAgent = new TransactionAgent();
const identityAgent = new IdentityAgent();
const notificationAgent = new NotificationAgent();
const complianceAgent = new ComplianceAgent();

export {
    transactionAgent,
    identityAgent,
    notificationAgent,
    complianceAgent,
};
