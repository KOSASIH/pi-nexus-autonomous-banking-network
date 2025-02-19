import TransactionAgent from './TransactionAgent';
import IdentityAgent from './IdentityAgent';
import NotificationAgent from './NotificationAgent';
import ComplianceAgent from './ComplianceAgent';
import RiskAssessmentAgent from './RiskAssessmentAgent';
import AnalyticsAgent from './AnalyticsAgent';
import CustomerSupportAgent from './CustomerSupportAgent';
import { exec } from 'child_process';

// Initialize JavaScript agents
const transactionAgent = new TransactionAgent();
const identityAgent = new IdentityAgent();
const notificationAgent = new NotificationAgent();
const complianceAgent = new ComplianceAgent();
const riskAssessmentAgent = new RiskAssessmentAgent();
const analyticsAgent = new AnalyticsAgent();
const customerSupportAgent = new CustomerSupportAgent();

// Function to run Python scripts
const runPythonScript = (scriptName, args = []) => {
    return new Promise((resolve, reject) => {
        const command = `python3 ${scriptName} ${args.join(' ')}`;
        exec(command, (error, stdout, stderr) => {
            if (error) {
                reject(`Error executing ${scriptName}: ${stderr}`);
            } else {
                resolve(stdout);
            }
        });
    });
};

// Example usage of the Python scripts
const runAgents = async () => {
    try {
        const piNexusOutput = await runPythonScript('pi_nexus_agent.py');
        console.log('Output from pi_nexus_agent.py:', piNexusOutput);

        const bankIntegrationOutput = await runPythonScript('bank_integration.py');
        console.log('Output from bank_integration.py:', bankIntegrationOutput);

        const networkAcceleratorOutput = await runPythonScript('network_accelerator.py');
        console.log('Output from network_accelerator.py:', networkAcceleratorOutput);
    } catch (error) {
        console.error(error);
    }
};

// Call the function to run the Python scripts
runAgents();

export {
    transactionAgent,
    identityAgent,
    notificationAgent,
    complianceAgent,
    riskAssessmentAgent,
    analyticsAgent,
    customerSupportAgent,
};
