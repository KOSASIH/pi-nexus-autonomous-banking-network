pragma solidity ^0.8.0;

contract AuditResultViewer {
    // Define the SmartContractAuditor contract
    SmartContractAuditor public auditor;

    // Constructor function to initialize the auditor
    constructor(address _auditorAddress) public {
        auditor = SmartContractAuditor(_auditorAddress);
    }

    // Function to view the audit result of a contract
    function viewAuditResult(address _contractAddress) public view returns (AuditResult memory) {
        // Return the audit result
        return auditor.auditResults[_contractAddress];
    }
}
