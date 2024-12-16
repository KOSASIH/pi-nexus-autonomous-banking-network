// disputeResolution.js

class DisputeResolution {
    constructor() {
        this.disputes = [];
    }

    // Report a new dispute
    reportDispute(userId, transactionId, reason) {
        const dispute = {
            userId,
            transactionId,
            reason,
            status: 'open',
            createdAt: new Date(),
        };
        this.disputes.push(dispute);
        console.log(`Dispute reported by user ${userId}:`, dispute);
        return dispute;
    }

    // Get all disputes for a user
    getUser Disputes(userId) {
        return this.disputes.filter(dispute => dispute.userId === userId);
    }

    // Update the status of a dispute
    updateDisputeStatus(userId, transactionId, newStatus) {
        const dispute = this.disputes.find(d => d.userId === userId && d.transactionId === transactionId);
        if (dispute) {
            dispute.status = newStatus;
            console.log(`Dispute status updated for user ${userId}:`, dispute);
            return dispute;
        } else {
            throw new Error('Dispute not found.');
        }
    }

    // Resolve a dispute
    resolveDispute(userId, transactionId) {
        const disputeIndex = this.disputes.findIndex(d => d.userId === userId && d.transactionId === transactionId);
        if (disputeIndex !== -1) {
            const resolvedDispute = this.disputes.splice(disputeIndex, 1);
            console.log(`Dispute resolved for user ${userId}:`, resolvedDispute);
            return resolvedDispute;
        } else {
            throw new Error('Dispute not found.');
        }
    }
}

// Example usage
const disputeManager = new DisputeResolution();
const newDispute = disputeManager.reportDispute('user123', 'txn_123456', 'Item not received');
const userDisputes = disputeManager.getUser Disputes('user123');
console.log('User  Disputes:', userDisputes);

disputeManager.updateDisputeStatus('user123', 'txn_123456', 'under review');
disputeManager.resolveDispute('user123', 'txn_123456');
