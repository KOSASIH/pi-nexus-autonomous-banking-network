// paymentPlan.js

class PaymentPlan {
    constructor() {
        this.plans = [];
    }

    // Create a new payment plan
    createPlan(userId, totalAmount, durationMonths) {
        const plan = {
            userId,
            totalAmount,
            durationMonths,
            monthlyPayment: totalAmount / durationMonths,
            status: 'active',
            createdAt: new Date(),
        };
        this.plans.push(plan);
        console.log(`Payment plan created for user ${userId}:`, plan);
        return plan;
    }

    // Get all payment plans for a user
    getUser Plans(userId) {
        return this.plans.filter(plan => plan.userId === userId);
    }

    // Update a payment plan status
    updatePlanStatus(userId, planId, newStatus) {
        const plan = this.plans.find(p => p.userId === userId && p.id === planId);
        if (plan) {
            plan.status = newStatus;
            console.log(`Payment plan status updated for user ${userId}:`, plan);
            return plan;
        } else {
            throw new Error('Payment plan not found.');
        }
    }

    // Cancel a payment plan
    cancelPlan(userId, planId) {
        const planIndex = this.plans.findIndex(p => p.userId === userId && p.id === planId);
        if (planIndex !== -1) {
            const canceledPlan = this.plans.splice(planIndex, 1);
            console.log(`Payment plan canceled for user ${userId}:`, canceledPlan);
            return canceledPlan;
        } else {
            throw new Error('Payment plan not found.');
        }
    }
}

// Example usage
const paymentPlanManager = new PaymentPlan();
const newPlan = paymentPlanManager.createPlan('user123', 1200, 12); // $1200 over 12 months
const userPlans = paymentPlanManager.getUser Plans('user123');
console.log('User  Plans:', userPlans);

paymentPlanManager.updatePlanStatus('user123', newPlan.id, 'completed');
paymentPlanManager.cancelPlan('user123', newPlan.id);

export default PaymentPlan;
