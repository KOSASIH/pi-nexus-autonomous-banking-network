import { Policy } from '../models/policy.model';

const policyService = {
  async createPolicy(policy) {
    try {
      const result = await Policy.create(policy);
      return result;
    } catch (error) {
      throw error;
    }
  },

  async getPolicies() {
    try {
      const policies = await Policy.find().exec();
      return policies;
    } catch (error) {
      throw error;
    }
  },

  async getPolicy(id) {
    try {
      const policy = await Policy.findById(id).exec();
      return policy;
    } catch (error) {
      throw error;
    }
  },

  async updatePolicy(id, policy) {
    try {
      const result = await Policy.findByIdAndUpdate(id, policy, { new: true });
      return result;
    } catch (error) {
      throw error;
    }
  },

  async deletePolicy(id) {
    try {
      await Policy.findByIdAndRemove(id).exec();
    } catch (error) {
      throw error;
    }
  },
};

export { policyService };
