import { Policy } from '../models/policy.model';
import { policyService } from '../services/policy.service';

const createPolicy = async (req, res) => {
  try {
    const policy = new Policy(req.body);
    const result = await policyService.createPolicy(policy);
    res.status(201).json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error creating policy' });
  }
};

const getPolicies = async (req, res) => {
  try {
    const policies = await policyService.getPolicies();
    res.json(policies);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error fetching policies' });
  }
};

const getPolicy = async (req, res) => {
  try {
    const id = req.params.id;
    const policy = await policyService.getPolicy(id);
    res.json(policy);
  } catch (error) {
    console.error(error);
    res.status(404).json({ message: 'Policy not found' });
  }
};

const updatePolicy = async (req, res) => {
  try {
    const id = req.params.id;
    const policy = await policyService.updatePolicy(id, req.body);
    res.json(policy);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error updating policy' });
  }
};

const deletePolicy = async (req, res) => {
  try {
    const id = req.params.id;
    await policyService.deletePolicy(id);
    res.status(204).json({ message: 'Policy deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error deleting policy' });
  }
};

export { createPolicy, getPolicies, getPolicy, updatePolicy, deletePolicy };
