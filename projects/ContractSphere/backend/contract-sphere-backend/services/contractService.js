import Contract from '../models/Contract';

export const createContract = async (req, res) => {
  try {
    const contract = new Contract(req.body);
    await contract.save();
    res.status(201).json(contract);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getContracts = async (req, res) => {
  try {
    const contracts = await Contract.find().populate('userId');
    res.json(contracts);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getContract = async (req, res) => {
  try {
    const contract = await Contract.findById(req.params.id).populate('userId');
    if (!contract) {
      res.status(404).json({ message: 'Contract not found' });
    } else {
      res.json(contract);
    }
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const updateContract = async (req, res) => {
  try {
    const contract = await Contract.findByIdAndUpdate(req.params.id, req.body, { new: true });
    res.json(contract);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const deleteContract = async (req, res) => {
  try {
    await Contract.findByIdAndRemove(req.params.id);
    res.status(204).json({ message: 'Contract deleted' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
