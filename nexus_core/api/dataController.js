import mongoose from 'mongoose';

const dataController = {
  async getData(req, res) {
    const data = await mongoose.model('Data').find();
    res.json(data);
  },
  async postData(req, res) {
    const data = new mongoose.model('Data')(req.body);
    await data.save();
    res.json(data);
  },
};

export default dataController;
