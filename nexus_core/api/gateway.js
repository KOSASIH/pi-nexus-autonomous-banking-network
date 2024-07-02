import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';

const app = express();
app.use(bodyParser.json());
app.use(cors());

app.get('/api/data', async (req, res) => {
  const data = await mongoose.model('Data').find();
  res.json(data);
});

app.post('/api/data', async (req, res) => {
  const data = new mongoose.model('Data')(req.body);
  await data.save();
  res.json(data);
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
