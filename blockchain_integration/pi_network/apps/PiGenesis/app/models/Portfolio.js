import mongoose, { Document, Model, Schema } from 'ongoose';

interface Portfolio {
  userId: string;
  assets: Array<{ symbol: string; amount: number }>;
  value: number;
}

const portfolioSchema = new Schema<Portfolio>({
  userId: { type: String, required: true },
  assets: [{ symbol: String, amount: Number }],
  value: { type: Number, required: true },
});

const Portfolio: Model<Portfolio> = mongoose.model('Portfolio', portfolioSchema);

export default Portfolio;
