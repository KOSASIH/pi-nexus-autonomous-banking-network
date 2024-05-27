import { model, Schema } from "mongoose";
import { v4 as uuidv4 } from "uuid";

const accountSchema = new Schema({
  _id: { type: String, default: uuidv4 },
  userId: { type: String, ref: "User" },
  balance: { type: Number, default: 0 },
  currency: { type: String, enum: ["USD", "EUR", "GBP"] },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
});

accountSchema.virtual("transactions", {
  ref: "Transaction",
  localField: "_id",
  foreignField: "accountId",
});

export default model("Account", accountSchema);
