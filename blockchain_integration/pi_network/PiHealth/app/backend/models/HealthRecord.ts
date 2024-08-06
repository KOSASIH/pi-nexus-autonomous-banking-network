import { model, Document } from 'mongoose';

interface HealthRecord {
  id: string;
  patientId: string;
  medicalHistory: string[];
  allergies: string[];
  medications: string[];
}

const healthRecordSchema = new mongoose.Schema({
  id: { type: String, required: true },
  patientId: { type: String, required: true },
  medicalHistory: { type: [String], required: true },
  allergies: { type: [String], required: true },
  medications: { type: [String], required: true },
});

const HealthRecordModel = model<HealthRecord & Document>('HealthRecord', healthRecordSchema);

export default HealthRecordModel;
