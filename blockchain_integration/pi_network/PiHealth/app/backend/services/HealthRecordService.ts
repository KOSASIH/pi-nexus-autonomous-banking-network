import { HealthRecordModel } from '../models/HealthRecord';

class HealthRecordService {
  async createHealthRecord(healthRecord: HealthRecord) {
    try {
      const newHealthRecord = new HealthRecordModel(healthRecord);
      await newHealthRecord.save();
      return newHealthRecord;
    } catch (error) {
      throw error;
    }
  }

  async getHealthRecord(id: string) {
    try {
      const healthRecord = await HealthRecordModel.findById(id);
      return healthRecord;
    } catch (error) {
      throw error;
    }
  }

  async updateHealthRecord(id: string, healthRecord: HealthRecord) {
    try {
      const updatedHealthRecord = await HealthRecordModel.findByIdAndUpdate(id, healthRecord, { new: true });
      return updatedHealthRecord;
    } catch (error) {
      throw error;
    }
  }

  async deleteHealthRecord(id: string) {
    try {
      await HealthRecordModel.findByIdAndRemove(id);
    } catch (error) {
      throw error;
    }
  }

  async getHealthRecordsByPatientId(patientId: string) {
    try {
      const healthRecords = await HealthRecordModel.find({ patientId });
      return healthRecords;
    } catch (error) {
      throw error;
    }
  }

  async getHealthRecordsByHealthcareProviderId(healthcareProviderId: string) {
    try {
      const healthRecords = await HealthRecordModel.find({ healthcareProviderId });
      return healthRecords;
    } catch (error) {
      throw error;
    }
  }
}

export default HealthRecordService;
