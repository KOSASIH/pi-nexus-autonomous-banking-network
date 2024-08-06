import { MedicalBillingModel } from '../models/MedicalBilling';

class MedicalBillingService {
  async createMedicalBilling(medicalBilling: MedicalBilling) {
    try {
      const newMedicalBilling = new MedicalBillingModel(medicalBilling);
      await newMedicalBilling.save();
      return newMedicalBilling;
    } catch (error) {
      throw error;
    }
  }

  async getMedicalBilling(id: string) {
    try {
      const medicalBilling = await MedicalBillingModel.findById(id);
      return medicalBilling;
    } catch (error) {
      throw error;
    }
  }

  async updateMedicalBilling(id: string, medicalBilling: MedicalBilling) {
    try {
      const updatedMedicalBilling = await MedicalBillingModel.findByIdAndUpdate(id, medicalBilling, { new: true });
      return updatedMedicalBilling;
    } catch (error) {
      throw error;
    }
  }

  async deleteMedicalBilling(id: string) {
    try {
      await MedicalBillingModel.findByIdAndRemove(id);
    } catch (error) {
      throw error;
    }
  }

  async getMedicalBillingsByPatientId(patientId: string) {
    try {
      const medicalBillings = await MedicalBillingModel.find({ patientId });
      return medicalBillings;
    } catch (error) {
      throw error;
    }
  }

  async getMedicalBillingsByHealthcareProviderId(healthcareProviderId: string) {
    try {
      const medicalBillings = await MedicalBillingModel.find({ healthcareProviderId });
      return medicalBillings;
    } catch (error) {
      throw error;
    }
  }
}

export default MedicalBillingService;
