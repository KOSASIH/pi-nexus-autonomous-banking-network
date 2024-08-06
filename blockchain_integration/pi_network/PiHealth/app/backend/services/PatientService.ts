import { PatientModel } from '../models/Patient';

class PatientService {
  async createPatient(patient: Patient) {
    try {
      const newPatient = new PatientModel(patient);
      await newPatient.save();
      return newPatient;
    } catch (error) {
      throw error;
    }
  }

  async getPatient(id: string) {
    try {
      const patient = await PatientModel.findById(id);
      return patient;
    } catch (error) {
      throw error;
    }
  }

  async updatePatient(id: string, patient: Patient) {
    try {
      const updatedPatient = await PatientModel.findByIdAndUpdate(id, patient, { new: true });
      return updatedPatient;
    } catch (error) {
      throw error;
    }
  }

  async deletePatient(id: string) {
    try {
      await PatientModel.findByIdAndRemove(id);
    } catch (error) {
      throw error;
    }
  }

  async getPatientsByHealthcareProviderId(healthcareProviderId: string) {
    try {
      const patients = await PatientModel.find({ healthcareProviderId });
      return patients;
    } catch (error) {
      throw error;
    }
  }
}

export default PatientService;
