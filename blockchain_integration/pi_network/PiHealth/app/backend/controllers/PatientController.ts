import { Request, Response } from 'express';
import { PatientService } from '../services/PatientService';

class PatientController {
  private patientService: PatientService;

  constructor() {
    this.patientService = new PatientService();
  }

  async createPatient(req: Request, res: Response) {
    try {
      const patient = await this.patientService.createPatient(req.body);
      res.status(201).json(patient);
    } catch (error) {
      res.status(400).json({ message: 'Invalid request' });
    }
  }

  async getPatient(req: Request, res: Response) {
    try {
      const patient = await this.patientService.getPatient(req.params.id);
      res.status(200).json(patient);
    } catch (error) {
      res.status(404).json({ message: 'Patient not found' });
    }
  }

  async updatePatient(req: Request, res: Response) {
    try {
      const patient = await this.patientService.updatePatient(req.params.id, req.body);
      res.status(200).json(patient);
    } catch (error) {
      res.status(400).json({ message: 'Invalid request' });
    }
  }

  async deletePatient(req: Request, res: Response) {
    try {
      await this.patientService.deletePatient(req.params.id);
      res.status(204).json({ message: 'Patient deleted' });
    } catch (error) {
      res.status(404).json({ message: 'Patient not found' });
    }
  }
}

export default PatientController;
