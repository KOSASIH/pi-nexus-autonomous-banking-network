import { Request, Response } from 'express';
import { MedicalBillingService } from '../services/MedicalBillingService';

class MedicalBillingController {
  private medicalBillingService: MedicalBillingService;

  constructor() {
    this.medicalBillingService = new MedicalBillingService();
  }

  async createMedicalBilling(req: Request, res: Response) {
    try {
      const medicalBilling = await this.medicalBillingService.createMedicalBilling(req.body);
      res.status(201).json(medicalBilling);
    } catch (error) {
      res.status(400).json({ message: 'Invalid request' });
    }
  }

  async getMedicalBilling(req: Request, res: Response) {
    try {
      const medicalBilling = await this.medicalBillingService.getMedicalBilling(req.params.id);
      res.status(200).json(medicalBilling);
    } catch (error) {
      res.status(404).json({ message: 'Medical billing not found' });
    }
  }

  async updateMedicalBilling(req: Request, res: Response) {
    try {
      const medicalBilling = await this.medicalBillingService.updateMedicalBilling(req.params.id, req.body);
      res.status(200).json(medicalBilling);
    } catch (error) {
      res.status(400).json({ message: 'Invalid request' });
    }
  }

  async deleteMedicalBilling(req: Request, res: Response) {
    try {
      await this.medicalBillingService.deleteMedicalBilling(req.params.id);
      res.status(204).json({ message: 'Medical billing deleted' });
    } catch (error) {
      res.status(404).json({ message: 'Medical billing not found' });
    }
  }
}

export default MedicalBillingController;
