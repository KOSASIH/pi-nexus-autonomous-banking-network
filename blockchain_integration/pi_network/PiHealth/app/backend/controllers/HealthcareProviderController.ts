import { Request, Response } from 'express';
import { HealthcareProviderService } from '../services/HealthcareProviderService';

class HealthcareProviderController {
  private healthcareProviderService: HealthcareProviderService;

  constructor() {
    this.healthcareProviderService = new HealthcareProviderService();
  }

  async createHealthcareProvider(req: Request, res: Response) {
    try {
      const healthcareProvider = await this.healthcareProviderService.createHealthcareProvider(req.body);
      res.status(201).json(healthcareProvider);
    } catch (error) {
      res.status(400).json({ message: 'Invalid request' });
    }
  }

  async getHealthcareProvider(req: Request, res: Response) {
    try {
      const healthcareProvider = await this.healthcareProviderService.getHealthcareProvider(req.params.id);
      res.status(200).json(healthcareProvider);
    } catch (error) {
      res.status(404).json({ message: 'Healthcare provider not found' });
    }
  }

  async updateHealthcareProvider(req: Request, res: Response) {
    try {
      const healthcareProvider = await this.healthcareProviderService.updateHealthcareProvider(req.params.id, req.body);
      res.status(200).json(healthcareProvider);
    } catch (error) {
      res.status(400).json({ message: 'Invalid request' });
    }
  }

  async deleteHealthcareProvider(req: Request, res: Response) {
    try {
      await this.healthcareProviderService.deleteHealthcareProvider(req.params.id);
      res.status(204).json({ message: 'Healthcare provider deleted' });
    } catch (error) {
      res.status(404).json({ message: 'Healthcare provider not found' });
    }
  }
}

export default HealthcareProviderController;
