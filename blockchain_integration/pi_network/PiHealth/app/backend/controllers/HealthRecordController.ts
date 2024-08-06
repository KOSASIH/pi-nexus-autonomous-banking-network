import { Request, Response } from 'express';
import { HealthRecordService } from '../services/HealthRecordService';

class HealthRecordController {
  private healthRecordService: HealthRecordService;

  constructor() {
    this.healthRecordService = new HealthRecordService();
  }

  async createHealthRecord(req: Request, res: Response) {
    try {
      const healthRecord = await this.healthRecordService.createHealthRecord(req.body);
      res.status(201).json(healthRecord);
    } catch (error) {
      res.status(400).json({ message: 'Invalid request' });
    }
  }

  async getHealthRecord(req: Request, res: Response) {
    try {
      const healthRecord = await this.healthRecordService.getHealthRecord(req.params.id);
      res.status(200).json(healthRecord);
    } catch (error) {
      res.status(404).json({ message: 'Health record not found' });
    }
  }

  async updateHealthRecord(req: Request, res: Response) {
    try {
      const healthRecord = await this.healthRecordService.updateHealthRecord(req.params.id, req.body);
      res.status(200).json(healthRecord);
    } catch (error) {
      res.status(400).json({ message: 'Invalid request' });
    }
  }

  async deleteHealthRecord(req: Request, res: Response) {
    try {
      await this.healthRecordService.deleteHealthRecord(req.params.id);
      res.status(204).json({ message: 'Health record deleted' });
    } catch (error) {
      res.status(404).json({ message: 'Health record not found' });
    }
  }
}

export default HealthRecordController;
