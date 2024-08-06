import { HealthRecordController } from './HealthRecordController';
import { mockHealthRecords } from '../mocks/healthRecords';

describe('HealthRecordController', () => {
  let controller;
  let req;
  let res;

  beforeEach(() => {
    controller = new HealthRecordController();
    req = { params: {}, query: {} };
    res = { json: jest.fn(), status: jest.fn(() => res) };
  });

  it('should get all health records', async () => {
    const healthRecords = await controller.getAllHealthRecords(req, res);
    expect(res.json).toHaveBeenCalledTimes(1);
    expect(res.json).toHaveBeenCalledWith(mockHealthRecords);
  });

  it('should get a health record by ID', async () => {
    req.params.id = '1';
    const healthRecord = await controller.getHealthRecordById(req, res);
    expect(res.json).toHaveBeenCalledTimes(1);
    expect(res.json).toHaveBeenCalledWith(mockHealthRecords[0]);
  });

  it('should create a new health record', async () => {
    req.body = { patientName: 'John Doe', diagnosis: 'COVID-19' };
    await controller.createHealthRecord(req, res);
    expect(res.status).toHaveBeenCalledTimes(1);
    expect(res.status).toHaveBeenCalledWith(201);
  });

  it('should update a health record', async () => {
    req.params.id = '1';
    req.body = { patientName: 'Jane Doe', diagnosis: 'Influenza' };
    await controller.updateHealthRecord(req, res);
    expect(res.status).toHaveBeenCalledTimes(1);
    expect(res.status).toHaveBeenCalledWith(200);
  });

  it('should delete a health record', async () => {
    req.params.id = '1';
    await controller.deleteHealthRecord(req, res);
    expect(res.status).toHaveBeenCalledTimes(1);
    expect(res.status).toHaveBeenCalledWith(204);
  });
});
