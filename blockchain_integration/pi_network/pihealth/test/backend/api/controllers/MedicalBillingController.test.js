import { MedicalBillingController } from './MedicalBillingController';
import { mockMedicalBills } from '../mocks/medicalBills';

describe('MedicalBillingController', () => {
  let controller;
  let req;
  let res;

  beforeEach(() => {
    controller = new MedicalBillingController();
    req = { params: {}, query: {} };
    res = { json: jest.fn(), status: jest.fn(() => res) };
  });

  it('should get all medical bills', async () => {
    const medicalBills = await controller.getAllMedicalBills(req, res);
    expect(res.json).toHaveBeenCalledTimes(1);
    expect(res.json).toHaveBeenCalledWith(mockMedicalBills);
  });

  it('should get a medical bill by ID', async () => {
    req.params.id = '1';
    const medicalBill = await controller.getMedicalBillById(req, res);
    expect(res.json).toHaveBeenCalledTimes(1);
    expect(res.json).toHaveBeenCalledWith(mockMedicalBills[0]);
  });

  it('should create a new medical bill', async () => {
    req.body = { patientName: 'John Doe', billAmount: 100.00 };
    await controller.createMedicalBill(req, res);
    expect(res.status).toHaveBeenCalledTimes(1);
    expect(res.status).toHaveBeenCalledWith(201);
  });

  it('should update a medical bill', async () => {
    req.params.id = '1';
    req.body = { patientName: 'Jane Doe', billAmount: 200.00 };
    await controller.updateMedicalBill(req, res);
    expect(res.status).toHaveBeenCalledTimes(1);
    expect(res.status).toHaveBeenCalledWith(200);
  });

  it('should delete a medical bill', async () => {
    req.params.id = '1';
    await controller.deleteMedicalBill(req, res);
    expect(res.status).toHaveBeenCalledTimes(1);
    expect(res.status).toHaveBeenCalledWith(204);
  });
});
