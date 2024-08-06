import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, CardBody, CardTitle, CardSubtitle, Button } from 'reactstrap';
import { getMedicalBills, createMedicalBill, updateMedicalBill } from '../api/medicalBillsApi';
import { getPatients } from '../api/patientsApi';

const MedicalBillingComponent = () => {
  const [medicalBills, setMedicalBills] = useState([]);
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [selectedMedicalBill, setSelectedMedicalBill] = useState(null);
  const [editMode, setEditMode] = useState(false);
  const [newMedicalBill, setNewMedicalBill] = useState({
    patientId: '',
    date: '',
    amount: '',
    description: '',
  });

  useEffect(() => {
    const fetchMedicalBills = async () => {
      const response = await getMedicalBills();
      setMedicalBills(response.data);
    };

    const fetchPatients = async () => {
      const response = await getPatients();
      setPatients(response.data);
    };

    fetchMedicalBills();
    fetchPatients();
  }, []);

  const handlePatientSelect = (patient) => {
    setSelectedPatient(patient);
    const medicalBill = medicalBills.find((bill) => bill.patientId === patient.id);
    setSelectedMedicalBill(medicalBill);
  };

  const handleMedicalBillSelect = (medicalBill) => {
    setSelectedMedicalBill(medicalBill);
  };

  const handleEditModeToggle = () => {
    setEditMode(!editMode);
  };

  const handleCreateMedicalBill = async () => {
    try {
      const response = await createMedicalBill(newMedicalBill);
      setMedicalBills(response.data);
      setNewMedicalBill({
        patientId: '',
        date: '',
        amount: '',
        description: '',
      });
    } catch (error) {
      console.error(error);
    }
  };

  const handleUpdateMedicalBill = async (medicalBill) => {
    try {
      const response = await updateMedicalBill(medicalBill);
      setMedicalBills(response.data);
      setEditMode(false);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <Container>
      <Row>
        <Col md={4}>
          <Card>
            <CardBody>
              <CardTitle>Patients</CardTitle>
              <CardSubtitle>Select a patient to view their medical bills</CardSubtitle>
              <ul>
                {patients.map((patient) => (
                  <li key={patient.id}>
                    <Button onClick={() => handlePatientSelect(patient)}>{patient.name}</Button>
                  </li>
                ))}
              </ul>
            </CardBody>
          </Card>
        </Col>
        <Col md={8}>
          {selectedPatient && (
            <Card>
              <CardBody>
                <CardTitle>Medical Bills for {selectedPatient.name}</CardTitle>
                <CardSubtitle>View and edit medical bill information</CardSubtitle>
                <table>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Amount</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {medicalBills.map((medicalBill) => (
                      <tr key={medicalBill.id}>
                        <td>{medicalBill.date}</td>
                        <td>{medicalBill.amount}</td>
                        <td>{medicalBill.description}</td>
                        <td>
                          <Button onClick={() => handleMedicalBillSelect(medicalBill)}>Edit</Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {editMode && (
                  <form>
                    <label>Date:</label>
                    <input type="date" value={selectedMedicalBill.date} onChange={(e) => handleUpdateMedicalBill({ ...selectedMedicalBill, date: e.target.value })} />
                    <label>Amount:</label>
                    <input type="number" value={selectedMedicalBill.amount} onChange={(e) => handleUpdateMedicalBill({ ...selectedMedicalBill, amount: e.target.value })} />
                    <label>Description:</label>
                    <input type="text" value={selectedMedicalBill.description} onChange={(e) => handleUpdateMedicalBill({ ...selectedMedicalBill, description: e.target.value })} />
                    <Button onClick={handleEditModeToggle}>Save Changes</Button>
                  </form>
                )}
                <Button onClick={handleEditModeToggle}>Edit</Button>
              </CardBody>
            </Card>
          )}
          <Card>
            <CardBody>
              <CardTitle>Create New Medical Bill</CardTitle>
                            <CardSubtitle>Enter medical bill information</CardSubtitle>
              <form>
                <label>Patient:</label>
                <select value={newMedicalBill.patientId} onChange={(e) => setNewMedicalBill({ ...newMedicalBill, patientId: e.target.value })}>
                  {patients.map((patient) => (
                    <option key={patient.id} value={patient.id}>{patient.name}</option>
                  ))}
                </select>
                <label>Date:</label>
                <input type="date" value={newMedicalBill.date} onChange={(e) => setNewMedicalBill({ ...newMedicalBill, date: e.target.value })} />
                <label>Amount:</label>
                <input type="number" value={newMedicalBill.amount} onChange={(e) => setNewMedicalBill({ ...newMedicalBill, amount: e.target.value })} />
                <label>Description:</label>
                <input type="text" value={newMedicalBill.description} onChange={(e) => setNewMedicalBill({ ...newMedicalBill, description: e.target.value })} />
                <Button onClick={handleCreateMedicalBill}>Create Medical Bill</Button>
              </form>
            </CardBody>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default MedicalBillingComponent;
