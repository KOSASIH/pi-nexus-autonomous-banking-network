import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, CardBody, CardTitle, CardSubtitle, Button } from 'reactstrap';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { getHealthRecords, updateHealthRecord } from '../api/healthRecordsApi';
import { getPatients } from '../api/patientsApi';

const HealthRecordComponent = () => {
  const [healthRecords, setHealthRecords] = useState([]);
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [selectedHealthRecord, setSelectedHealthRecord] = useState(null);
  const [editMode, setEditMode] = useState(false);

  useEffect(() => {
    const fetchHealthRecords = async () => {
      const response = await getHealthRecords();
      setHealthRecords(response.data);
    };

    const fetchPatients = async () => {
      const response = await getPatients();
      setPatients(response.data);
    };

    fetchHealthRecords();
    fetchPatients();
  }, []);

  const handlePatientSelect = (patient) => {
    setSelectedPatient(patient);
    const healthRecord = healthRecords.find((record) => record.patientId === patient.id);
    setSelectedHealthRecord(healthRecord);
  };

  const handleHealthRecordSelect = (healthRecord) => {
    setSelectedHealthRecord(healthRecord);
  };

  const handleEditModeToggle = () => {
    setEditMode(!editMode);
  };

  const handleUpdateHealthRecord = async (healthRecord) => {
    try {
      const response = await updateHealthRecord(healthRecord);
      setHealthRecords(response.data);
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
              <CardSubtitle>Select a patient to view their health record</CardSubtitle>
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
                <CardTitle>Health Record for {selectedPatient.name}</CardTitle>
                <CardSubtitle>View and edit health record information</CardSubtitle>
                <LineChart width={500} height={300} data={selectedHealthRecord.data}>
                  <Line type="monotone" dataKey="value" stroke="#8884d8" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <CartesianGrid stroke="#ccc" />
                  <Tooltip />
                  <Legend />
                </LineChart>
                {editMode && (
                  <form>
                    <label>Value:</label>
                    <input type="number" value={selectedHealthRecord.value} onChange={(e) => handleUpdateHealthRecord({ ...selectedHealthRecord, value: e.target.value })} />
                    <label>Date:</label>
                    <input type="date" value={selectedHealthRecord.date} onChange={(e) => handleUpdateHealthRecord({ ...selectedHealthRecord, date: e.target.value })} />
                    <Button onClick={handleEditModeToggle}>Save Changes</Button>
                  </form>
                )}
                <Button onClick={handleEditModeToggle}>Edit</Button>
              </CardBody>
            </Card>
          )}
        </Col>
      </Row>
    </Container>
  );
};

export default HealthRecordComponent;
