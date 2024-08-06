import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { MedicalBillingComponent } from './MedicalBillingComponent';
import { mockMedicalBill } from '../mocks/medicalBills';

describe('MedicalBillingComponent', () => {
  it('renders medical bill details', () => {
    const { getByText } = render(<MedicalBillingComponent medicalBill={mockMedicalBill} />);
    expect(getByText(mockMedicalBill.patientName)).toBeTruthy();
    expect(getByText(mockMedicalBill.billAmount.toString())).toBeTruthy();
  });

  it('calls onDelete when delete button is pressed', () => {
    const onDelete = jest.fn();
    const { getByRole } = render(<MedicalBillingComponent medicalBill={mockMedicalBill} onDelete={onDelete} />);
    const deleteButton = getByRole('button', { name: 'Delete' });
    fireEvent.press(deleteButton);
    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  it('calls onUpdate when update button is pressed', () => {
    const onUpdate = jest.fn();
    const { getByRole } = render(<MedicalBillingComponent medicalBill={mockMedicalBill} onUpdate={onUpdate} />);
    const updateButton = getByRole('button', { name: 'Update' });
    fireEvent.press(updateButton);
    expect(onUpdate).toHaveBeenCalledTimes(1);
  });

  it('renders error message when medical bill is not found', () => {
    const { getByText } = render(<MedicalBillingComponent medicalBill={null} />);
    expect(getByText('Medical bill not found')).toBeTruthy();
  });
});
