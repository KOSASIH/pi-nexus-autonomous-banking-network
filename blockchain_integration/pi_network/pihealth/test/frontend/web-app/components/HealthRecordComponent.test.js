import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react';
import { HealthRecordComponent } from './HealthRecordComponent';
import { mockHealthRecord } from '../mocks/healthRecords';

describe('HealthRecordComponent', () => {
  it('renders health record details', () => {
    const { getByText } = render(<HealthRecordComponent healthRecord={mockHealthRecord} />);
    expect(getByText(mockHealthRecord.patientName)).toBeInTheDocument();
    expect(getByText(mockHealthRecord.diagnosis)).toBeInTheDocument();
  });

  it('calls onDelete when delete button is clicked', () => {
    const onDelete = jest.fn();
    const { getByRole } = render(<HealthRecordComponent healthRecord={mockHealthRecord} onDelete={onDelete} />);
    const deleteButton = getByRole('button', { name: 'Delete' });
    fireEvent.click(deleteButton);
    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  it('calls onUpdate when update button is clicked', () => {
    const onUpdate = jest.fn();
    const { getByRole } = render(<HealthRecordComponent healthRecord={mockHealthRecord} onUpdate={onUpdate} />);
    const updateButton = getByRole('button', { name: 'Update' });
    fireEvent.click(updateButton);
    expect(onUpdate).toHaveBeenCalledTimes(1);
  });

  it('renders error message when health record is not found', () => {
    const { getByText } = render(<HealthRecordComponent healthRecord={null} />);
    expect(getByText('Health record not found')).toBeInTheDocument();
  });
});
