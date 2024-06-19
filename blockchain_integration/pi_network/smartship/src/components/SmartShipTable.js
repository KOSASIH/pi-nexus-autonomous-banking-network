import React, { useState, useEffect } from 'eact';
import { useWeb3React } from '@web3-react/core';
import { useContractLoader, useContractReader } from 'eth-hooks';
import { Table, TableHead, TableRow, TableCell } from '@material-ui/core';

const SmartShipTable = () => {
  const { account, library } = useWeb3React();
  const contract = useContractLoader('SmartShip', library);
  const [shipments, setShipments] = useState([]);

  useEffect(() => {
    async function fetchData() {
      const shipments = await contract.methods.getShipments().call();
      setShipments(shipments);
    }
    fetchData();
  }, [account, library]);

  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableCell>ID</TableCell>
          <TableCell>Status</TableCell>
          <TableCell>Sender</TableCell>
          <TableCell>Receiver</TableCell>
          <TableCell>Amount</TableCell>
        </TableRow>
      </TableHead>
      <tbody>
        {shipments.map((shipment, index) => (
          <TableRow key={index}>
            <TableCell>{shipment.id}</TableCell>
            <TableCell>{shipment.status}</TableCell>
            <TableCell>{shipment.sender}</TableCell>
            <TableCell>{shipment.receiver}</TableCell>
            <TableCell>{shipment.amount} ETH</TableCell>
          </TableRow>
        ))}
      </tbody>
    </Table>
  );
};

export default SmartShipTable;
