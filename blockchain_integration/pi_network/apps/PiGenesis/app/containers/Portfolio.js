import React, { useState, useEffect } from 'react';
import { Table, TableHead, TableBody, TableRow, TableCell } from '@material-ui/core';
import { useWeb3React } from '@web3-react/core';
import { Web3Provider } from '@ethersproject/providers';
import { useEthers } from '@ethersproject/ethers-react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBitcoin, faEthereum } from '@fortawesome/free-brands-svg-icons';

const Portfolio = () => {
  const [portfolioData, setPortfolioData] = useState([]);
  const [account, setAccount] = useState(null);

  const { library } = useWeb3React();
  const { ethers } = useEthers();

  useEffect(() => {
    const fetchPortfolioData = async () => {
      try {
        const response = await library.send('eth_call', [
          {
            to: '0x...PortfolioContractAddress...',
            data: '0x...getPortfolio...',
          },
          'latest',
        ]);
        const data = await response.json();
        setPortfolioData(data);
      } catch (error) {
        console.error(error);
      }
    };
    fetchPortfolioData();
  }, [library]);

  useEffect(() => {
    const fetchAccount = async () => {
      try {
        const response = await library.send('eth_accounts', []);
        const accounts = await response.json();
        setAccount(accounts[0]);
      } catch (error) {
        console.error(error);
      }
    };
    fetchAccount();
  }, [library]);

  return (
    <div className="portfolio">
      <h1>Portfolio</h1>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Asset</TableCell>
            <TableCell>Balance</TableCell>
            <TableCell>Value</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {portfolioData.map((row, index) => (
            <TableRow key={index}>
              <TableCell>
                {row.asset === 'ETH'? (
                  <FontAwesomeIcon icon={faEthereum} size="lg" />
                ) : (
                  <FontAwesomeIcon icon={faBitcoin} size="lg" />
                )}
                {row.asset}
              </TableCell>
              <TableCell>{row.balance}</TableCell>
              <TableCell>{row.value}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default Portfolio;
