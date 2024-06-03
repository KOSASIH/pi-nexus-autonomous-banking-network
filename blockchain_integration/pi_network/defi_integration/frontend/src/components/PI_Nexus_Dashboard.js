import React, { useState, useEffect, useContext, useRef } from 'react';
import Web3 from 'web3';
import { useWeb3React } from '@web3-react/core';
import { Contract } from 'web3/eth/contract';
import PI_Nexus_ABI from '../../../contracts/PI_Nexus_Autonomous_Banking_Network_v3.json';
import { ThemeContext } from '../../contexts/ThemeContext';
import { WalletContext } from '../../contexts/WalletContext';
import { NotificationContext } from '../../contexts/NotificationContext';
import { makeStyles } from '@material-ui/core/styles';
import {
  Container,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Tooltip,
  Zoom,
} from '@material-ui/core';
import { LineChart, Line, CartesianGrid, XAxis, YAxis } from 'recharts';
import { useMediaQuery } from 'react-responsive';
import { useInterval } from 'react-use';
import { debounce } from 'lodash';
import { toast } from 'react-toastify';

const PI_NEXUS_CONTRACT_ADDRESS = '0x...'; // Replace with actual contract address

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
  },
  card: {
    margin: theme.spacing(2),
  },
  table: {
    minWidth: 650,
  },
  progress: {
    width: '100%',
  },
}));

const PI_Nexus_Dashboard = () => {
  const { active, account, library } = useWeb3React();
  const { theme } = useContext(ThemeContext);
  const { wallet } = useContext(WalletContext);
  const { notify } = useContext(NotificationContext);
  const classes = useStyles();
  const [piNexusContract, setPiNexusContract] = useState(null);
  const [governanceProposals, setGovernanceProposals] = useState([]);
  const [rewards, setRewards] = useState([]);
  const [liquidity, setLiquidity] = useState(0);
  const [borrowingRequests, setBorrowingRequests] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [isMobile] = useMediaQuery({ query: '(max-width: 768px)' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (active && library && !piNexusContract) {
      const web3 = new Web3(library.provider);
      const contract = new Contract(
        PI_NEXUS_CONTRACT_ADDRESS,
        PI_Nexus_ABI,
        web3.currentProvider
      );
      setPiNexusContract(contract);
    }
  }, [active, library, piNexusContract]);

  useEffect(() => {
    if (piNexusContract) {
      piNexusContract.methods.getGovernanceProposals().call().then(proposals => {
        setGovernanceProposals(proposals);
      });
      piNexusContract.methods.getRewards().call().then(rewards => {
        setRewards(rewards);
      });
      piNexusContract.methods.getLiquidity().call().then(liquidity => {
        setLiquidity(liquidity);
      });
      piNexusContract.methods.getBorrowingRequests().call().then(requests => {
        setBorrowingRequests(requests);
      });
      piNexusContract.methods.getChartData().call().then(data => {
        setChartData(data);
      });
    }
  }, [piNexusContract]);

  const handleAddProposal = async (title, description, target, value, duration) => {
    setLoading(true);
    try {
      const response = await piNexusContract.methods
        .addProposal(title, description, target, value, duration)
        .send({ from: account });
      notify('success', 'Proposal added successfully!');
      console.log(response);
    } catch (error) {
      notify('error', 'Error adding proposal!');
      console.error(error);
    }
    setLoading(false);
  };

  const handleVote = async (proposalId, support) => {
    setLoading(true);
    try {
      const response = await piNexusContract.methods
        .vote(proposalId, support)
        .send({ from: account });
      notify('success', 'Vote recorded successfully!');
      console.log(response);
    } catch (error) {
      notify('error', 'Error voting on proposal!');
      console.error(error);
    }
    setLoading(false);
  };

  const handleQueue = async (proposalId) => {
    setLoading(true);
    try {
      const response = await piNexusContract.methods
        .queue(proposalId)
        .send({ from: account });
      notify('success', 'Proposal queued successfully!');
      console.log(response);
    } catch (error) {
      notify('error', 'Error queuing proposal!');
      console.error(error);
    }
    setLoading(false);
  };

  const handleExecute = async (proposalId) => {
    setLoading(true);
    try {
      const response = await piNexusContract.methods
        .execute(proposalId)
        .send({ from: account });
      notify('success', 'Proposal executed successfully!');
      console.log(response);
    } catch (error) {
      notify('error', 'Error executing proposal!');
      console.error(error);
    }
    setLoading(false);
  };

  const handleStake = async (amount) => {
    setLoading(true);
    try {
      const response = await piNexusContract.methods
        .stake(amount)
        .send({ from: account });
      notify('success', 'Stake recorded successfully!');
      console.log(response);
    } catch (error) {
      notify('error', 'Error staking tokens!');
      console.error(error);
    }
    setLoading(false);
  };

  const handleWithdraw = async (amount) => {
    setLoading(true);
    try {
      const response = await piNexusContract.methods
        .withdraw(amount)
        .send({ from: account });
      notify('success', 'Withdrawal recorded successfully!');
      console.log(response);
    } catch (error) {
      notify('error', 'Error withdrawing tokens!');
      console.error(error);
    }
    setLoading(false);
  };

  const handleClaimRewards = async () => {
    setLoading(true);
    try {
      const response = await piNexusContract.methods
        .claimRewards()
        .send({ from: account });
      notify('success', 'Rewards claimed successfully!');
      console.log(response);
    } catch (error) {
      notify('error', 'Error claiming rewards!');
      console.error(error);
    }
    setLoading(false);
  };

  const handleBorrow = async (amount, duration) => {
    setLoading(true);
    try {
      const response = await piNexusContract.methods
        .borrow(amount, duration)
        .send({ from: account });
      notify('success', 'Borrow request recorded successfully!');
      console.log(response);
    } catch (error) {
      notify('error', 'Error making borrow request!');
      console.error(error);
    }
    setLoading(false);};

  const handleApproveBorrow = async (borrowId) => {
    setLoading(true);
    try {
      const response = await piNexusContract.methods
        .approveBorrow(borrowId)
        .send({ from: account });
      notify('success', 'Borrow request approved successfully!');
      console.log(response);
    } catch (error) {
      notify('error', 'Error approving borrow request!');
      console.error(error);
    }
    setLoading(false);
  };

  const handleRepay = async (borrowId, amount) => {
    setLoading(true);
    try {
      const response = await piNexusContract.methods
        .repay(borrowId, amount)
        .send({ from: account });
      notify('success', 'Repayment recorded successfully!');
      console.log(response);
    } catch (error) {
      notify('error', 'Error making repayment!');
      console.error(error);
    }
    setLoading(false);
  };

  const handleLiquidate = async (borrowId) => {
    setLoading(true);
    try {
      const response = await piNexusContract.methods
        .liquidate(borrowId)
        .send({ from: account });
      notify('success', 'Liquidation recorded successfully!');
      console.log(response);
    } catch (error) {
      notify('error', 'Error liquidating borrow request!');
      console.error(error);
    }
    setLoading(false);
  };

  useInterval(() => {
    if (piNexusContract) {
      piNexusContract.methods.getChartData().call().then(data => {
        setChartData(data);
      });
    }
  }, 10000);

  if (error) {
    return <div>Error: {error.message}</div>;
  }

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <Container className={classes.root}>
      <Typography variant="h4" component="h1" gutterBottom>
        PI Nexus Dashboard
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6} lg={4}>
          <Card className={classes.card}>
            <CardContent>
              <Typography variant="h6" component="h2">
                Governance
              </Typography>
              <Typography variant="body1" component="p">
                Add a new proposal, vote on existing proposals, and manage the queue.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                variant="contained"
                color="primary"
                size="small"
                onClick={() => setOpenAddProposal(true)}
              >
                Add Proposal
              </Button>
              <Button
                variant="contained"
                color="primary"
                size="small"
                onClick={() => setOpenProposals(true)}
              >
                View Proposals
              </Button>
            </CardActions>
          </Card>
        </Grid>
        <Grid item xs={12} md={6} lg={4}>
          <Card className={classes.card}>
            <CardContent>
              <Typography variant="h6" component="h2">
                Staking
              </Typography>
              <Typography variant="body1" component="p">
                Stake your tokens to earn rewards and participate in governance.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                variant="contained"
                color="primary"
                size="small"
                onClick={() => setOpenStaking(true)}
              >
                Stake Tokens
              </Button>
              <Button
                variant="contained"
                color="primary"
                size="small"
                onClick={() => setOpenStakingHistory(true)}
              >
                Staking History
              </Button>
            </CardActions>
          </Card>
</Grid>
        <Grid item xs={12} md={6} lg={4}>
          <Card className={classes.card}>
            <CardContent>
              <Typography variant="h6" component="h2">
                Borrowing
              </Typography>
              <Typography variant="body1" component="p">
                Borrow tokens using your staked tokens as collateral.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                variant="contained"
                color="primary"
                size="small"
                onClick={() => setOpenBorrowing(true)}
              >
                Borrow Tokens
              </Button>
              <Button
                variant="contained"
                color="primary"
                size="small"
                onClick={() => setOpenBorrowingHistory(true)}
              >
                Borrowing History
              </Button>
            </CardActions>
          </Card>
        </Grid>
      </Grid>
      <Typography variant="h5" component="h2" gutterBottom>
        Liquidity Pool
      </Typography>
      <LineChart width={500} height={300} data={chartData}>
        <Line type="monotone" dataKey="value" stroke="#8884d8" />
        <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
        <XAxis dataKey="date" />
        <YAxis />
      </LineChart>
      <Typography variant="h5" component="h2" gutterBottom>
        Governance Proposals
      </Typography>
      <TableContainer component={Paper}>
        <Table className={classes.table} aria-label="simple table">
          <TableHead>
            <TableRow>
              <TableCell>Proposal ID</TableCell>
              <TableCell>Proposal Title</TableCell>
              <TableCell>Proposal Description</TableCell>
              <TableCell>Target</TableCell>
              <TableCell>Value</TableCell>
              <TableCell>Duration</TableCell>
              <TableCell>Vote</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {governanceProposals.map((proposal, index) => (
              <TableRow key={index}>
                <TableCell component="th" scope="row">
                  {proposal.id}
                </TableCell>
                <TableCell>{proposal.title}</TableCell>
                <TableCell>{proposal.description}</TableCell>
                <TableCell>{proposal.target}</TableCell>
                <TableCell>{proposal.value}</TableCell>
                <TableCell>{proposal.duration}</TableCell>
                <TableCell>
                  <Button
                    variant="contained"
                    color="primary"
                    size="small"
                    onClick={() => handleVote(proposal.id, true)}
                  >
                    Vote For
                  </Button>
                  <Button
                    variant="contained"
                    color="primary"
                    size="small"
                    onClick={() => handleVote(proposal.id, false)}
                  >
                    Vote Against
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <Typography variant="h5" component="h2" gutterBottom>
        Borrowing Requests
      </Typography>
      <TableContainer component={Paper}>
        <Table className={classes.table} aria-label="simple table">
          <TableHead>
            <TableRow>
              <TableCell>Borrow ID</TableCell>
              <TableCell>Borrow Amount</TableCell>
              <TableCell>Borrow Duration</TableCell>
              <TableCell>Approve</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {borrowingRequests.map((request, index) => (
              <TableRow key={index}>
                <TableCell component="th" scope="row">
                  {request.id}
                </TableCell>
                <TableCell>{request.amount}</TableCell>
                <TableCell>{request.duration}</TableCell>
                <TableCell>
                  <Button
                    variant="contained"
                    color="primary"
                    size="small"
                    onClick={() => handleApproveBorrow(request.id)}
                  >
                    Approve
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Container>
  );
};

export default PI_Nexus_Dashboard;
