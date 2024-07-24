import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { LineChart, Grid } from 'react-native-svg-charts';
import { useDispatch, useSelector } from 'react-redux';
import { dexActions } from '../../actions';
import { dexReducer } from '../../reducers';
import { useWalletConnect } from '../../hooks/useWalletConnect';

const Chart = () => {
  const dispatch = useDispatch();
  const chartState = useSelector((state) => state.chart);
  const [data, setData] = useState([]);
  const { connectWallet, walletConnected } = useWalletConnect();

  useEffect(() => {
    if (walletConnected) {
      dispatch(dexActions.fetchChartData());
    }
  }, [walletConnected]);

  useEffect(() => {
    if (chartState.data) {
      setData(chartState.data);
    }
  }, [chartState]);

  return (
    <View style={styles.container}>
      <Text>Chart</Text>
      {data.length > 0 ? (
        <LineChart
          style={{ height: 200 }}
          data={data}
          svg={{ stroke: 'rgb(134, 65, 244)' }}
          contentInset={{ top: 20, bottom: 20 }}
        >
          <Grid />
        </LineChart>
      ) : (
        <Text>Loading...</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default Chart;
