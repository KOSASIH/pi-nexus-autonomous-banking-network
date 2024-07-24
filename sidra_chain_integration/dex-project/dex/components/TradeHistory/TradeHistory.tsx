import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useWebSocket } from '../../hooks/useWebSocket';

const TradeHistory = () => {
  const { trades, subscribeToTrades } = useWebSocket();
  const [tradeHistory, setTradeHistory] = useState([]);

  useEffect(() => {
    subscribeToTrades();
  }, []);

  useEffect(() => {
    if (trades) {
      setTradeHistory(trades);
    }
  }, [trades]);

  return (
    <View style={styles.container}>
      <Text>Trade History</Text>
      <View style={styles.tableContainer}>
        {tradeHistory.map((trade, index) => (
          <View key={index} style={styles.tableRow}>
            <Text>{trade.price}</Text>
            <Text>{trade.amount}</Text>
            <Text>{trade.side}</Text>
          </View>
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  tableContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  tableRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
});

export default TradeHistory;
