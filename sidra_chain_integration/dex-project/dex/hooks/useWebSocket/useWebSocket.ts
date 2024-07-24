import { useState, useEffect } from 'react';
import WebSocket from 'ws';

const useWebSocket = (url: string) => {
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [message, setMessage] = useState('');
  const [orders, setOrders] = useState([]);
  const [trades, setTrades] = useState([]);
  const [orderBook, setOrderBook] = useState({});

  useEffect(() => {
    if (!ws) {
      setWs(new WebSocket(url));
    }
  }, [url]);

  useEffect(() => {
    if (ws) {
      ws.onopen = () => {
        setConnected(true);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case 'orders':
            setOrders(data.orders);
            break;
          case 'trades':
            setTrades(data.trades);
            break;
          case 'orderBook':
            setOrderBook(data.orderBook);
            break;
          default:
            setMessage(data.message);
        }
      };

      ws.onclose = () => {
        setConnected(false);
      };

      ws.onerror = (event) => {
        console.error(event);
      };
    }
  }, [ws]);

  const sendMessage = (message: string) => {
    if (ws) {
      ws.send(message);
    }
  };

  const subscribeToOrders = () => {
    sendMessage('subscribe:orders');
  };

  const subscribeToTrades = () => {
    sendMessage('subscribe:trades');
  };

  const subscribeToOrderBook = () => {
    sendMessage('subscribe:orderBook');
  };

  return {
    connected,
    message,
    orders,
    trades,
    orderBook,
    subscribeToOrders,
    subscribeToTrades,
    subscribeToOrderBook,
  };
};

export default useWebSocket;
