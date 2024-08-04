import React, { useState, useEffect } from 'eact';
import { WebSocket } from 'ws';
import { RTCPeerConnection, RTCSessionDescription } from 'wrtc';
import Chart from 'chart.js';
import canvas from 'canvas';

const RealtimeChart = () => {
  const [chartData, setChartData] = useState([]);
  const [ws, setWs] = useState(null);
  const [pc, setPc] = useState(null);
  const [chart, setChart] = useState(null);
  const [canvas, setCanvas] = useState(null);

  useEffect(() => {
    const wsUrl = 'wss://example.com/ws';
    const wsOptions = {
      // WebSocket options
    };
    const ws = new WebSocket(wsUrl, wsOptions);
    setWs(ws);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setChartData((prevData) => [...prevData, data]);
    };

    ws.onopen = () => {
      console.log('WebSocket connection established');
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    // Create a new RTCPeerConnection
    const pc = new RTCPeerConnection({
      iceServers: [
        { urls: 'tun:stun.l.google.com:19302' },
        { urls: 'tun:stun1.l.google.com:19302' },
      ],
    });
    setPc(pc);

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        ws.send(JSON.stringify({ type: 'candidate', candidate: event.candidate }));
      }
    };

    pc.onaddstream = (event) => {
      console.log('Received stream:', event.stream);
    };

    pc.onremovestream = (event) => {
      console.log('Removed stream:', event.stream);
    };

    // Create a new Chart.js chart
    const chartCanvas = document.getElementById('chart-canvas');
    const chart = new Chart(chartCanvas, {
      type: 'line',
      data: chartData,
      options: {
        // Chart options
      },
    });
    setChart(chart);

    // Create a new canvas element
    const canvas = document.createElement('canvas');
    setCanvas(canvas);

    return () => {
      ws.close();
      pc.close();
    };
  }, []);

  useEffect(() => {
    if (chart && canvas) {
      chart.render({
        canvas: canvas,
        width: 400,
        height: 200,
      });
    }
  }, [chart, canvas]);

  return (
    <div>
      <canvas id="chart-canvas" />
      {chart && (
        <div>
          <h2>Realtime Chart</h2>
          <canvas ref={(canvas) => setCanvas(canvas)} />
        </div>
      )}
    </div>
  );
};

export default RealtimeChart;
