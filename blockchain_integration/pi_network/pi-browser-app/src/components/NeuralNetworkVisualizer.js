import React, { useState, useEffect } from 'eact';
import * as THREE from 'three';
import { NeuralNetworkAPI } from '../api';

const NeuralNetworkVisualizer = () => {
  const [network, setNetwork] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchNetwork = async () => {
      const response = await NeuralNetworkAPI.getNetwork();
      setNetwork(response.data);
      setLoading(false);
    };
    fetchNetwork();
  }, []);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({
    canvas: document.getElementById('neural-network-canvas'),
    antialias: true,
  });

  useEffect(() => {
    if (network) {
      const nodes = network.layers.map((layer, index) => {
        const node = new THREE.Mesh(new THREE.SphereGeometry(0.5, 60, 60), new THREE.MeshBasicMaterial({ color: 0xffffff }));
        node.position.x = index * 2;
        node.position.y = 0;
        node.position.z = 0;
        return node;
      });

      const edges = network.layers.map((layer, index) => {
        const edge = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff }));
        edge.geometry.setFromPoints([
          new THREE.Vector3(index * 2, 0, 0),
          new THREE.Vector3((index + 1) * 2, 0, 0),
        ]);
        return edge;
      });

      scene.add(...nodes);
      scene.add(...edges);

      camera.position.z = 5;

      const animate = () => {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
      };
      animate();
    }
  }, [network]);

  return (
    <div className="neural-network-visualizer">
      <canvas id="neural-network-canvas" width={window.innerWidth} height={window.innerHeight} />
      {loading? (
        <p>Loading...</p>
      ) : (
        <p>Neural Network Visualizer</p>
      )}
    </div>
  );
};

export default NeuralNetworkVisualizer;
