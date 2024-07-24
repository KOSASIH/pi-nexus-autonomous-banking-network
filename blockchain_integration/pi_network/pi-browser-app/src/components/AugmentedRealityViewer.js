import React, { useState, useEffect } from 'react';
import { AugmentedRealityAPI } from '../api';
import * as THREE from 'three';

const AugmentedRealityViewer = () => {
  const [scene, setScene] = useState(new THREE.Scene());
  const [camera, setCamera] = useState(new THREE.Camera());
  const [renderer, setRenderer] = useState(new THREE.WebGLRenderer({
    canvas: document.getElementById('ar-canvas'),
    antialias: true,
  }));
  const [marker, setMarker] = useState(null);
  const [loading, setLoading] = useState(true);
  const [videoStream, setVideoStream] = useState(null);
  const [arToolkit, setArToolkit] = useState(null);

  useEffect(() => {
    const fetchMarker = async () => {
      const response = await AugmentedRealityAPI.getMarker();
      setMarker(response.data);
      setLoading(false);
    };
    fetchMarker();
  }, []);

  useEffect(() => {
    if (marker) {
      const geometry = new THREE.SphereGeometry(0.5, 60, 60);
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.x = marker.x;
      mesh.position.y = marker.y;
      mesh.position.z = marker.z;
      scene.add(mesh);
    }
  }, [marker]);

  useEffect(() => {
    const initARToolkit = async () => {
      const arToolkit = await import('ar.js');
      setArToolkit(arToolkit);
    };
    initARToolkit();
  }, []);

  useEffect(() => {
    if (arToolkit) {
      const video = document.getElementById('video');
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          setVideoStream(stream);
          video.srcObject = stream;
          video.play();
        })
        .catch((error) => {
          console.error('Error accessing camera:', error);
        });
    }
  }, [arToolkit]);

  useEffect(() => {
    if (videoStream && arToolkit) {
      const arContext = new arToolkit.ARContext({
        camera: camera,
        scene: scene,
        renderer: renderer,
        video: videoStream,
      });
      arContext.init();
    }
  }, [videoStream, arToolkit]);

  const handleResize = () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
  };

  window.addEventListener('resize', handleResize);

  return (
    <div className="augmented-reality-viewer">
      <canvas id="ar-canvas" />
      <video id="video" width="640" height="480" />
      {loading ? (
        <p>Loading...</p>
      ) : (
        <p>AR Experience Ready!</p>
      )}
    </div>
  );
};

export default AugmentedRealityViewer;
