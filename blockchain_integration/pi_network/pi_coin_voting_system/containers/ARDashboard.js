import React, { useState, useEffect } from 'eact';
import { connect } from 'eact-redux';
import { getARDashboardData } from '../actions/ar.actions';
import { ThreeCanvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

const ARDashboard = ({ getARDashboardData, arData }) => {
  const [camera, setCamera] = useState(null);
  const [controls, setControls] = useState(null);

  useEffect(() => {
    getARDashboardData();
  }, []);

  useEffect(() => {
    if (arData) {
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000,
      );
      const renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('ar-canvas'),
        antialias: true,
      });

      setCamera(camera);
      setControls(new OrbitControls(camera, renderer.domElement));

      arData.forEach((item) => {
        const mesh = new THREE.Mesh(
          new THREE.SphereGeometry(0.5, 60, 60),
          new THREE.MeshBasicMaterial({ color: item.color }),
        );
        mesh.position.set(item.x, item.y, item.z);
        scene.add(mesh);
      });

      renderer.render(scene, camera);
    }
  }, [arData]);

  return (
    <div>
      <h1>AR Dashboard</h1>
      <ThreeCanvas id="ar-canvas" style={{ width: '800px', height: '400px' }}>
        <OrbitControls ref={controls} enableDamping={false} />
      </ThreeCanvas>
    </div>
  );
};

const mapStateToProps = (state) => {
  return {
    arData: state.ar.data,
  };
};

export default connect(mapStateToProps, { getARDashboardData })(ARDashboard);
