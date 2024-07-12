// virtual_reality.js
import * as THREE from 'three';
import * as VR from 'three-vr';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({
  canvas: document.getElementById('vr-canvas'),
  antialias: true,
});

const vrManager = new VR.Manager(renderer, camera);

function animate() {
  requestAnimationFrame(animate);
  vrManager.render(scene, camera);
}

animate();
