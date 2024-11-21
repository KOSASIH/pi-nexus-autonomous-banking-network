// augmented_reality.js
import * as THREE from "three";
import * as ARJS from "ar.js";

const scene = new THREE.Scene();
const camera = new THREE.Camera();
const renderer = new THREE.WebGLRenderer({
  canvas: document.getElementById("ar-canvas"),
  antialias: true,
});

const markerRoot = new ARJS.MarkerRoot(camera, {
  patternUrl: "pattern.patt",
  markerSize: 1,
});

scene.add(markerRoot);

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

animate();
