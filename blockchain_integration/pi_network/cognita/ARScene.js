import * as THREE from 'three';
import { ARCamera } from 'three-ar';

class ARScene {
  constructor(element) {
    this.sceneElement = element;
    this.scene = new THREE.Scene();
    this.camera = new ARCamera();
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.sceneElement,
      antialias: true,
    });
    this.renderer.setClearColor(0xffffff, 1);

    this.addObjects();
  }

  addObjects() {
    const cube = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x00ff00 }));
    this.scene.add(cube);
  }

  animate() {
    requestAnimationFrame(() => {
      this.animate();
    });
    this.renderer.render(this.scene, this.camera);
  }
}

export default ARScene;
