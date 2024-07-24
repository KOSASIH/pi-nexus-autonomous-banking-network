import * as THREE from 'three';

class VRRenderer {
  constructor(scene, camera, container) {
    this.scene = scene;
    this.camera = camera;
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({
      canvas: container,
      antialias: true,
    });
    this.renderer.setClearColor(0x000000);
  }

  init() {
    this.renderer.setSize(this.container.offsetWidth, this.container.offsetHeight);
  }

  render(scene, camera) {
    this.renderer.render(scene, camera);
  }
}

export default VRRenderer;
