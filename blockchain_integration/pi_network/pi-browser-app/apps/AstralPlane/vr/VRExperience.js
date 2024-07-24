import * as THREE from 'three';
import { VRRenderer } from './VRRenderer';
import { AstralPlaneAPI } from '../api/AstralPlaneAPI';

class VRExperience {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, container.offsetWidth / container.offsetHeight, 0.1, 1000);
    this.renderer = new VRRenderer(this.scene, this.camera, container);
    this.assets = [];
    this.api = new AstralPlaneAPI();
  }

  async init() {
    await this.loadAssets();
    this.renderer.init();
    this.animate();
  }

  async loadAssets() {
    const assets = await this.api.getAssets();
    assets.forEach((asset) => {
      const geometry = new THREE.BoxGeometry(1, 1, 1);
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(asset.x, asset.y, asset.z);
      this.scene.add(mesh);
      this.assets.push(mesh);
    });
  }

  animate() {
    requestAnimationFrame(this.animate.bind(this));
    this.renderer.render(this.scene, this.camera);
  }

  handleMouseMove(event) {
    const x = event.clientX;
    const y = event.clientY;
    const raycaster = new THREE.Raycaster(this.camera.position, new THREE.Vector3(x, y, 0).sub(this.camera.position).normalize());
    const intersects = raycaster.intersectObjects(this.assets);
    if (intersects.length > 0) {
      const asset = intersects[0].object;
      this.api.buyAsset(asset.assetId);
    }
  }

  handleTouchStart(event) {
    const touch = event.touches[0];
    const x = touch.clientX;
    const y = touch.clientY;
    const raycaster = new THREE.Raycaster(this.camera.position, new THREE.Vector3(x, y, 0).sub(this.camera.position).normalize());
    const intersects = raycaster.intersectObjects(this.assets);
    if (intersects.length > 0) {
      const asset = intersects[0].object;
      this.api.buyAsset(asset.assetId);
    }
  }
}

export default VRExperience;
