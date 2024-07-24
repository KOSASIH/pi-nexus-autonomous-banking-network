import * as THREE from 'three';
import * as ARjs from 'ar.js';

class AstralPlaneAR {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.camera = new THREE.Camera();
    this.renderer = new THREE.WebGLRenderer({
      canvas: container,
      antialias: true,
    });
    this.arToolkitContext = new ARjs.Context(this.container, this.camera, this.scene);
  }

  async init() {
    await this.arToolkitContext.init();
    this.animate();
  }

  animate() {
    requestAnimationFrame(this.animate.bind(this));
    this.arToolkitContext.update();
    this.renderer.render(this.scene, this.camera);
  }

  async loadMarker(markerId) {
    const marker = await this.arToolkitContext.loadMarker(markerId);
    const asset = await this.getAssetFromMarker(marker);
    this.addAssetToScene(asset);
  }

  async getAssetFromMarker(marker) {
    const assetId = marker.userData.assetId;
    const asset = await this.api.getAsset(assetId);
    return asset;
  }

  addAssetToScene(asset) {
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(asset.x, asset.y, asset.z);
    this.scene.add(mesh);
  }
}

export default AstralPlaneAR;
