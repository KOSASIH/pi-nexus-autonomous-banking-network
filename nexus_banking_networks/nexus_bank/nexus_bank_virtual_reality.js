import * as THREE from 'three';

class VirtualReality {
    constructor() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({
            canvas:document.getElementById('canvas'),
            antialias: true
        });
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    addObject(object) {
        this.scene.add(object);
    }
}

const vr = new VirtualReality();
vr.addObject(new THREE.Mesh(new THREE.SphereGeometry(1, 60, 60), new THREE.MeshBasicMaterial({ color: 0xff0000 })));
vr.render();
