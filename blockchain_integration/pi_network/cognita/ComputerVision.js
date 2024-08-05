import * as cv from 'opencv-js';

class ComputerVision {
  constructor() {
    this.video = document.getElementById('video');
    this.canvas = document.getElementById('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.faceCascade = new cv.CascadeClassifier(cv.FACE_CASCADE);

    this.video.addEventListener('play', () => {
      this.interval = setInterval(() => {
        this.detectFaces();
      }, 100);
    });
  }

  detectFaces() {
    this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    const gray = new cv.Mat();
    cv.cvtColor(this.ctx, gray, cv.COLOR_RGBA2GRAY);
    const faces = new cv.RectVector();
    this.faceCascade.detectMultiScale(gray, faces);
    for (let i = 0; i < faces.size(); i++) {
      const face = faces.get(i);
      this.ctx.beginPath();
      this.ctx.rect(face.x, face.y, face.width, face.height);
      this.ctx.strokeStyle = 'green';
      this.ctx.stroke();
    }
  }
}

export default ComputerVision;
