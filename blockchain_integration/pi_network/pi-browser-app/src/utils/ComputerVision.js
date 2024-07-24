import * as cv from 'opencv-js';
import * as tf from '@tensorflow/tfjs';

class ComputerVision {
  constructor() {
    this.net = null;
  }

  async loadModel() {
    this.net = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/ mobilenet_v2_1.0_224.tgz');
  }

  async processImage(image) {
    const tensor = tf.tensor3d(image, [1, 224, 224, 3]);
    const output = this.net.predict(tensor);
    const predictions = output.dataSync();
    const topPrediction = predictions[0];
    return topPrediction;
  }

  async detectObjects(image) {
    const gray = new cv.Mat();
    cv.cvtColor(image, gray, cv.COLOR_RGBA2GRAY);
    const threshold = 127;
    const maxThreshold = 255;
    const thresholdType = cv.THRESH_BINARY;
    const binaryImage = new cv.Mat();
    cv.threshold(gray, binaryImage, threshold, maxThreshold, thresholdType);
    const contours = new cv.MatVector();
    cv.findContours(binaryImage, contours, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    const objects = [];
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);
      if (area > 100) {
        const x = contour.data32S[0];
        const y = contour.data32S[1];
        const w = contour.data32S[2];
        const h = contour.data32S[3];
        objects.push({ x, y, w, h });
      }
    }
    return objects;
  }
}

export default ComputerVision;
