const cv = require('opencv-js');

class ComputerVisionQualityControlController {
  async performQualityCheck(req, res) {
    const { imageBase64 } = req.body;
    const imageBuffer = Buffer.from(imageBase64, 'base64');
    const image = cv.imread(imageBuffer);
    const result = analyzeImage(image);
    res.json({ result });
  }
}
