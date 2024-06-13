import { QRCode } from 'qrcode';

class QRCodeGenerator {
  constructor() {
    this.qrCode = new QRCode();
  }

  async generateQRCode(data) {
    const qrCode = await this.qrCode.generate(data);
    return qrCode;
  }
}

export default QRCodeGenerator;
