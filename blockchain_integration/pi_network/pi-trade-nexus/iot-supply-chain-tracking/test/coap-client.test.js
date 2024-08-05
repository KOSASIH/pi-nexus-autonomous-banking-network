import { CoapClient } from '../src/coap-client';
import { expect } from 'chai';

describe('CoAP Client', () => {
  let coapClient;

  beforeEach(() => {
    coapClient = new CoapClient('coap://localhost:5683');
  });

  afterEach(() => {
    coapClient.close();
  });

  it('should send a GET request to a CoAP resource', (done) => {
    coapClient.get('test/resource', (err, response) => {
      expect(err).to.be.null;
      expect(response.code).to.equal('2.05');
      expect(response.payload.toString()).to.equal('Hello, CoAP!');
      done();
    });
  });

  it('should send a POST request to a CoAP resource', (done) => {
    coapClient.post('test/resource', 'Hello, CoAP!', (err, response) => {
      expect(err).to.be.null;
      expect(response.code).to.equal('2.01');
      done();
    });
  });

  it('should observe a CoAP resource and receive notifications', (done) => {
    coapClient.observe('test/resource', (err, response) => {
      expect(err).to.be.null;
      expect(response.code).to.equal('2.05');

      coapClient.on('notification', (response) => {
        expect(response.code).to.equal('2.05');
        expect(response.payload.toString()).to.equal('Hello, CoAP!');
        done();
      });

      // Simulate a notification from the CoAP server
      coapClient.emit('notification', {
        code: '2.05',
        payload: 'Hello, CoAP!'
      });
    });
  });
});
