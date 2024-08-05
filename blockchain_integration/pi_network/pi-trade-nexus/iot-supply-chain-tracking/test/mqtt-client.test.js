import { MqttClient } from '../src/mqtt-client';
import { expect } from 'chai';

describe('MQTT Client', () => {
  let mqttClient;

  beforeEach(() => {
    mqttClient = new MqttClient('mqtt://localhost:1883');
  });

  afterEach(() => {
    mqttClient.disconnect();
  });

  it('should connect to MQTT broker', (done) => {
    mqttClient.connect((err) => {
      expect(err).to.be.null;
      done();
    });
  });

  it('should publish a message to a topic', (done) => {
    mqttClient.publish('test/topic', 'Hello, MQTT!', (err) => {
      expect(err).to.be.null;
      done();
    });
  });

  it('should subscribe to a topic and receive a message', (done) => {
    mqttClient.subscribe('test/topic', (err, granted) => {
      expect(err).to.be.null;
      expect(granted).to.be.an('array');

      mqttClient.on('message', (topic, message) => {
        expect(topic).to.equal('test/topic');
        expect(message.toString()).to.equal('Hello, MQTT!');
        done();
      });

      mqttClient.publish('test/topic', 'Hello, MQTT!');
    });
  });
});
