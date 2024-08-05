import { RabbitMQ } from 'amqplib';

const rabbitMQ = new RabbitMQ('amqp://localhost');

export async function sendMessage(message: string, queue: string) {
  await rabbitMQ.connect();
  await rabbitMQ.channel.assertQueue(queue, { durable: true });
  await rabbitMQ.channel.sendToQueue(queue, Buffer.from(message));
  await rabbitMQ.close();
}

export async function consumeMessage(queue: string, callback: (message: string) => void) {
  await rabbitMQ.connect();
  await rabbitMQ.channel.assertQueue(queue, { durable: true });
  await rabbitMQ.channel.consume(queue, (message) => {
    callback(message.content.toString());
  });
  await rabbitMQ.close();
}
