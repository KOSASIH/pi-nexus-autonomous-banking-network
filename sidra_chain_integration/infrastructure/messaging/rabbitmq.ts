import { RabbitMQ } from 'amqplib';

const rabbitMQ = new RabbitMQ('amqp://localhost');

export async function sendMessage(message: string) {
  await rabbitMQ.connect();
  await rabbitMQ.channel.assertQueue('sidra_chain_queue', { durable: true });
  await rabbitMQ.channel.sendToQueue('sidra_chain_queue', Buffer.from(message));
  await rabbitMQ.close();
}
