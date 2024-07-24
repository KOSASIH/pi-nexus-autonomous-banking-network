import { Injectable } from '@nestjs/common';
import { OrderRepository } from './OrderRepository';
import { Order } from './Order';

@Injectable()
export class OrderService {
  constructor(private readonly orderRepository: OrderRepository) {}

  async createOrder(order: Order) {
    return this.orderRepository.create(order);
  }

  async getOrders() {
    return this.orderRepository.findAll();
  }

  async getOrder(id: number) {
    return this.orderRepository.findOne(id);
  }

  async deleteOrder(id: number) {
    return this.orderRepository.delete(id);
  }
}
