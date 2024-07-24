import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Order } from './Order';
import { Repository } from 'typeorm';

@Injectable()
export class OrderRepository {
  constructor(
    @InjectRepository(Order)
    private readonly orderRepository: Repository<Order>,
  ) {}

  async findAll(): Promise<Order[]> {
    return this.orderRepository.find();
  }

  async findOne(id: number): Promise<Order> {
    return this.orderRepository.findOne(id);
  }

  async create(order: Order): Promise<Order> {
    return this.orderRepository.save(order);
  }

  async update(order: Order): Promise<Order> {
    return this.orderRepository.save(order);
  }

  async delete(id: number): Promise<void> {
    await this.orderRepository.delete(id);
  }
}
