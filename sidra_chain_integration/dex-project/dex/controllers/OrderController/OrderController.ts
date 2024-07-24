import { Controller, Post, Body, Get, Param, Delete } from '@nestjs/common';
import { OrderService } from './OrderService';
import { Order } from './Order';

@Controller('orders')
export class OrderController {
  constructor(private readonly orderService: OrderService) {}

  @Post()
  async createOrder(@Body() order: Order) {
    return this.orderService.createOrder(order);
  }

  @Get()
  async getOrders() {
    return this.orderService.getOrders();
  }

  @Get(':id')
  async getOrder(@Param('id') id: number) {
    return this.orderService.getOrder(id);
  }

  @Delete(':id')
  async deleteOrder(@Param('id') id: number) {
    return this.orderService.deleteOrder(id);
  }
}
