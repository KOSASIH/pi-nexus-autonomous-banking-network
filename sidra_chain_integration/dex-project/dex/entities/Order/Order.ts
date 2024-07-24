import { Entity, Column, PrimaryGeneratedColumn } from 'typeorm';

@Entity()
export class Order {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  userId: string;

  @Column()
  symbol: string;

  @Column()
  side: string;

  @Column()
  amount: number;

  @Column()
  price: number;

  @Column()
  type: string;

  @Column()
  status: string;

  @Column()
  createdAt: Date;

  @Column()
  updatedAt: Date;
}
