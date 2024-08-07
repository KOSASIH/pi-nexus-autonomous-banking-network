import { Entity, Column, PrimaryGeneratedColumn } from 'typeorm';

@Entity()
export class Insurance {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  policyHolder: string;

  @Column('decimal', { precision: 10, scale: 2 })
  amount: number;

  @Column('decimal', { precision: 10, scale: 2 })
  premium: number;

  @Column()
  createdAt: Date;

  @Column()
  updatedAt: Date;
}
