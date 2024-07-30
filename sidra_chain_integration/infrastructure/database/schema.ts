import { Entity, Column, PrimaryGeneratedColumn } from 'typeorm';

@Entity()
export class SidraChainTransaction {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  transactionHash: string;

  @Column()
  blockNumber: number;

  @Column()
  timestamp: Date;

  @Column()
  from: string;

  @Column()
  to: string;

  @Column()
  value: string;

  @Column()
  gasUsed: number;

  @Column()
  gasPrice: number;
}

@Entity()
export class SidraChainBlock {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  blockNumber: number;

  @Column()
  blockHash: string;

  @Column()
  timestamp: Date;

  @Column()
  transactions: SidraChainTransaction[];
}
