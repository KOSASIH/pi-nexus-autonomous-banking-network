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

  @Column()
  nonce: number;

  @Column()
  transactionIndex: number;

  @Column()
  blockHash: string;
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

@Entity()
export class Token {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  tokenAddress: string;

  @Column()
  tokenName: string;

  @Column()
  tokenSymbol: string;

  @Column()
  totalSupply: number;

  @Column()
  decimals: number;
}
