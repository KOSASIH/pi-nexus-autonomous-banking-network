import { Entity, Column, PrimaryGeneratedColumn, ManyToOne } from 'typeorm';
import { PiCoin } from './piCoin.entity';
import { User } from './user.entity';

@Entity()
export class Transaction {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column('decimal', { precision: 10, scale: 2 })
  amount: number;

  @Column('varchar', { length: 255 })
  type: string; // e.g. 'send', 'receive', etc.

  @Column('datetime', { default: () => 'CURRENT_TIMESTAMP' })
  createdAt: Date;

  @ManyToOne(() => User, (user) => user.id)
  sender: User;

  @ManyToOne(() => User, (user) => user.id)
  recipient: User;

  @ManyToOne(() => PiCoin, (piCoin) => piCoin.id)
  piCoin: PiCoin;
}
