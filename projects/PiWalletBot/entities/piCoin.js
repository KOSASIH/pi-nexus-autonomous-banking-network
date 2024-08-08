import { Entity, Column, PrimaryGeneratedColumn } from 'typeorm';

@Entity()
export class PiCoin {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column('decimal', { precision: 10, scale: 2 })
  amount: number;

  @Column('varchar', { length: 255 })
  type: string; // e.g. 'Pi Coin', 'Pi Token', etc.

  @Column('boolean', { default: true })
  isAvailable: boolean;

  @Column('datetime', { default: () => 'CURRENT_TIMESTAMP' })
  createdAt: Date;

  @Column('datetime', { default: () => 'CURRENT_TIMESTAMP', onUpdate: 'CURRENT_TIMESTAMP' })
  updatedAt: Date;
}
