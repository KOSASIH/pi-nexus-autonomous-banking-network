import { Entity, Column, PrimaryGeneratedColumn, ManyToOne, OneToMany } from 'typeorm';

@Entity()
export class Node {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  name: string;

  @Column()
  description: string;

  @Column()
  incentivizationType: string;

  @Column()
  reputation: number;

  @ManyToOne(() => Node, (node) => node.id)
  parent: Node;

  @OneToMany(() => Node, (node) => node.parent)
  children: Node[];

  @Column('jsonb')
  metadata: {
    [key: string]: any;
  };

  @Column('timestamp', { default: () => 'CURRENT_TIMESTAMP' })
  createdAt: Date;

  @Column('timestamp', { default: () => 'CURRENT_TIMESTAMP', onUpdate: 'CURRENT_TIMESTAMP' })
  updatedAt: Date;
}
