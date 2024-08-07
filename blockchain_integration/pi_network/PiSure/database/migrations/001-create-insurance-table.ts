import { MigrationInterface, QueryRunner } from 'typeorm';

export class CreateInsuranceTable1589465113411 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE insurance (
        id SERIAL PRIMARY KEY,
        policy_holder VARCHAR(255) NOT NULL,
        amount DECIMAL(10, 2) NOT NULL,
        premium DECIMAL(10, 2) NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE insurance`);
  }
}
