import { PostgresConnectionOptions } from 'typeorm/driver/postgres/PostgresConnectionOptions';

const config: PostgresConnectionOptions = {
  type: 'postgres',
  url: process.env.DATABASE_URL,
  synchronize: true,
  logging: true,
  entities: [__dirname + '/../models/*.entity{.ts,.js}'],
  migrations: [__dirname + '/../migrations/*{.ts,.js}'],
};

export default config;
