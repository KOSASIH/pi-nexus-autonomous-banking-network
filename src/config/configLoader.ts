import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';
import * as yaml from 'js-yaml';
import * as Joi from 'joi';
import * as winston from 'winston';

// Load environment variables
dotenv.config();

interface DatabaseConfig {
    host: string;
    port: number;
    user: string;
    password: string;
    name: string;
}

interface ApiConfig {
    key: string;
}

interface LoggingConfig {
    level: string;
    file: string;
}

interface AppConfig {
    database: DatabaseConfig;
    api: ApiConfig;
    logging: LoggingConfig;
}

// Define the validation schema
const schema = Joi.object({
    database: Joi.object({
        host: Joi.string().hostname().required(),
        port: Joi.number().integer().min(1).max(65535).required(),
        user: Joi.string().min(1).required(),
        password: Joi.string().min(8).required(), // Require a minimum length for security
        name: Joi.string().min(1).required(),
    }).required(),
    api: Joi.object({
        key: Joi.string().min(1).required(),
    }).required(),
    logging: Joi.object({
        level: Joi.string().valid('error', 'warn', 'info', 'debug').required(),
        file: Joi.string().required(),
    }).required(),
});

const loadConfig = (): AppConfig => {
    const env = process.env.NODE_ENV || 'development';
    const jsonConfigPath = path.join(__dirname, 'config.json');
    const yamlConfigPath = path.join(__dirname, 'config.yaml');

    let config: AppConfig = { database: {} as DatabaseConfig, api: {} as ApiConfig, logging: {} as LoggingConfig };

    // Load JSON config
    if (fs.existsSync(jsonConfigPath)) {
        const jsonConfig = JSON.parse(fs.readFileSync(jsonConfigPath, 'utf8'));
        config = { ...config, ...jsonConfig[env] };
    }

    // Load YAML config
    if (fs.existsSync(yamlConfigPath)) {
        const yamlConfig = yaml.load(fs.readFileSync(yamlConfigPath, 'utf8'));
        config = { ...config, ...yamlConfig[env] };
    }

    // Load environment variables
    config.database.host = process.env.DB_HOST || config.database.host;
    config.database.port = Number(process.env.DB_PORT) || config.database.port;
    config.database.user = process.env.DB_USER || config.database.user;
    config.database.password = process.env.DB_PASSWORD || config.database.password;
    config.database.name = process.env.DB_NAME || config.database.name;
    config.api.key = process.env.API_KEY || config.api.key;

    // Validate configuration
    const { error } = schema.validate(config);
    if (error) {
        throw new Error(`Configuration validation error: ${error.message}`);
    }

    // Set up logging
    const logger = winston.createLogger({
        level: config.logging.level,
        format: winston.format.json(),
        transports: [
            new winston.transports.File({ filename: config.logging.file }),
            new winston.transports.Console(),
        ],
    });

    logger.info('Configuration loaded successfully');

    return config;
};

export default loadConfig;
