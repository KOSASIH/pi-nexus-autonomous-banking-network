{
  "name": "pi-sure-database",
  "version": "1.0.0",
  "description": "PiSure database",
  "main": "index.ts",
  "scripts": {
    "migrate": "typeorm migration:run",
    "revert": "typeorm migration:revert",
    "seed": "typeorm seed:run"
  },
  "dependencies": {
    "typeorm": "^0.2.25",
    "pg": "^8.5.1",
    "mongodb": "^3.6.3"
  }
}
