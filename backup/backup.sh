#!/bin/bash

# Backup script

DATE=$(date +%Y-%m-%d-%H-%M-%S)
BACKUP_DIR="/path/to/backup/directory"

tar -czf "${BACKUP_DIR}/backup-${DATE}.tar.gz" /path/to/data

# Restore script

RESTORE_DIR="/path/to/restore/directory"

tar -xzf "${RESTORE_DIR}/backup-*.tar.gz" -C /path/to/data
