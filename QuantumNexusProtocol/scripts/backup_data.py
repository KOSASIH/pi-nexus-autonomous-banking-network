import shutil
import os
from datetime import datetime

def backup_data():
    source_dir = "/path/to/blockchain/data"
    backup_dir = f"/path/to/backup/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        shutil.copytree(source_dir, backup_dir)
        print(f"Backup completed successfully to {backup_dir}.")
    except Exception as e:
        print("Error during backup:", e)

if __name__ == "__main__":
    backup_data()
