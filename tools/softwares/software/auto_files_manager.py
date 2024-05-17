# Import required modules
import datetime
import logging
import os
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AutoFileManager:
    def __init__(self, root_dir: str, file_types: dict = None, backup_dir: str = None):
        """
        Initialize the AutoFileManager instance.

        :param root_dir: The root directory to manage files.
        :param file_types: A dictionary of file types and their corresponding extensions.
        :param backup_dir: The directory to backup files to.
        """
        self.root_dir = root_dir
        self.file_types = file_types or {
            "images": ["jpg", "jpeg", "png", "gif", "bmp"],
            "videos": ["mp4", "avi", "mov", "wmv"],
            "documents": ["docx", "doc", "pdf", "txt"],
            "audio": ["mp3", "wav", "ogg"],
        }
        self.backup_dir = backup_dir or os.path.join(self.root_dir, "backup")

    def organize_files(self):
        """
        Organize files in the root directory based on their types.
        """
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1][1:].lower()
                for file_type, extensions in self.file_types.items():
                    if file_ext in extensions:
                        dest_dir = os.path.join(self.root_dir, file_type)
                        if not os.path.exists(dest_dir):
                            os.makedirs(dest_dir)
                        try:
                            shutil.move(file_path, dest_dir)
                            logger.info(f"Moved {file} to {dest_dir}")
                        except shutil.Error as e:
                            logger.error(f"Error moving {file}: {e}")
                        break

    def clean_up_empty_dirs(self):
        """
        Remove empty directories in the root directory.
        """
        for root, dirs, files in os.walk(self.root_dir, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if not os.listdir(dir_path):
                    try:
                        os.rmdir(dir_path)
                        logger.info(f"Removed empty directory {dir_path}")
                    except OSError as e:
                        logger.error(f"Error removing directory {dir_path}: {e}")

    def backup_files(self):
        """
        Backup files in the root directory to the specified backup directory.
        """
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                backup_file_path = os.path.join(self.backup_dir, file)
                try:
                    shutil.copy2(file_path, backup_file_path)
                    logger.info(f"Backed up {file} to {backup_file_path}")
                except shutil.Error as e:
                    logger.error(f"Error backing up {file}: {e}")

    def run(self):
        """
        Run the AutoFileManager.
        """
        self.organize_files()
        self.clean_up_empty_dirs()
        self.backup_files()


if __name__ == "__main__":
    manager = AutoFileManager("/path/to/root/directory")
    manager.run()
