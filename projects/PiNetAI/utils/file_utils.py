# file_utils.py

import os
import shutil

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def delete_directory(path):
    shutil.rmtree(path, ignore_errors=True)

def copy_file(src, dst):
    shutil.copyfile(src, dst)

def move_file(src, dst):
    shutil.move(src, dst)
