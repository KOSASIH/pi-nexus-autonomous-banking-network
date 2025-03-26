import fileinput
import os


def auto_modifier(local_path: str, modifier_config: dict) -> None:
    """
    Auto-modifies the local codebase based on the provided configuration.

    Args:
        local_path (str): Path to the local codebase.
        modifier_config (dict): Configuration for the auto-modifier.
    """
    # Iterate through the modifier configuration
    for file_pattern, modifications in modifier_config.items():
        # Find files matching the pattern
        files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(local_path)
            for file in files
            if file_pattern in file
        ]

        # Apply modifications to each file
        for file in files:
            with fileinput.input(file, inplace=True) as f:
                for line in f:
                    for search, replace in modifications.items():
                        line = line.replace(search, replace)
                    print(line, end="")

    print("Auto-modification successful!")
