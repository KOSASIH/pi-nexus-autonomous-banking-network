import os
import zipfile

import requests


def auto_upgrader(repo_url: str, local_path: str) -> None:
    """
    Auto-upgrades the local codebase to the latest version from the specified repository.

    Args:
        repo_url (str): URL of the GitHub repository.
        local_path (str): Path to the local codebase.
    """
    # Get the latest release from the repository
    response = requests.get(f"{repo_url}/releases/latest")
    latest_release = response.json()["tag_name"]

    # Download the latest release as a ZIP file
    zip_url = f"{repo_url}/releases/download/{latest_release}/pi-nexus-autonomous-banking-network.zip"
    zip_response = requests.get(zip_url, stream=True)
    with open("pi-nexus-autonomous-banking-network.zip", "wb") as f:
        for chunk in zip_response.iter_content(1024):
            f.write(chunk)

    # Extract the ZIP file to the local path
    with zipfile.ZipFile("pi-nexus-autonomous-banking-network.zip", "r") as zip_ref:
        zip_ref.extractall(local_path)

    # Remove the ZIP file
    os.remove("pi-nexus-autonomous-banking-network.zip")

    print(f"Auto-upgrade successful! Latest version: {latest_release}")
