#!/usr/bin/env python3
"""Download required files from HuggingFace for BachGen deployment."""

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SF2_REPO_ID = os.getenv("SF2_REPO_ID", "NicolaDiSalvatore/florestan-ahh-choir-soundfont")
SF2_FILENAME = os.getenv("SF2_FILENAME", "052_Florestan_Ahh_Choir.sf2")
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "NicolaDiSalvatore/BachGen1.0")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model.pth")

DEPLOY_DIR = Path(os.getenv("DEPLOY_DIR", "deploy"))
RESOURCES_DIR = Path(os.getenv("RESOURCES_DIR", "resources"))

REPO_CONFIGS = [
    {
        "repo_id": SF2_REPO_ID,
        "filename": SF2_FILENAME,
        "repo_type": "dataset",
        "local_dir": str(RESOURCES_DIR),
    },
    {
        "repo_id": MODEL_REPO_ID,
        "filename": MODEL_FILENAME,
        "local_dir": str(DEPLOY_DIR),
    },
]


def download_file(repo_id: str, filename: str, repo_type: str | None = None, local_dir: str | None = None) -> Path:
    """Download a single file from HuggingFace Hub."""
    if local_dir:
        local_path = Path(local_dir) / filename
        if local_path.exists():
            logger.info(f"File {filename} already exists at {local_path}, skipping download")
            return local_path

    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
    }
    if repo_type:
        kwargs["repo_type"] = repo_type
    if local_dir:
        kwargs["local_dir"] = local_dir

    logger.info(f"Downloading {filename} from {repo_id}...")
    path = hf_hub_download(**kwargs)
    logger.info(f"Downloaded to: {path}")
    return Path(path)


def main():
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

    for config in REPO_CONFIGS:
        try:
            download_file(**config)
        except Exception as e:
            logger.error(f"Failed to download {config['filename']}: {e}")
            raise

    logger.info("All downloads complete!")


if __name__ == "__main__":
    main()
