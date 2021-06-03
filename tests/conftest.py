import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def data_folder():
    path = Path(os.getenv("SOLARNET_PROCESSED_FOLDER"))
    if not path:
        raise ValueError("Missing path environment variable")
    return path