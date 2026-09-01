import json
from pathlib import Path


BUSINESS_FILE = Path(__file__).resolve().parent.parent / "business.json"


def load_business_config() -> dict:
    """Load business configuration from business.json."""

    if not BUSINESS_FILE.exists():
        raise FileNotFoundError(
            f"Business configuration not found: {BUSINESS_FILE}"
        )

    with BUSINESS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)