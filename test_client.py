import argparse
import base64
from pathlib import Path

import requests


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--image", required=True, help="Path to jpg/png image")
    args = parser.parse_args()

    api_url = args.api.rstrip("/")
    image_path = Path(args.image)

    print("Health:")
    print(requests.get(f"{api_url}/health", timeout=30).json())

    print("Classes:")
    print(requests.get(f"{api_url}/classes", timeout=30).json())

    payload = {"image": encode_image(image_path)}
    response = requests.post(f"{api_url}/predict", json=payload, timeout=60)
    response.raise_for_status()

    print("Prediction:")
    print(response.json())


if __name__ == "__main__":
    main()
