"""
Batch enrollment script for face images.

Processes images from a source directory and enrolls them via the API.
Expected filename format: {room_id}_{user_name}_{location}_{timestamp}.jpg

Example: 101_john_lobby_20250101120000.jpg
- room_id: 101
- user_name: john
- location: lobby
- timestamp: 20250101120000

Usage:
    # Set credentials via environment variables or .env file
    export API_BASE_URL=http://localhost:8000/api/v1
    export API_AUTH_USERNAME=your-username
    export API_AUTH_PASSWORD=your-password
    export API_TOKEN=your-secret-key

    python scripts/batch_enroll.py
    python scripts/batch_enroll.py --source /path/to/images --processed /path/to/done
"""

import os
import re
import sys
import time
import shutil
import argparse
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
API_AUTH_USERNAME = os.getenv("API_AUTH_USERNAME", "")
API_AUTH_PASSWORD = os.getenv("API_AUTH_PASSWORD", "")
API_TOKEN = os.getenv("API_TOKEN", "")

DEFAULT_SOURCE_DIR = "sample_data/unprocessed_face_images"
DEFAULT_PROCESSED_DIR = "sample_data/processed_face_images"
DEFAULT_FAILED_DIR = "sample_data/failed_face_images"


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse filename format: {room_id}_{user_name}_{location}_{timestamp}.jpg

    Examples:
        101_john_lobby_20250101120000.jpg -> {room: '101', user: 'john', ...}
        202_jane_20250101120000.jpg -> {room: '202', user: 'jane', ...}
    """
    basename = os.path.splitext(filename)[0]

    # Pattern: room_user_location_timestamp
    match = re.match(r"(\d+)_([a-zA-Z0-9_]+)_([a-zA-Z_]+)_(\d{14})", basename)
    if match:
        return {
            "room_id": match.group(1),
            "user_name": match.group(2),
            "location": match.group(3),
            "timestamp": match.group(4),
        }

    # Pattern: room_user_timestamp (no location)
    match = re.match(r"(\d+)_([a-zA-Z0-9_]+)_(\d{14})", basename)
    if match:
        return {
            "room_id": match.group(1),
            "user_name": match.group(2),
            "location": "",
            "timestamp": match.group(3),
        }

    return None


def check_api_health() -> bool:
    """Check if the API is available."""
    try:
        # Strip /api/v1 suffix to get base URL for health check
        base_url = API_BASE_URL.rsplit("/api/", 1)[0]
        response = requests.get(
            f"{base_url}/health",
            auth=(API_AUTH_USERNAME, API_AUTH_PASSWORD),
            timeout=5,
        )
        if response.status_code == 200:
            print(f"{Colors.OKGREEN}API is healthy{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}API health check failed: {response.status_code}{Colors.ENDC}")
            return False
    except Exception as e:
        print(f"{Colors.FAIL}Cannot connect to API: {e}{Colors.ENDC}")
        return False


def enroll_face(filepath: str, info: Dict[str, str]) -> tuple:
    """Enroll a face using the API. Returns (success, message)."""
    try:
        filename = os.path.basename(filepath)

        with open(filepath, "rb") as f:
            files = {"image": (filename, f, "image/jpeg")}
            data = {
                "user_name": info["user_name"],
                "user_email": f"{info['user_name']}@example.com",
            }

            headers = {"x-face-token": API_TOKEN}
            response = requests.post(
                f"{API_BASE_URL}/faces/enroll",
                files=files,
                data=data,
                headers=headers,
                auth=(API_AUTH_USERNAME, API_AUTH_PASSWORD),
                timeout=30,
            )

            if response.status_code in [200, 201]:
                result = response.json()
                face_data = result.get("face", {})
                confidence = face_data.get("confidence_score")
                confidence_str = f"{confidence:.2%}" if confidence else "N/A"
                return True, confidence_str
            else:
                error_msg = response.text[:200]
                return False, error_msg

    except Exception as e:
        return False, str(e)


def move_file(src: str, dst_dir: str):
    """Move file to destination directory, handling duplicates."""
    os.makedirs(dst_dir, exist_ok=True)
    filename = os.path.basename(src)
    dst = os.path.join(dst_dir, filename)

    counter = 1
    base, ext = os.path.splitext(dst)
    while os.path.exists(dst):
        dst = f"{base}_{counter}{ext}"
        counter += 1

    shutil.move(src, dst)
    return dst


def main():
    parser = argparse.ArgumentParser(description="Batch enroll face images via the API")
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE_DIR, help="Directory containing images to enroll"
    )
    parser.add_argument(
        "--processed", default=DEFAULT_PROCESSED_DIR, help="Directory for successfully enrolled images"
    )
    parser.add_argument(
        "--failed", default=DEFAULT_FAILED_DIR, help="Directory for failed enrollment images"
    )
    args = parser.parse_args()

    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'BATCH FACE ENROLLMENT':^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")

    if not check_api_health():
        print(f"\n{Colors.FAIL}API is not available. Start it with: docker-compose up -d{Colors.ENDC}\n")
        sys.exit(1)

    if not os.path.isdir(args.source):
        print(f"{Colors.FAIL}Directory not found: {args.source}{Colors.ENDC}")
        sys.exit(1)

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(Path(args.source).glob(ext))

    if not image_files:
        print(f"{Colors.WARNING}No images found in {args.source}{Colors.ENDC}")
        sys.exit(0)

    print(f"Found {len(image_files)} images to process\n")

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for i, filepath in enumerate(sorted(image_files), 1):
        filename = filepath.name
        print(f"[{i}/{len(image_files)}] {filename}... ", end="", flush=True)

        info = parse_filename(filename)
        if not info:
            print(f"{Colors.WARNING}SKIP (invalid format){Colors.ENDC}")
            skipped_count += 1
            continue

        success, message = enroll_face(str(filepath), info)

        if success:
            print(f"{Colors.OKGREEN}OK {info['user_name']} (confidence: {message}){Colors.ENDC}")
            success_count += 1
            move_file(str(filepath), args.processed)
        else:
            print(f"{Colors.FAIL}FAILED: {message[:50]}{Colors.ENDC}")
            failed_count += 1
            move_file(str(filepath), args.failed)

        time.sleep(0.2)

    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}SUMMARY:{Colors.ENDC}")
    print(f"  Total processed: {len(image_files)}")
    print(f"  {Colors.OKGREEN}Successful: {success_count}{Colors.ENDC}")
    print(f"  {Colors.FAIL}Failed: {failed_count}{Colors.ENDC}")
    print(f"  {Colors.WARNING}Skipped: {skipped_count}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


if __name__ == "__main__":
    main()
