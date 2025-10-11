#!/usr/bin/env python3
"""Download and prepare InsightFace models during Docker build."""

import os
import shutil
import sys
from pathlib import Path


def main():
    """Download and fix InsightFace models."""
    model_name = os.environ.get("INSIGHTFACE_MODEL", "antelopev2")

    print(f"Downloading {model_name} model...")

    # Import here to ensure dependencies are installed
    from insightface.app import FaceAnalysis

    # Download model (triggers automatic download from GitHub)
    try:
        app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"]
        )
        print("Model download complete")
    except Exception as e:
        # Expected to fail on first load, but download happens
        print(f"Download completed (initialization error expected): {e}")

    # Fix nested directory structure if it occurred
    model_root = Path.home() / ".insightface" / "models"
    model_dir = model_root / model_name
    nested_dir = model_dir / model_name

    if nested_dir.is_dir():
        print(f"Fixing nested directory for {model_name}...")
        for item in nested_dir.iterdir():
            dest = model_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
                print(f"  Moved {item.name}")
        nested_dir.rmdir()
        print(f"✅ Fixed {model_name} directory structure")
    else:
        print(f"✅ {model_name} directory structure is correct")

    # Validate model files exist
    required_files = ["glintr100.onnx", "scrfd_10g_bnkps.onnx"]
    missing = [f for f in required_files if not (model_dir / f).exists()]

    if missing:
        print(f"❌ Model validation failed. Missing files: {missing}")
        print(f"   Model directory: {model_dir}")
        print(f"   Directory contents: {list(model_dir.iterdir())}")
        sys.exit(1)

    print(f"✅ {model_name} model ready and validated")
    print(f"   Location: {model_dir}")
    print(f"   Files: {[f.name for f in model_dir.iterdir()]}")


if __name__ == "__main__":
    main()
