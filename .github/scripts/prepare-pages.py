"""Copy the static website for deployment, excluding archived video assets."""

import argparse
import json
import shutil
from pathlib import Path


def active_media(video_root):
    """Follow the same manifest fields used by the two website video viewers."""
    required = set()
    headline = video_root / "headline"
    manifest = json.loads((headline / "manifest.json").read_text())
    for stage in manifest["stages"]:
        required.add(headline / stage["video"])
        required.add(headline / stage["poster"])

    for directory in (video_root, video_root / "ogpo", video_root / "fsft"):
        manifest = json.loads((directory / "selected_examples.json").read_text())
        for example in manifest["examples"]:
            video = directory / example["combined_file"]
            required.add(video)
            required.add(video.with_suffix(".jpg"))

    for asset in required:
        if not asset.resolve().is_relative_to(video_root):
            raise ValueError(f"Media path leaves the video directory: {asset}")
        if not asset.is_file():
            raise FileNotFoundError(f"Required website media is missing: {asset}")
    return required


def prepare_site(source, destination):
    source = source.resolve()
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("The deployment directory must be outside the source site.")

    video_root = source / "static/videos/molmo"
    required = active_media(video_root)
    archived = {
        asset
        for asset in video_root.rglob("*")
        if asset.is_file() and asset.suffix.lower() in {".mp4", ".jpg"}
        and asset not in required
    }

    def exclude_archives(directory, names):
        return [name for name in names if Path(directory) / name in archived]

    shutil.copytree(source, destination, ignore=exclude_archives)
    published_size = sum(asset.stat().st_size for asset in destination.rglob("*") if asset.is_file())
    excluded_size = sum(asset.stat().st_size for asset in archived)
    print(f"Prepared {published_size / 1024**2:.2f} MiB in {destination}")
    print(f"Included {len(required)} active videos/posters.")
    print(f"Excluded {len(archived)} archived videos/posters ({excluded_size / 1024**2:.2f} MiB).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Static website source directory")
    parser.add_argument("destination", type=Path, help="New deployment output directory")
    args = parser.parse_args()
    prepare_site(args.source, args.destination)


if __name__ == "__main__":
    main()
