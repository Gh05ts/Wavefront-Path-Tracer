#!/usr/bin/env python3
"""Recreate the selected research scenes and PNG-compatible Niagara Bistro.

Run from any directory: python3 /path/to/Wavefront-Path-Tracer/tools/setup_assets.py
Requires Git and Pillow; Git LFS is optional. Existing valid assets are preserved.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from urllib.parse import quote, unquote, urlsplit
from urllib.request import urlopen

from convert_dds_gltf import convert
from prepare_gltf_lighting import prepare_lighting


ROOT = Path(__file__).resolve().parents[1]
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def git(root, *args, env=None):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, env=env
    ).strip()


def initialize(root, source, verify_only=False):
    relative = source["path"]
    checkout = root / relative
    entry = git(root, "ls-files", "--stage", "--", relative).split()
    if len(entry) < 4 or entry[0] != "160000" or entry[1] != source["revision"]:
        raise ValueError(f"Submodule gitlink does not match the pinned revision: {relative}")
    fresh = not (checkout / ".git").exists()
    if fresh:
        if verify_only:
            raise FileNotFoundError(f"Submodule is not initialized: {relative}")
        # Avoid fetching all the source-format LFS files. Runtime payloads below
        # are fetched individually and checked against pinned content hashes.
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        no_smudge = ("-c", "filter.lfs.required=false", "-c", "filter.lfs.smudge=",
                     "-c", "filter.lfs.process=")
        git(root, *no_smudge, "submodule", "update", "--init",
            "--depth", "1", "--", relative, env=env)
        if source.get("sparse_paths"):
            git(checkout, *no_smudge, "sparse-checkout", "set", *source["sparse_paths"], env=env)
    if git(checkout, "rev-parse", "HEAD") != source["revision"]:
        raise ValueError(f"Unexpected revision in {relative}; existing checkout was preserved")
    return checkout


def is_pointer(path):
    with path.open("rb") as stream:
        return stream.read(len(LFS_HEADER)) == LFS_HEADER


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def safe_path(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Asset path escapes its repository: {relative}")
    return path


def fetch_lfs(root, record, url, verify_only=False):
    target = safe_path(root, record["path"])
    if target.is_file() and not is_pointer(target):
        if target.stat().st_size != record["bytes"] or digest(target) != record["sha256"]:
            raise ValueError(f"Existing asset has unexpected content; preserved: {target}")
        return False
    if verify_only:
        raise FileNotFoundError(f"Missing LFS payload: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".download", delete=False) as output:
            temporary = Path(output.name)
            checksum, size = hashlib.sha256(), 0
            with urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > record["bytes"]:
                        raise ValueError(f"Download exceeds expected size: {target}")
                    checksum.update(chunk)
                    output.write(chunk)
        if size != record["bytes"] or checksum.hexdigest() != record["sha256"]:
            raise ValueError(f"Download failed size/SHA-256 verification: {target}")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Downloaded {record['path']} ({size:,} bytes)", flush=True)
    return True


def verify_document(root, relative):
    source = safe_path(root, relative)
    if is_pointer(source):
        raise ValueError(f"Unresolved LFS pointer: {source}")
    if source.suffix.lower() == ".glb":
        with source.open("rb") as stream:
            magic, version, length = struct.unpack("<4sII", stream.read(12))
            chunk_size, chunk_type = struct.unpack("<II", stream.read(8))
            if (magic != b"glTF" or version != 2 or length != source.stat().st_size
                    or chunk_type != 0x4E4F534A or chunk_size > length - 20):
                raise ValueError(f"Invalid GLB header: {source}")
            document = json.loads(stream.read(chunk_size))
    else:
        document = json.loads(source.read_text())
    for item in document.get("buffers", []) + document.get("images", []):
        uri = item.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        parsed = urlsplit(uri)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError(f"Expected a local glTF URI: {uri}")
        path = safe_path(root, source.parent.relative_to(root) / unquote(parsed.path))
        if not path.is_file() or is_pointer(path):
            raise FileNotFoundError(f"Missing glTF dependency: {path}")
        if "byteLength" in item and path.stat().st_size < item["byteLength"]:
            raise ValueError(f"Truncated buffer: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="Check existing assets without downloads or file changes")
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "tools/asset_sources.json").read_text())
    research = initialize(ROOT, manifest["research"], args.verify_only)
    bistro = initialize(ROOT, manifest["bistro"], args.verify_only)
    # Generated PNGs are untracked. Refuse to reuse conversions of edited inputs.
    if git(bistro, "diff", "--name-only", "HEAD"):
        raise ValueError("Bistro has edited tracked source files; existing assets were preserved")
    source = manifest["research"]
    base = source["url"].removesuffix(".git").replace(
        "https://github.com/", "https://media.githubusercontent.com/media/")
    downloaded = 0
    for record in source["lfs_files"]:
        url = f"{base}/{source['revision']}/{quote(record['path'], safe='/')}"
        downloaded += fetch_lfs(research, record, url, args.verify_only)
    for relative in source["documents"]:
        verify_document(research, relative)
    derived = convert(bistro / "bistro.gltf", reuse=True, verify_only=args.verify_only)
    verify_document(bistro, derived.relative_to(bistro))
    balanced = prepare_lighting(derived, ROOT / "assets/scenes/bistro-lighting.gltf",
                                ROOT / "assets/scenes/bistro-lighting.json", verify_only=args.verify_only,
                                replace_existing=not args.verify_only)
    verify_document(ROOT, balanced.relative_to(ROOT))
    print(f"Assets ready: {len(source['documents'])} research variants and Bistro; "
          f"{downloaded} LFS payloads downloaded. Source files and licenses preserved.")


if __name__ == "__main__":
    main()
