#!/usr/bin/env python3
"""Create a PNG-only copy of an external-image glTF using Pillow's DDS decoder.

Example:
    python tools/convert_dds_gltf.py assets/niagara_bistro/bistro.gltf

Install Pillow in your Python environment first. The source glTF and DDS files
are preserved. PNGs are written beside DDS files; the derived glTF stays beside
the source so geometry-buffer paths continue to resolve. No gamma correction,
channel swizzling, normal reconstruction, resizing or vertical flipping is done.
Only the base mip is exported; renderer mip generation is a separate concern.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from PIL import Image


def local_path(root, uri):
    parsed = urlsplit(uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"Expected a local image URI: {uri}")
    path = (root / unquote(uri)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image URI escapes the asset directory: {uri}")
    return path


def convert(source, *, reuse=False, verify_only=False):
    source = source.resolve()
    if source.suffix.lower() != ".gltf":
        raise ValueError("Expected a JSON .gltf file")
    target = source.with_name(source.stem + "-png.gltf")
    if target.exists() and not (reuse or verify_only):
        raise FileExistsError(target)
    if verify_only and not target.is_file():
        raise FileNotFoundError(target)
    document = json.loads(source.read_text())
    root = source.parent
    images = document.get("images", [])
    converted = {}
    verified = set()
    result = deepcopy(document)
    remap, new_images = {}, []
    # DDS extension sources and PNG fallback entries can describe the same image.
    # Remap both to one PNG image to avoid loading the full texture set twice.
    identities = {}
    for index, entry in enumerate(images):
        if "uri" not in entry:
            raise ValueError("This utility expects external image URIs")
        path = local_path(root, entry["uri"])
        dds = path if path.suffix.lower() == ".dds" else path.with_suffix(".dds")
        output = path.with_suffix(".png") if path.suffix.lower() == ".dds" else path
        if (path.suffix.lower() == ".dds" or not path.exists()) and not output.exists():
            if verify_only:
                raise FileNotFoundError(output)
            if not dds.is_file():
                raise FileNotFoundError(f"Neither image nor DDS alternative exists: {path}")
            if output not in converted:
                with Image.open(dds) as image:
                    image.load()
                    pixels = image.convert("RGBA")
                    temporary = output.with_name(output.name + ".partial")
                    pixels.save(temporary, format="PNG", compress_level=3)
                    # Check lossless encoding of the decoded bytes, including alpha.
                    with Image.open(temporary) as check:
                        if check.size != pixels.size or check.convert("RGBA").tobytes() != pixels.tobytes():
                            raise ValueError(f"PNG round-trip failed: {output}")
                    temporary.replace(output)
                converted[output] = dds
                print(f"Converted {len(converted)}: {output.relative_to(root)}", flush=True)
        if not output.is_file():
            raise FileNotFoundError(output)
        if (reuse or verify_only) and output not in verified:
            with Image.open(output) as check:
                check.verify()
            verified.add(output)
        image = deepcopy(entry)
        image["uri"] = quote(output.relative_to(root).as_posix(), safe="/")
        image.pop("mimeType", None)
        identity = json.dumps(image, sort_keys=True)
        if identity not in identities:
            identities[identity] = len(new_images)
            new_images.append(image)
        remap[index] = identities[identity]
    result["images"] = new_images
    for texture in result.get("textures", []):
        extensions = texture.get("extensions", {})
        dds = extensions.pop("MSFT_texture_dds", None)
        if "source" in texture:
            texture["source"] = remap[texture["source"]]
        elif dds is not None:
            texture["source"] = remap[dds["source"]]
        if extensions:
            # Other extensions can refer to image indices (e.g. BasisU/WebP).
            # Refuse to emit stale indices rather than silently changing them.
            raise ValueError(f"Unsupported texture extensions: {list(extensions)}")
        texture.pop("extensions", None)
    for key in ("extensionsUsed", "extensionsRequired"):
        if key in result:
            result[key] = [name for name in result[key] if name != "MSFT_texture_dds"]
    if target.exists():
        if json.loads(target.read_text()) != result:
            raise ValueError(f"Existing derived glTF differs from the conversion: {target}")
        print(f"Verified {target}: {len(new_images)} images (existing files preserved)")
        return target
    partial = target.with_name(target.name + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    partial.replace(target)
    print(f"Wrote {target}: {len(new_images)} images, {len(converted)} converted DDS files")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gltf", type=Path)
    parser.add_argument("--reuse", action="store_true", help="Verify and reuse existing output files")
    args = parser.parse_args()
    convert(args.gltf, reuse=args.reuse)
