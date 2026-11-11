#!/usr/bin/env python3
"""Apply offline glTF lighting settings; the renderer needs no scene branches."""
import argparse
import json
import math
import os
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


def prepare_lighting(source: Path, output: Path, settings: Path, *, verify_only=False, replace_existing=False) -> Path:
    source, output, settings = source.resolve(), output.resolve(), settings.resolve()
    if source == output:
        raise ValueError("Output must differ from the source asset")
    profile = json.loads(settings.read_text())
    if set(profile) - {"description", "punctual_intensity_scale", "emissive_intensity_scale"}:
        raise ValueError("Unknown lighting setting")
    scales = [profile.get(key, 1.0) for key in ("punctual_intensity_scale", "emissive_intensity_scale")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in scales):
        raise ValueError("Lighting scales must be finite and nonnegative")
    document = json.loads(source.read_text())
    for light in document.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", []):
        light["intensity"] = light.get("intensity", 1.0) * scales[0]
    for material in document.get("materials", []):
        if "emissiveFactor" in material:
            material["emissiveFactor"] = [v * scales[1] for v in material["emissiveFactor"]]
    # Preserve external payloads and textures when writing beside scene configs.
    for item in document.get("buffers", []) + document.get("images", []):
        uri = item.get("uri", "")
        if not uri or uri.startswith("data:"):
            continue
        if urlsplit(uri).scheme:
            raise ValueError("Only local/data glTF resources are supported")
        path = (source.parent / unquote(uri)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        item["uri"] = quote(Path(os.path.relpath(path, output.parent)).as_posix(), safe="/")
    document.setdefault("asset", {}).setdefault("extras", {})["lighting_profile"] = profile
    if output.exists():
        if json.loads(output.read_text()) == document:
            return output
        if verify_only or not replace_existing:
            raise ValueError(f"Existing derived asset differs; preserved: {output}")
    elif verify_only:
        raise FileNotFoundError(output)
    if not verify_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        temporary.replace(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--replace-existing", action="store_true", help="Regenerate a differing derived output")
    args = parser.parse_args(argv)
    print(prepare_lighting(args.source, args.output, args.settings, verify_only=args.verify_only,
                           replace_existing=args.replace_existing))


if __name__ == "__main__":
    main()
