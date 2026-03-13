"""
fetch_ifixit_devices.py

Downloads all hardware device models from the iFixit public API and saves
them as a flat list to ifixit_devices.json. Run this once to cache the data.

Usage:
    python fetch_ifixit_devices.py
"""
import json
import requests

CATEGORIES_URL = "https://www.ifixit.com/api/2.0/categories"
OUTPUT_FILE = "ifixit_devices.json"


def flatten_categories(node, path=None, devices=None):
    """
    Recursively walks the iFixit category tree and builds a flat list of leaf
    device entries, each with their full category path.

    The tree uses None as a leaf node value.
    """
    if path is None:
        path = []
    if devices is None:
        devices = []

    for name, children in node.items():
        current_path = path + [name]
        if children is None:
            # This is a leaf device node
            devices.append({
                "name": name,
                "category": current_path[0] if current_path else "",
                "subcategory": current_path[1] if len(current_path) > 1 else "",
                "path": " > ".join(current_path),
                "url": "https://www.ifixit.com/Device/" + name.replace(" ", "_"),
            })
        elif isinstance(children, dict):
            flatten_categories(children, current_path, devices)

    return devices


def main():
    print(f"📡 Fetching hardware categories from iFixit API...")
    resp = requests.get(CATEGORIES_URL, timeout=30)
    resp.raise_for_status()
    tree = resp.json()

    print("🌳 Flattening category tree...")
    devices = flatten_categories(tree)

    print(f"✅ Found {len(devices):,} device models. Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(devices, f, ensure_ascii=False, indent=2)

    print("🎉 Done!")

    # Print a sample
    print("\nSample entries:")
    for d in devices[:5]:
        print(f"  {d['path']}")


if __name__ == "__main__":
    main()
