import os
import csv
import xml.etree.ElementTree as ET

BASE_DIR = "Data/NEU-DET"
SPLITS = ["train", "validation"]
OUTPUT_CSV = "data/processed/patch_metadata.csv"

def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        width = xmax - xmin
        height = ymax - ymin
        aspect_ratio = round(width / height, 2) if height != 0 else 0
        area = width * height

        objects.append((label, width, height, aspect_ratio, area))
    return objects

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "filename", "label", "width", "height", "aspect_ratio", "area"])

        for split in SPLITS:
            ann_dir = os.path.join(BASE_DIR, split, "annotations")

            for xml_file in os.listdir(ann_dir):
                if not xml_file.endswith(".xml"):
                    continue

                xml_path = os.path.join(ann_dir, xml_file)
                base_name = os.path.splitext(xml_file)[0]

                try:
                    objects = parse_annotation(xml_path)
                    for idx, (label, w, h, ar, area) in enumerate(objects):
                        patch_name = f"{base_name}_{idx}.jpg"
                        writer.writerow([split, patch_name, label, w, h, ar, area])
                except Exception as e:
                    print(f"❌ Failed to parse {xml_path}: {e}")

    print(f"✅ Saved patch metadata to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
