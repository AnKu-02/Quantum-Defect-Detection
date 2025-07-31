import os
import cv2
import xml.etree.ElementTree as ET

BASE_DIR = "Data/NEU-DET"
OUT_DIR = "data/processed/patches_32x32"
SPLITS = ["train", "validation"]

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        objects.append((label, xmin, ymin, xmax, ymax))
    return objects

def find_image(img_dir, filename):
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file == f"{filename}.jpg":
                return os.path.join(root, file)
    return None

def extract_and_save(img_path, ann_path, split):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        return

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    objects = parse_annotation(ann_path)

    for idx, (label, xmin, ymin, xmax, ymax) in enumerate(objects):
        patch = img[ymin:ymax, xmin:xmax]
        if patch.size == 0:
            print(f"⚠️ Empty patch for {base_name}, skipped.")
            continue

        patch = cv2.resize(patch, (32, 32))

        save_dir = os.path.join(OUT_DIR, split, label)
        os.makedirs(save_dir, exist_ok=True)

        out_path = os.path.join(save_dir, f"{base_name}_{idx}.jpg")
        cv2.imwrite(out_path, patch)
        print(f"✅ Saved patch: {out_path}")

def process_split(split):
    ann_dir = os.path.join(BASE_DIR, split, "annotations")
    img_dir = os.path.join(BASE_DIR, split, "images")

    for xml_file in os.listdir(ann_dir):
        if not xml_file.endswith(".xml"):
            continue

        base = os.path.splitext(xml_file)[0]
        ann_path = os.path.join(ann_dir, xml_file)
        img_path = find_image(img_dir, base)

        if img_path and os.path.exists(ann_path):
            extract_and_save(img_path, ann_path, split)
        else:
            print(f"⚠️ Image or annotation missing for {base}")

def main():
    for split in SPLITS:
        print(f"\n🚀 Processing {split} set...")
        process_split(split)

if __name__ == "__main__":
    main()
