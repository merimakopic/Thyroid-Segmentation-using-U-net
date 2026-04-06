import os
import json
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from tqdm import tqdm

def parse_svg(svg_string):
    try:
        polygons = json.loads(svg_string)
    except json.JSONDecodeError:
        return []

    parsed_polygons = []
    for region in polygons:
        points = region.get("points", [])
        polygon = [(int(p["x"]), int(p["y"])) for p in points]
        if polygon:
            parsed_polygons.append(polygon)
    return parsed_polygons

def create_mask(image_path, xml_path, output_path, expected_image_number):
    img = Image.open(image_path)
    width, height = img.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for mark in root.findall("mark"):
        image_elem = mark.find("image")
        svg_elem = mark.find("svg")

        if image_elem is None or svg_elem is None:
            continue

        if image_elem.text != expected_image_number:
            continue 

        if not svg_elem.text:
            continue

        polygons = parse_svg(svg_elem.text)
        for polygon in polygons:
            draw.polygon(polygon, fill=255)

    mask.save(output_path)

def convert_all(images_dir, annotations_dir, masks_dir):
    os.makedirs(masks_dir, exist_ok=True)
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]

    for img_name in tqdm(images, desc="Generating masks"):
        base_name = os.path.splitext(img_name)[0]

        try:
            case_number, image_number = base_name.split("_")
        except ValueError:
            print(f"Skipping {img_name}: unexpected name format.")
            continue

        image_path = os.path.join(images_dir, img_name)
        xml_path = os.path.join(annotations_dir, f"{case_number}.xml")
        mask_path = os.path.join(masks_dir, f"{base_name}_mask.png")

        if os.path.exists(xml_path):
            create_mask(image_path, xml_path, mask_path, expected_image_number=image_number)
        else:
            print(f"Warning: XML not found for {img_name}")

if __name__ == "__main__":
    images_dir = "data/images"
    annotations_dir = "data/annotations"
    masks_dir = "data/masks"

    convert_all(images_dir, annotations_dir, masks_dir)
