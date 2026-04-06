import os
import xml.etree.ElementTree as ET
import pandas as pd

def parse_metadata(annotations_dir):
    data = {
        "Number": [],
        "Age": [],
        "Sex": [],
        "Composition": [],
        "Echogenicity": [],
        "Margins": [],
        "Calcifications": [],
        "Tirads": [],
        "Reportbacaf": [],
        "Reporteco": [],
    }

    svg_strings = {}

    xmls = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]

    for xml_file in xmls:
        xml_path = os.path.join(annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        case_number = int(root.find("number").text)
        data["Number"].append(case_number)
        data["Age"].append(int(root.find("age").text) if root.find("age").text else None)
        data["Sex"].append(root.find("sex").text)
        data["Composition"].append(root.find("composition").text)
        data["Echogenicity"].append(root.find("echogenicity").text)
        data["Margins"].append(root.find("margins").text)
        data["Calcifications"].append(root.find("calcifications").text)
        data["Tirads"].append(root.find("tirads").text)
        data["Reportbacaf"].append(root.find("reportbacaf").text)
        data["Reporteco"].append(root.find("reporteco").text)

        for mark in root.findall("mark"):
            image_idx = mark.find("image").text
            svg_strings[f"{case_number}_{image_idx}"] = mark.find("svg").text

    df = pd.DataFrame(data)
    df.set_index("Number", inplace=True)

    return df, svg_strings

if __name__ == "__main__":
    annotations_dir = "data/annotations"
    df, svg_data = parse_metadata(annotations_dir)
    df.to_csv("outputs/metadata.csv")
    print("Saved metadata to outputs/metadata.csv")
