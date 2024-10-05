import random
import shlex
import subprocess
import xml.etree.ElementTree as ET

import torchvision


def main(val_percent=0.2, keep_files=False):

    torchvision.datasets.OxfordIIITPet(".", download=True)

    run_cmd("mkdir -p oxford-iiit-pet-yolo/labels/train")
    run_cmd("mkdir -p oxford-iiit-pet-yolo/labels/val")
    run_cmd("mkdir -p oxford-iiit-pet-yolo/labels/test")

    run_cmd("mkdir -p oxford-iiit-pet-yolo/images/train")
    run_cmd("mkdir -p oxford-iiit-pet-yolo/images/val")
    run_cmd("mkdir -p oxford-iiit-pet-yolo/images/test")

    train_val_filenames = read_file("oxford-iiit-pet/annotations/trainval.txt")
    train_val_filenames = [
        filename.split()[:2]
        for filename in train_val_filenames
        if filename != ""
    ]

    # Shuffle and split
    random.shuffle(train_val_filenames)
    val_n = int(val_percent * len(train_val_filenames))

    train_filenames = train_val_filenames[val_n:]
    val_filenames = train_val_filenames[:val_n]

    test_filenames = read_file("oxford-iiit-pet/annotations/test.txt")
    test_filenames = [
        filename.split()[:2]
        for filename in test_filenames
        if filename != ""
    ]

    for set_name, filenames in zip(
        ("train", "val", "test"), (train_filenames, val_filenames, test_filenames)
    ):
        for filename, class_id in filenames:

            try:
                xml_to_yolo(
                    f"oxford-iiit-pet/annotations/xmls/{filename}.xml",
                    f"oxford-iiit-pet-yolo/labels/{set_name}/{filename}.txt",
                    class_id,
                )
            except FileNotFoundError:
                pass

            run_cmd(
                f"cp oxford-iiit-pet/images/{filename}.jpg oxford-iiit-pet-yolo/images/{set_name}/"
            )
    
    # if not keep files, delete the original data
    # TODO


def run_cmd(cmd):
    subprocess.run(shlex.split(cmd))


def read_file(file_path) -> list[str]:

    with open(file_path) as f_obj:
        lines = f_obj.readlines()

    lines = [line.rstrip("\n") for line in lines if line != ""]

    return lines


def xml_to_yolo(xml_path, out_path, class_id) -> None:

    to_write_lines = []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    obj = root.find("object")

    for bndbox in obj.findall("bndbox"):
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        to_write_lines.append(
            f"{int(class_id) - 1} {((xmin + xmax) / 2) / width} {((ymin + ymax) / 2) / height} {(xmax - xmin) / width} {(ymax - ymin) / height}"
        )
    
    with open(out_path, "w") as f_obj:
        f_obj.writelines(to_write_lines)


# class x_center y_center width height


if __name__ == "__main__":
    main()
