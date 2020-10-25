import pathlib
from typing import List

from tqdm import tqdm
import numpy as np
import click
import cv2

from model_utils import load_net, forward_pass
from utils import derive_bounding_boxes, CONF_THRESHOLD, NMS_THRESHOLD, BoundingBox

IMAGE_FORMATS = [".png", ".jpg", ".jpeg"]
DEBUG = False


def list_all_images(image_folder: pathlib.Path) -> List[pathlib.Path]:

    images_list = list()

    for ext in IMAGE_FORMATS:
        images = list(image_folder.rglob("*%s" % ext))
        images_list += images
    
    return images_list


def load_image(image_path: pathlib.Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    return img


def adapt_bounding_box(bb: BoundingBox, padding: int = 5):

    bb_width = bb.right - bb.left
    bb_height = bb.bottom - bb.top

    padding_pct = padding/100

    padding_vertical = int(bb_height * (1 + padding_pct))
    padding_horizontal = int(bb_width * (1 + padding_pct))

    top, left, bottom, right = bb.top, bb.left, bb.bottom, bb.right

    top -= padding_vertical
    bottom += padding_vertical
    left -= padding_horizontal
    right += padding_horizontal

    return BoundingBox(left, top, right, bottom, 1.0)


def cut_bounding_box_from_image(img_data: np.ndarray, bb: BoundingBox):

    top = bb.top if bb.top >= 0 else 0
    left = bb.left if bb.left >= 0 else 0
    bottom = bb.bottom if bb.bottom < img_data.shape[0] else img_data.shape[0]-1
    right = bb.right if bb.right < img_data.shape[1] else img_data.shape[1]-1

    return img_data[top:bottom, left:right, :]


def cut_bounding_boxes(img_data: np.ndarray, bb_list: List[BoundingBox]) -> List[np.ndarray]:

    image_parts = list()
    for bb in bb_list:
        image_parts.append(cut_bounding_box_from_image(img_data, bb))

    return image_parts


@click.command()
@click.argument("image_folder", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path(exists=False))
@click.option("-c", "--config-path", default="./cfg/yolov3-face.cfg")
@click.option("-w", "--weights-path", default="./model-weights/yolov3-wider_16000.weights")
def main(config_path: str, weights_path: str, image_folder: str, output_folder: str):

    net = load_net(config_path, weights_path)

    print("Net successfully loaded.")

    output_folder = pathlib.Path(output_folder)

    if not output_folder.is_dir():
        print("Output folder '%s' created!" % output_folder.absolute())
        output_folder.mkdir()

    image_folder = pathlib.Path(image_folder)

    images = list_all_images(image_folder)

    print("%d images found!" % len(images))

    if DEBUG:
        images = images[100:200]

    print("Writing images to %s" % str(output_folder))

    for image_path in tqdm(images):

        image_basename = image_path.name
        image_basename_wo_ext = "".join(image_basename.split(".")[:-1])
        image_ext = image_basename.split(".")[-1]

        image_data = cv2.imread(str(image_path))

        outs = forward_pass(net, image_data)

        bounding_boxes = derive_bounding_boxes(image_data, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        bounding_boxes_padded = [adapt_bounding_box(bb) for bb in bounding_boxes]

        if len(bounding_boxes) < 1:
            continue

        bounding_box_images = cut_bounding_boxes(image_data, bounding_boxes_padded)

        for i, bb_img in enumerate(bounding_box_images):
            
            bb_image_basename = "%s_%03d.%s" % (image_basename_wo_ext, i, "jpg")
            bb_image_path = output_folder / bb_image_basename

            if bb_img.size > 0:
                cv2.imwrite(str(bb_image_path), bb_img)



if __name__=="__main__":
    main()
