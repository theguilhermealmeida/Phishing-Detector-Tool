from __future__ import unicode_literals
import cv2
import json
import os
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

OUPUT_FOLDER_COCO2YOLO = 'project-1-at-2023-06-21-16-21-c0dad87c/yolo_annotations'

# blow out the bbox into max.min values
def expand_bbox_coords(bbox):
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[0] + bbox[2]
    ymax = bbox[1] + bbox[3]

    return (xmin, ymin, xmax, ymax)


def obtain_bbox_label(categories, bbox_tag):
    
    label = categories[categories['id']==bbox_tag['category_id']]['name'].item()
    label = str(label)

    return (label)

def copy_files(list_img, list_annot, split, output_base):

    # copy the images over
    img_folder = os.path.join(output_base, 'images', split)
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    for x in list_img:
        shutil.copy(x, img_folder)

    # copy the annotation files over
    annot_folder = os.path.join(output_base, 'labels', split)
    if not os.path.isdir(annot_folder):
        os.makedirs(annot_folder)

    for x in list_annot:
        shutil.copy(x, annot_folder)
    
    return 

def coco_to_dict(img, annots_df):

    # obtain relevant image data
    img_name = os.path.basename(img['file_name'])
    img_size = (img['width'], img['height'], 3)
    
    # cut to relevant bbox annotations
    img_id = img['id']
    tags = annots_df[annots_df['image_id']==img_id]

    # iterate through bbox annotations
    bboxes = []
    for _, tag in tags.iterrows():
        bbox_annot = {
            'label': tag['category_id'],
            'xmin': tag['xmin'],
            'ymin': tag['ymin'],
            'xmax': tag['xmax'],
            'ymax': tag['ymax']
        }
        bboxes.append(bbox_annot)
       
    img_dict = {
        'bboxes': bboxes,
        'image_name': img_name,
        'image_size': img_size
    }
    
    return img_dict


def dict_to_yolo(img_dict):
    img_name = img_dict['image_name']
    img_width, img_height, img_depth = img_dict['image_size']

    annot_txt =[]
    for box in img_dict['bboxes']:

        # extract abs bbox info
        lbl = box['label']
        x_centre = (box['xmin'] + box['xmax']) / 2
        y_centre = (box['ymin'] + box['ymax']) / 2
        width = box['xmax'] - box['xmin']
        height = box['ymax'] - box['ymin']

        # convert bbx infor to rel
        x_centre = round(x_centre / img_width, 3)
        y_centre = round(y_centre / img_height, 3)
        width = round(width / img_width, 3)
        height = round(height / img_height, 3)

        annot_txt.append(" ".join([
            str(lbl), str(x_centre), str(y_centre), str(width), str(height)
            ]))

    annot_name = os.path.splitext(img_name)[0] + '.txt'
        

    return annot_name, annot_txt

def make_bound_box():
    # load the annotation set
    data = json.load(open('fedex_test/result.json'))

    # convert annotated images json to dataframe to make slicing easier
    images = pd.DataFrame(data['images'])

    # convert annotated labels json to dataframe to make slicing easier
    annots = pd.DataFrame(data['annotations'])
    annots[['xmin', 'ymin', 'xmax', 'ymax']] = annots.apply(lambda x: expand_bbox_coords(x['bbox']), axis=1, result_type='expand')

    # convert annotated images json to dataframe to make slicing easier
    labels = pd.DataFrame(data['categories'])

    # print a test annotation
    img_id = 0

    # take the entry for the relevant image id
    test_img = images[images['id']==img_id]

    # load image
    path = f"fedex_test/images/{os.path.basename(test_img['file_name'][0])}"
    image = cv2.imread(path)

    # ensure we are using the correct colour spectrum when displaying
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # overlay relevant bounding boxes
    relevant_annots = annots[annots['image_id']==img_id]
    for index, tag in relevant_annots.iterrows():
        # display bbox
        cv2.rectangle(
                image, 
                (int(tag.xmin), int(tag.ymin)), (int(tag.xmax), int(tag.ymax)),
                (255, 0, 0), 2
            )
        
        # display text label
        text = obtain_bbox_label(labels, tag)
        cv2.putText(
            image, text, (int(tag.xmin), int(tag.ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
            (255,0,0), 2
            )

    # display
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def convert_coco2yolo():
    # define output dir (creating it if doesnt exist)
    output_folder = OUPUT_FOLDER_COCO2YOLO 
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # convert COCO to Yolo format
    for _, image in images.iterrows():

        # extract COCO annotations to Yolo format
        image_dict = coco_to_dict(image, annots)
        file_name, file_txt = dict_to_yolo(image_dict)

        # save the file
        with open(os.path.join(output_folder, file_name), 'w') as f:
            for entry in file_txt:
                f.write(f"{entry}\n")

def split_dataset():
    base_path = 'data_open_logo/'

    image_list = [os.path.join(base_path, 'images', x) for x in os.listdir(os.path.join(base_path, 'images')) if not x.startswith('.DS_Store')]
    annot_list = [os.path.join(base_path, 'labels', x) for x in os.listdir(os.path.join(base_path, 'labels')) if not x.startswith('.DS_Store')]

    # to be sure the files are in the same order we sort
    image_list.sort()
    annot_list.sort()
    print("image_list", image_list)
    print("\n")
    print("annot_list", annot_list)
    print(len(image_list))
    print(len(annot_list))

    # obtain the train and test 
    img_train, img_test, annot_train, annot_test = train_test_split(image_list, annot_list, test_size = 0.2, random_state = 1)
    img_val, img_test, annot_val, annot_test = train_test_split(img_test, annot_test, test_size = 0.5, random_state = 1)

    # copy files to the relevant folders
    copy_files(img_train, annot_train, 'train', 'data/logo_detection')
    copy_files(img_val, annot_val, 'val', 'data/logo_detection')
    copy_files(img_test, annot_test, 'test', 'data/logo_detection')

def test_model():
    # test model
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/runs/train/yolo_logo_detection23/weights/best.pt')

    # select an example image from our test images
    img_path = 'ford.png'

    # run inference
    results = model(img_path)

    print(results.pandas().xyxy[0])

def main():
    # test bounding box
    # make_bound_box()

    # convert coco to yolo
    # convert_coco2yolo()

    # split dataset
    # split_dataset()

    # test model
    # test_model()

    print("")


if __name__ == "__main__":
    main()