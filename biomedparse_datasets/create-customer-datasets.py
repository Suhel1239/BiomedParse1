# import glob
# from tqdm import tqdm
# import pandas as pd
# import os
# from create_annotations import *


# # provide the path to the dataset. There should be train, train_mask, test, test_mask under this folder
# targetpath = '/kaggle/input/newr-1/Test_Dataset_NewResearch'


# image_size = 640


# ### Load Biomed Label Base
# # provide path to predefined label base
# with open('label_base.json', 'r') as f:
#     label_base = json.load(f)
    
    
    
# # get parent class for the names
# parent_class = {}
# for i in label_base:
#     subnames = [label_base[i]['name']] + label_base[i].get('child', [])
#     for label in subnames:
#         parent_class[label] = int(i)
    
# # Label ids of the dataset
# category_ids = {label_base[i]['name']: int(i) for i in label_base if 'name' in label_base[i]}



# # Get "images" and "annotations" info 
# def images_annotations_info(maskpath):
    
#     imagepath = maskpath.replace('_mask', '')
#     # This id will be automatically increased as we go
#     annotation_id = 0
    
#     sent_id = 0
#     ref_id = 0
    
#     annotations = []
#     images = []
#     image_to_id = {}
#     n_total = len(glob.glob(maskpath + "*.png"))
#     n_errors = 0
    
#     def extra_annotation(ann, file_name, target):
#         nonlocal sent_id, ref_id
#         ann['file_name'] = file_name
#         ann['split'] = keyword
        
#         ### modality
#         mod = file_name.split('.')[0].split('_')[-2]
#         ### site
#         site = file_name.split('.')[0].split('_')[-1]
        
#         task = {'target': target, 'modality': mod, 'site': site}
#         if 'T1' in mod or 'T2' in mod or 'FLAIR' in mod or 'ADC' in mod:
#             task['modality'] = 'MRI'
#             if 'MRI' not in mod:
#                 task['sequence'] = mod
#             else:
#                 task['sequence'] = mod[4:]
            
#         prompts = [f'{target} in {site} {mod}']
        
#         ann['sentences'] = []
#         for p in prompts:
#             ann['sentences'].append({'raw': p, 'sent': p, 'sent_id': sent_id})
#             sent_id += 1
#         ann['sent_ids'] = [s['sent_id'] for s in ann['sentences']]
        
#         ann['ann_id'] = ann['id']
#         ann['ref_id'] = ref_id
#         ref_id += 1
        
#         return ann
    
#     for mask_image in tqdm(glob.glob(maskpath + "*.png")):
#         # The mask image is *.png but the original image is *.jpg.
#         # We make a reference to the original file in the COCO JSON file
#         filename_parsed = os.path.basename(mask_image).split("_")
#         target_name = filename_parsed[-1].split(".")[0].replace("+", " ")
        
#         original_file_name = "_".join(filename_parsed[:-1]) + ".png"
        
#         if original_file_name not in os.listdir(imagepath):
#             print("Original file not found: {}".format(original_file_name))
#             n_errors += 1
#             continue
        
#         if original_file_name not in image_to_id:
#             image_to_id[original_file_name] = len(image_to_id)

#             # "images" info 
#             image_id = image_to_id[original_file_name]
#             image = create_image_annotation(original_file_name, image_size, image_size, image_id)
#             images.append(image)
            
        
#         annotation = {
#             "mask_file": os.path.basename(mask_image),
#             "iscrowd": 0,
#             "image_id": image_to_id[original_file_name],
#             "category_id": parent_class[target_name],
#             "id": annotation_id,
#         }

#         annotation = extra_annotation(annotation, original_file_name, target_name)
                
#         annotations.append(annotation)
#         annotation_id += 1
            
#     #print(f"Number of errors in conversion: {n_errors}/{n_total}")
#     return images, annotations, annotation_id




# if __name__ == "__main__":
#     # Get the standard COCO JSON format
#     coco_format = get_coco_json_format()

#     for keyword in ['train', 'test']:
#         mask_path = os.path.join(targetpath, "{}_mask/".format(keyword))
        
#         # Create category section
#         coco_format["categories"] = create_category_annotation(category_ids)
    
#         # Create images and annotations sections
#         coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

#         # post-process file
#         images_with_ann = set()
#         for ann in coco_format['annotations']:
#             images_with_ann.add(ann['file_name'])
#         for im in coco_format['images']:
#             if im["file_name"] not in images_with_ann:
#                 coco_format['images'].remove(im)

#         # with open(os.path.join(targetpath, "{}.json".format(keyword)),"w") as outfile:
#         #     json.dump(coco_format, outfile)
        
#         output_dir = "/kaggle/working/biomedparse_output"
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Write the JSON file there instead of targetpath
#         output_file = os.path.join(output_dir, f"{keyword}.json")
#         with open(output_file, "w") as outfile:
#             json.dump(coco_format, outfile)
        
#         print("Created %d annotations for %d images in folder: %s" % (annotation_cnt, len(coco_format['images']), mask_path))

#################################################

import os
import glob
import json
from tqdm import tqdm

# === Config ===
targetpath = '/kaggle/input/newr-1/Test_Dataset_NewResearch'
image_size = 640

# Load category mapping from label_base.json
with open('label_base.json', 'r') as f:
    label_base = json.load(f)

# Build parent class mapping
parent_class = {}
for i in label_base:
    subnames = [label_base[i]['name']] + label_base[i].get('child', [])
    for label in subnames:
        parent_class[label] = int(i)

# Dataset category IDs
category_ids = {label_base[i]['name']: int(i) for i in label_base if 'name' in label_base[i]}

# === Helper Functions ===
def create_category_annotation(category_dict):
    return [{"supercategory": key, "id": value, "name": key} for key, value in category_dict.items()]

def create_image_annotation(file_name, width, height, image_id):
    return {"file_name": file_name, "height": height, "width": width, "id": image_id}

def get_coco_json_format():
    return {"info": {}, "licenses": [], "images": [], "categories": [], "annotations": []}

def images_annotations_info(maskpath, keyword):
    imagepath = maskpath.replace('_mask', '')
    annotation_id = 0
    sent_id = 0
    ref_id = 0
    annotations = []
    images = []
    image_to_id = {}
    n_errors = 0

    for mask_image in tqdm(glob.glob(maskpath + "*.png")):
        filename_parsed = os.path.basename(mask_image).split("_")
        description_raw = filename_parsed[-1].split(".")[0].replace("+", " ")

        category_main = description_raw.split()[0]  # First word only for category lookup
        if category_main not in parent_class:
            print(f"Warning: category '{category_main}' not found in label_base.json")
            continue

        original_file_name = "_".join(filename_parsed[:-1]) + ".png"
        if original_file_name not in os.listdir(imagepath):
            print("Original file not found:", original_file_name)
            n_errors += 1
            continue

        if original_file_name not in image_to_id:
            image_to_id[original_file_name] = len(image_to_id)
            image_id = image_to_id[original_file_name]
            images.append(create_image_annotation(original_file_name, image_size, image_size, image_id))

        bbox = [150, 200, 75, 67]  # <-- Replace this with your actual bbox loading logic
        area = bbox[2] * bbox[3]

        annotation = {
            "mask_file": os.path.basename(mask_image),
            "area": area,
            "iscrowd": 0,
            "image_id": image_to_id[original_file_name],
            "bbox": bbox,
            "category_id": parent_class[category_main],
            "id": annotation_id,
            "slice_ratio": 0.5,
            "file_name": original_file_name,
            "split": keyword,
            "sentences": [
                {"raw": description_raw, "sent": description_raw, "sent_id": sent_id},
                {"raw": " ".join(description_raw.split()[1:]), "sent": " ".join(description_raw.split()[1:]), "sent_id": sent_id + 1}
            ]
        }
        sent_id += 2
        annotation_id += 1
        annotations.append(annotation)

    return images, annotations, annotation_id, n_errors

# === Main Conversion ===
if __name__ == "__main__":
    for keyword in ['train', 'test']:
        coco_format = get_coco_json_format()
        mask_path = os.path.join(targetpath, f"{keyword}_mask/")

        coco_format["categories"] = create_category_annotation(category_ids)
        coco_format["images"], coco_format["annotations"], ann_count, errors = images_annotations_info(mask_path, keyword)

        output_dir = "/kaggle/working/biomedparse_detection_output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{keyword}.json")

        with open(output_file, "w") as outfile:
            json.dump(coco_format, outfile, indent=4)

        print(f"[{keyword}] Created {ann_count} annotations with {errors} missing images.")

