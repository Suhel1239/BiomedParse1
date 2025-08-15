import glob
from tqdm import tqdm
import pandas as pd
import random

from create_annotations import *


# provide the path to the dataset. There should be train, demo_mask, test, test_mask under this folder
targetpath = '/kaggle/working/data'


image_size = 1024


### Load Biomed Label Base
# provide path to predefined label base
with open('label_base1.json', 'r') as f:
    label_base = json.load(f)
    
    
    
# get parent class for the names
parent_class = {}
for i in label_base:
    subnames = [label_base[i]['name']] + label_base[i].get('child', [])
    for label in subnames:
        parent_class[label] = int(i)
    
# Label ids of the dataset
category_ids = {label_base[i]['name']: int(i) for i in label_base if 'name' in label_base[i]}

# 词库定义
lesion_descriptions_en = {
    'yuji': {
        'names': ['yuji', 'LU10'],
        'function': ['highly effective in clearing heat from the lungs and throat', 'relieve sore throat', 'cough', 'asthma', 'fever'],
        'location': ['base of the thumb', 'located on the palm side of the hand', 'midpoint of the first metacarpal bone', 'Lung Meridian']
    },
    'laogong': {
        'names': ['laogong', 'P8'],
        'function': ['clearing heat from the heart', 'calming the mind', 'revitalizing energy flow', 'stress', 'anxiety', 'insomnia', 'heart palpitations', 'excessive sweating', 'digestive discomfort'],
        'location': ['located on the palm of the hand', 'Pericardium Meridian', 'where the tip of the middle finger naturally touches the palm','second and third metacarpal bones']
    },
    'zhongchong': {
        'names': ['zhongchong', 'P9'],
        'function': ['clear heat', 'revive consciousness', 'regulate heart energy', 'stroke', 'fainting', 'high fever', 'heatstroke', 'loss of consciousness', 'heart palpitations'],
        'location': ['tip of the middle finger', 'Pericardium Meridian']
    },
    'shaofu': {
        'names': ['shaofu', 'HT8'],
        'function': ['clearing heat from the Heart', 'calming mind', 'emotional stress', 'anxiety', 'heart palpitations', 'excessive sweating', 'mouth ulcers'],
        'location': ['palm of the hand', 'Heart Meridian', 'lies where the tip of the little finger naturally touches the palm', 'fourth and fifth metacarpal bones']
    }
}

def get_bbox_and_area(mask_path):
    """Calculate bbox and area from a binary mask."""
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0], 0

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    area = int(np.sum(mask > 0))
    return bbox, area

def generate_description(lesion, site, mod):
    description_list = []

    desc = lesion_descriptions_en[lesion]
    name = random.choice(desc['names'])
    function = random.choice(desc['function'])
    location = random.choice(desc['location'])

    # 随机选择生成 1 到 3 句描述
    num_sentences = random.randint(1, 3)
    for _ in range(num_sentences):
        template = random.choice([
        f"{name} is observed, used for {function}, located at {location}.",
        f"{name} appears in the image, used for {function}, mainly distributed around {location}.",
        f"{name} can be seen in the image, usually used for {function}, commonly found at {location}.",
    ])
        description_list.append(template)

    return description_list


# Get "images" and "annotations" info 
def images_annotations_info(maskpath):
    
    imagepath = maskpath.replace('_mask', '')
    # This id will be automatically increased as we go
    annotation_id = 0
    
    sent_id = 0
    ref_id = 0
    
    annotations = []
    images = []
    image_to_id = {}
    n_total = len(glob.glob(maskpath + "*.png"))
    n_errors = 0
    
    def extra_annotation(ann, file_name, target):
        nonlocal sent_id, ref_id
        ann['file_name'] = file_name
        ann['split'] = keyword
        bbox, area = get_bbox_and_area(mask_path)
        ### modality
        mod = file_name.split('.')[0].split('_')[-2]
        ### site
        site = file_name.split('.')[0].split('_')[-1]
        
        task = {'target': target, 'modality': mod, 'site': site}
        if 'T1' in mod or 'T2' in mod or 'FLAIR' in mod or 'ADC' in mod:
            task['modality'] = 'MRI'
            if 'MRI' not in mod:
                task['sequence'] = mod
            else:
                task['sequence'] = mod[4:]
            
        # prompts = [f'{target} in {site} {mod}']  # 这里要改一下，希望句子能够描述病灶特点，并且能够随机生成描述语句。
        prompts = generate_description(target, site, mod)

        ann['sentences'] = []
        for p in prompts:
            ann['sentences'].append({'raw': p, 'sent': p, 'sent_id': sent_id})
            sent_id += 1
        ann['sent_ids'] = [s['sent_id'] for s in ann['sentences']]
        
        ann['ann_id'] = ann['id']
        ann['ref_id'] = ref_id
        ref_id += 1

        # Restore COCO-required keys
        if bbox is not None:
            ann["bbox"] = bbox
        if area is not None:
            ann["area"] = area
        
        return ann
    
    for mask_image in tqdm(glob.glob(maskpath + "*.png")):
        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file
        filename_parsed = os.path.basename(mask_image).split("_")
        target_name = filename_parsed[-1].split(".")[0].replace("+", " ")
        if target_name == 'MA':
            target_name = 'microaneurysm'
        if target_name == 'HE':
            target_name = 'hemorrhage'
        if target_name == 'EX':
            target_name = 'hard exudate'
        if target_name == 'SE':   # 注意之后写一个将名字随机设置为“soft exudate”或“ cotton wool spot”
            target_name = 'soft exudate'

        original_file_name = "_".join(filename_parsed[:-1]) + ".png"
        
        if original_file_name not in os.listdir(imagepath):
            print("Original file not found: {}".format(original_file_name))
            n_errors += 1
            continue
        
        if original_file_name not in image_to_id:
            image_to_id[original_file_name] = len(image_to_id)

            # "images" info 
            image_id = image_to_id[original_file_name]
            image = create_image_annotation(original_file_name, image_size, image_size, image_id)
            images.append(image)
            
        
        annotation = {
            "mask_file": os.path.basename(mask_image),
            "iscrowd": 0,
            "image_id": image_to_id[original_file_name],
            "category_id": parent_class[target_name],
            "id": annotation_id,
        }

        annotation = extra_annotation(annotation, original_file_name, target_name)
                
        annotations.append(annotation)
        annotation_id += 1
            
    #print(f"Number of errors in conversion: {n_errors}/{n_total}")
    return images, annotations, annotation_id




if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()

    for keyword in ['train', 'test']:
        mask_path = os.path.join(targetpath, "{}_mask/".format(keyword))
        
        # Create category section
        coco_format["categories"] = create_category_annotation(category_ids)
    
        # Create images and annotations sections
        coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

        # post-process file
        images_with_ann = set()
        for ann in coco_format['annotations']:
            images_with_ann.add(ann['file_name'])
        for im in coco_format['images']:
            if im["file_name"] not in images_with_ann:
                coco_format['images'].remove(im)

        with open(os.path.join(targetpath, "{}.json".format(keyword)),"w") as outfile:
            json.dump(coco_format, outfile)
        
        print("Created %d annotations for %d images in folder: %s" % (annotation_cnt, len(coco_format['images']), mask_path))
