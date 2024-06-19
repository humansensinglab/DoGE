import numpy as np
from PIL import Image

ade20k_palette = np.asarray([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])

gta_classes = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static',
           'dynamic', 'ground', 'road', 'sidewalk', 'parking',
           'rail track', 'building', 'wall', 'fence', 'guard rail',
           'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light',
           'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
           'rider', 'car', 'truck', 'bus', 'caravan', 'trailer',
           'train', 'motorcycle', 'bicycle', 'license plate']
gta_class_map = {i: name for i, name in enumerate(gta_classes)}

gta_to_ade20k_map = {
    'unlabeled': 0, 
    'ego vehicle': 0, 
    'rectification border': 0, 
    'out of roi': 0, 
    'static': 0,
    'dynamic': 0, 
    'ground': 14, 
    'road': 7, 
    'sidewalk': 12, 
    'parking': 14,
    'rail track': 39, 
    'building': 2, 
    'wall': 1, 
    'fence': 33, 
    'guard rail': 33,
    'bridge': 62, 
    'tunnel': 69, 
    'pole': 94, 
    'polegroup': 94, 
    'traffic light': 137,
    'traffic sign': 44, 
    'vegetation': 18,  # 10
    'terrain': 69,  # 14
    'sky': 3, 
    'person': 13,
    'rider': 13, 
    'car': 21, 
    'truck': 84, 
    'bus': 81, 
    'caravan': 103, 
    'trailer': 84,
    'train': 0,  # 81
    'motorcycle': 117, 
    'bicycle': 128, 
    'license plate': 0  
}

synthia_classes = ["void", "sky", "building", "road", "sidewalk", "fence", "vegetation", "pole",
                    "car", "traffic sign", "pedestrian", "bicycle", "motorcycle", "parking-slot",
                    "road-work", "traffic light", "terrain", "rider", "truck", "bus", "train",
                    "wall", "lanemarking"]
synthia_class_map = {i: name for i, name in enumerate(synthia_classes)}

synthia_to_ade20k_map = {
    'void': 0, 
    'sky': 3, 
    'building': 2, 
    'road': 7, 
    'sidewalk': 12, 
    'fence': 33, 
    'vegetation': 18, 
    'pole': 94,
    'car': 21, 
    'traffic sign': 44, 
    'pedestrian': 13, 
    'bicycle': 128, 
    'motorcycle': 117, 
    'parking-slot': 14, 
    'road-work': 0,
    'traffic light': 137, 
    'terrain': 69, 
    'rider': 13, 
    'truck': 84, 
    'bus': 81, 
    'train': 0, 
    'wall': 1, 
    'lanemarking': 0
}


def gta_to_ade20k(gta_label):
    if not isinstance(gta_label, np.ndarray):
        gta_label = np.array(gta_label)
    ade_label = np.zeros((*gta_label.shape, 3))
    for id, label in enumerate(gta_classes):
        ade_label[gta_label == id] = ade20k_palette[gta_to_ade20k_map[label]]
    ade_label = Image.fromarray(ade_label.astype(np.uint8))
    return ade_label

def synthia_to_ade20k(synthia_label):
    if not isinstance(synthia_label, np.ndarray):
        synthia_label = np.array(synthia_label)
    ade_label = np.zeros((*synthia_label.shape, 3))
    for id, label in enumerate(synthia_classes):
        ade_label[synthia_label == id] = ade20k_palette[synthia_to_ade20k_map[label]]
    ade_label = Image.fromarray(ade_label.astype(np.uint8))
    return ade_label