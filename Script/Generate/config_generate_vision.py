VISION_TASKS = {
    'image_classification': ['objects', 'scenes', 'actions', 'emotions'],
    'object_detection': ['people', 'vehicles', 'animals', 'objects'],
    'image_segmentation': ['semantic', 'instance', 'panoptic'],
    'image_to_text': ['captions', 'descriptions', 'explanations'],
    'text_to_image': ['scenes', 'objects', 'concepts', 'styles'],
    'image_to_image': ['style_transfer', 'enhancement', 'restoration', 'manipulation'],
    'image_feature_extraction': ['textures', 'shapes', 'colors', 'patterns'],
    'depth_estimation': ['indoor', 'outdoor', 'objects', 'scenes'],
    'keypoint_detection': ['poses', 'faces', 'hands', 'landmarks'],
    'mask_generation': ['objects', 'background', 'instances', 'parts'],
    'video_classification': ['actions', 'events', 'emotions', 'scenes'],
    'zero_shot_classification': ['novel_objects', 'unseen_categories', 'transfer']
}

VISION_TOPICS = {
    'objects': ['furniture', 'vehicles', 'electronics', 'tools'],
    'scenes': ['indoor', 'outdoor', 'urban', 'nature'],
    'actions': ['walking', 'running', 'sitting', 'eating'],
    'emotions': ['happy', 'sad', 'angry', 'surprised'],
    'people': ['individuals', 'groups', 'activities', 'expressions'],
    'vehicles': ['cars', 'bikes', 'boats', 'aircraft'],
    'animals': ['pets', 'wildlife', 'birds', 'marine'],
    'semantic': ['background', 'foreground', 'parts', 'regions'],
    'instance': ['objects', 'entities', 'elements', 'components'],
    'panoptic': ['things', 'stuff', 'scene', 'context'],
    'captions': ['descriptions', 'titles', 'summaries', 'narratives'],
    'descriptions': ['detailed', 'simple', 'technical', 'creative'],
    'explanations': ['analytical', 'instructional', 'comparative', 'interpretive'],
    'scenes': ['landscapes', 'cityscapes', 'interiors', 'events'],
    'concepts': ['abstract', 'concrete', 'symbolic', 'metaphorical'],
    'styles': ['artistic', 'realistic', 'minimalist', 'vintage'],
    'textures': ['rough', 'smooth', 'patterned', 'regular'],
    'shapes': ['geometric', 'organic', 'regular', 'irregular'],
    'colors': ['primary', 'secondary', 'warm', 'cool'],
    'patterns': ['repeated', 'random', 'natural', 'artificial'],
    'indoor': ['rooms', 'furniture', 'objects', 'lighting'],
    'outdoor': ['nature', 'buildings', 'streets', 'landmarks'],
    'poses': ['standing', 'sitting', 'walking', 'running'],
    'faces': ['expressions', 'features', 'angles', 'demographics'],
    'hands': ['gestures', 'actions', 'signs', 'interactions'],
    'landmarks': ['natural', 'architectural', 'historical', 'cultural'],
    'actions': ['physical', 'social', 'work', 'leisure'],
    'events': ['social', 'sports', 'cultural', 'natural'],
    'novel_objects': ['unseen', 'rare', 'unique', 'unusual'],
    'unseen_categories': ['new_classes', 'variations', 'combinations']
}

OUTPUT_DIR = 'Script/Generate/VISION_OUTPUT'
MAX_RETRIES = 3
RETRY_DELAY = 2
NUM_SAMPLES_PER_TASK = 100
DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'
DEEPSEEK_MODEL = 'deepseek-chat'

# Image properties
IMAGE_SIZE = (224, 224)  # Standard size for most vision models
IMAGE_CHANNELS = 3       # RGB
IMAGE_FORMAT = 'JPEG'    # Standard format for images
IMAGE_QUALITY = 95       # JPEG quality (0-100)

# Annotation formats
ANNOTATION_FORMATS = {
    'image_classification': 'class_label',
    'object_detection': 'bounding_boxes',
    'image_segmentation': 'pixel_masks',
    'image_to_text': 'captions',
    'text_to_image': 'prompts',
    'image_to_image': 'image_pairs',
    'image_feature_extraction': 'feature_vectors',
    'depth_estimation': 'depth_maps',
    'keypoint_detection': 'keypoints',
    'mask_generation': 'binary_masks',
    'video_classification': 'video_labels',
    'zero_shot_classification': 'class_mappings'
}