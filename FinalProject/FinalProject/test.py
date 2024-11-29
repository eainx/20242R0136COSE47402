from transformers import DetrImageProcessor, DetrForSegmentation
from PIL import Image
import numpy as np

# 모델 및 프로세서 불러오기
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

# 이미지 열기
image = Image.open("example.jpg")
image_width, image_height = image.size

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
# Extract predictions
logits = outputs.logits.softmax(-1)[0, :, :-1]  # Ignore the last "no-object" class
boxes = outputs.pred_boxes[0]  # Bounding boxes

# Map class indices to COCO labels
# The COCO dataset has predefined class labels
COCO_LABELS = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Get the labels and corresponding boxes
threshold = 0.9  # Confidence threshold for filtering predictions
labels = logits.argmax(-1)

# 결과에서 마스크 추출
result = outputs.pred_masks  # 이 변수에는 각 객체에 대한 분할 마스크가 포함되어 있습니다.
target_label = 17  # COCO 데이터셋에서 고양이의 레이블

# 마스크에서 고양이에 해당하는 부분만 추출
cat_masks = result[0, labels == target_label]

# 첫 번째 고양이의 마스크를 이미지로 변환 (예시)
if cat_masks.size(0) > 0:
    cat_mask = cat_masks[0]
    # Boolean 마스크를 이미지로 변환
    mask_image = Image.fromarray(cat_mask.detach().cpu().numpy().astype("uint8") * 255, mode="L")
    mask_image.save("cat_mask.jpg")
else:
    print("No cat detected.")

image = Image.open("example.jpg")
mask_image = Image.open("cat_mask.jpg")  # 이전에 저장된 마스크 이미지

# 원본 이미지 크기로 마스크 확대
mask_resized = mask_image.resize(image.size, Image.NEAREST)

# 마스크를 배열로 변환하고, 원본 이미지와 같은 크기의 투명 배경 이미지 생성
mask_array = np.array(mask_resized)
original = np.array(image)
result_image = np.zeros_like(original)

# 마스크에 해당하는 부분만 원본 이미지에서 추출
result_image[mask_array > 128] = original[mask_array > 128]  # 마스크가 128 이상인 픽셀만 복사

# 결과 이미지를 PNG로 저장 (투명 배경 포함)
result_pil = Image.fromarray(result_image)
result_pil.save("extracted_cat.png")