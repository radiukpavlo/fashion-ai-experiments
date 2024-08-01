import torch
import cv2
import numpy as np
import math
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode

# Завантажити зображення
image_path = '1.jpg'
image = cv2.imread(image_path)

# Налаштування Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Поріг впевненості для передбачень
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Застосування моделі для передбачення
outputs = predictor(image)

# Отримання передбачень
instances = outputs["instances"].to("cpu")
pred_boxes = instances.pred_boxes if instances.has("pred_boxes") else None
pred_masks = instances.pred_masks if instances.has("pred_masks") else None

# Візуалізація результатів
v = Visualizer(image[:, :, ::-1], scale=1.2, instance_mode=ColorMode.IMAGE_BW)
out = v.draw_instance_predictions(instances)
segmented_image = out.get_image()[:, :, ::-1]

# Відображення сегментованого зображення
cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Визначення центру мотора за допомогою масок
motor_center = None
for mask in pred_masks:
    # Знаходження контурів маски
    contours, _ = cv2.findContours(mask.numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Визначення моментів
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            motor_center = (center_x, center_y)
            break
    if motor_center is not None:
        break

if motor_center is not None:
    # Відображення центру мотора на зображенні
    cv2.circle(image, motor_center, 5, (255, 0, 0), -1)

    # Виділення контурів за допомогою Canny
    edges = cv2.Canny(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY), 50, 150, apertureSize=3)

    # Використання Hough Transform для знаходження ліній
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # Перетворення полярних координат в декартові для малювання ліній
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Малювання ліній на зображенні
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Обчислення кутів лопатей відносно центру мотора
            dx = x2 - motor_center[0]
            dy = y2 - motor_center[1]
            angle = math.degrees(math.atan2(dy, dx))
            angles.append(angle)

    # Відображення зображення з накладеними лініями
    cv2.imshow('Detected Lines and Motor Center', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Виведення кутів
    print("Angles of the blades relative to the horizontal axis:", angles)
else:
    print("Motor center not found.")