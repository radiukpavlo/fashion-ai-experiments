import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import cv2
import numpy as np
import json

# Set the model path
model_path = "./pose_tf-lite_models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"


class Pose:
    def __init__(self, id, keypoints):
        self.id = id
        self.keypoints = keypoints


def main():
    # Load the TensorFlow Lite model.
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get the input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get the image path from the user.
    image_path = "../images_val/val_female_01.png"

    # Load the image.
    image = cv2.imread(image_path)

    # Preprocess the image.
    image = cv2.resize(image, (257, 257))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    # Run the pose estimator.
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    poses = interpreter.get_tensor(output_details[0]['index'])

    # Draw the keypoints on the image.
    for pose in poses:
        for key_point in pose.keypoints:
            x, y = key_point.x * image.shape[1], key_point.y * image.shape[0]
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Save the image with superimposed skeleton.
    cv2.imwrite("pose_tf-lite_images/image_with_skeleton.png", image)

    # Save the JSON file with keypoints.
    json_data = {}
    for pose in poses:
        pose_data = {}
        for key_point in pose.keypoints:
            pose_data[key_point.part_id] = {
                "x": key_point.x,
                "y": key_point.y
            }
        json_data[pose.id] = pose_data

    with open("keypoints.json", "w") as json_file:
        json.dump(json_data, json_file)


if __name__ == "__main__":
    main()
