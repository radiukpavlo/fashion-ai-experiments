import tensorflow as tf
import cv2
import numpy as np
import json
import posenet
import argparse

# Set the model path
model_path = "./pose_tf-lite_models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"


def draw_skeleton(keypoints, img_file):

    img = cv2.imread(img_file)

    for i in range(len(keypoints)//2):
        start = tuple(keypoints[i*2])
        end = tuple(keypoints[i*2 + 1])
        cv2.line(img, start, end, (0,255,0), 2)

    return img


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default="../images_val/val_female_01.png", help="path to input image")
    args = vars(parser.parse_args())

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get the input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    img_file = args["image"]
    img = cv2.imread(img_file)
    img = tf.image.resize_with_pad(img, 257, 257)  # input shape for model
    input_data = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get keypoints
    heatmap = interpreter.get_tensor(output_details[0]['index'])
    offsets = interpreter.get_tensor(output_details[1]['index'])

    # Handle batch dimension
    heatmap = np.squeeze(heatmap)
    offsets = np.squeeze(offsets)

    num_people = np.sum(np.max(heatmap, axis=2))

    keypoint_coords = []

    for i in range(heatmap.shape[1]):
        heatmap_y = np.sum(heatmap[:, i, :], axis=0)
        coords = np.argmax(heatmap_y)

        if num_people > 0:
            offsets_y = offsets[i, coords]
        else:
            offsets_y = 0
        keypoint_coords.append([coords+offsets_y, i])

    keypoint_coords = np.array(keypoint_coords)

    # Draw skeleton
    output_img = draw_skeleton(keypoint_coords, img_file)

    # Save output
    cv2.imwrite('pose_tf-lite_images/output.png', output_img)

    # Convert to json serializable format
    keypoint_coords = keypoint_coords.tolist()

    # Save json
    with open('keypoints.json', 'w') as json_file:
        json.dump(keypoint_coords, json_file)

    print("Pose estimation complete!")


if __name__ == "__main__":
    main()
