import json
import os
# from inferencer_demo import *  # assuming inferencer_demo.py exports relevant functionalities


def is_standing_tall(pose):
    # Extract keypoint data
    keypoints = pose['keypoints']

    # Extract the coordinates of relevant keypoints (assuming COCO keypoints structure)
    right_ankle = keypoints[10]
    left_ankle = keypoints[13]
    nose = keypoints[0]
    neck = keypoints[1]

    # Check if both feet are on the same horizontal plane (with a tolerance)
    if abs(right_ankle[1] - left_ankle[1]) > 30:  # Adjust the tolerance value as needed
        return False

    # Check if the head (nose) is above both feet
    if nose[1] > right_ankle[1] or nose[1] > left_ankle[1]:
        return False

    # Check if the spine is vertically aligned (by checking if the neck is vertically between the nose and feet)
    if not nose[1] < neck[1] < (right_ankle[1] + left_ankle[1]) / 2:
        return False

    return True


def main():
    # Path to the folder containing JSON files
    folder_path = './mmpose_predictions/'

    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Filter to get only JSON files
    json_files = [f for f in all_files if f.endswith('.json')]

    for json_file in json_files:
        # Construct full file path
        file_path = os.path.join(folder_path, json_file)

        # Load the JSON data from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            pose_data_list = json.load(file)

            # Assume there might be multiple poses in a file, loop through them
            for pose in pose_data_list:
                result = is_standing_tall(pose)
                print(f"File {json_file}, Pose {pose_data_list.index(pose) + 1}: Standing tall? {'Yes' if result else 'No'}")


if __name__ == '__main__':
    main()
