from ultralytics import YOLO
from PIL import Image, ImageDraw
from collections import Counter
import matplotlib.pyplot as plt
import os
import torch
from collections import defaultdict

# model=YOLO("yolov8n.pt")

# results=model.train(data="coco128.yaml",epochs=100)

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Provide the path to the test image or directory containing test images
test_images_path = "dataset_hate/test/images/0_no_captions"

# Perform object detection on the test images
results = model(test_images_path)

# # Create a directory to save the results if it doesn't exist
# save_dir = "results"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # Iterate through the results and save each image
# for idx, r in enumerate(results):
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    
#     # Save the image with a unique filename
#     save_path = os.path.join(save_dir, f"result_{idx+1}.jpg")
#     im.save(save_path)  # save image

# print("Results saved successfully.")

class_confidences = defaultdict(float)
class_counts = defaultdict(int)


# Initialize a counter to store the frequency of each class label
class_counter = Counter()

# for i in range(0, len(results)):
#     for box in results[i].boxes:
#         class_id=int(box.cls)
#         class_label=results[i].names[class_id]
#         class_counter.update([class_label])
#         confidence = float(box.conf)
#         class_confidences[class_label] += confidence
#         class_counts[class_label] += 1

# labels, frequencies = zip(*class_counter.items())


# # Calculate the average confidence for each class label
# average_confidences = {label: total_confidence / class_counts[label] for label, total_confidence in class_confidences.items()}
# overall_average_confidence = sum(class_confidences.values()) / sum(class_counts.values()) if sum(class_counts.values()) > 0 else 0
# frequency_distribution = dict(class_counter)
# print("Frequency distribution of all class labels:")
# print(frequency_distribution)

# labels, frequencies = zip(*class_counter.items())
# # Plot a bar graph with average confidences displayed on each bar
# plt.figure(figsize=(10, 6))
# bars = plt.bar(labels, frequencies)
# plt.xlabel('Class Label')
# plt.ylabel('Frequency')
# plt.title('Object Detection Results')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# # Display average confidences inside each bar
# for i, label in enumerate(labels):
#     avg_confidence = round(average_confidences.get(label, 0), 2)
#     plt.text(i, bars[i].get_height() - 0.1, f' {avg_confidence}', ha='center', va='bottom')  # Adjust y-coordinate
#     plt.text(0.02, 0.95, f'Overall Avg Conf: {round(overall_average_confidence, 2)}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
# plt.show()
            
# Iterate through the results and count the occurrences of each class label
for i in range(len(results)):
    for box in results[i].boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        if confidence > 0.5:  # Consider detections with confidence > 0.5
            class_label = results[i].names[class_id]
            class_counter.update([class_label])
            confidence = float(box.conf)
            class_confidences[class_label] += confidence
            class_counts[class_label] += 1

# Extract the class labels and their frequencies
labels, frequencies = zip(*class_counter.items())

# Calculate the average confidence for each class label
average_confidences = {label: total_confidence / class_counts[label] for label, total_confidence in class_confidences.items()}

overall_average_confidence = sum(class_confidences.values()) / sum(class_counts.values()) if sum(class_counts.values()) > 0 else 0

# Create a frequency distribution dictionary of all class labels
frequency_distribution = dict(class_counter)
print("Frequency distribution of class labels predicted with more than 0.5 certainty:")
print(frequency_distribution)

# Plot a bar graph
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, frequencies)
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.title('Object Detection Results (Confidence > 0.5)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Display average confidences inside each bar
for i, label in enumerate(labels):
    avg_confidence = round(average_confidences.get(label, 0), 2)
    plt.text(i, bars[i].get_height() - 0.1, f' {avg_confidence}', ha='center', va='bottom')  # Adjust y-coordinate


plt.text(0.02, 0.95, f'Overall Avg Conf: {round(overall_average_confidence, 2)}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.show()
            

#labels predicted with more than 0.5 certainty means more reliable 


