#final code 
import cv2
import numpy as np
import datetime
import csv
import os
import uuid
import time
import pandas as pd
from ultralytics import YOLO
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# Load face detection model
modelFile = "/home/gokul/Downloads/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "/home/gokul/Downloads/deploy.prototxt"
face_net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load the YOLOv8n model
yolo_model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

known_faces = []
face_ids = []
face_first_seen_times = {}
face_last_seen_times = {}
face_screenshot_taken = {}
face_customer_status = {}
face_walkin_start_times = {}
face_walkin_durations = {}

screenshot_folder = '/home/gokul/Documents/outputs'
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

# CSV file setup
csv_file = 'face_logs.csv'
csv_header = ["Timestamp", "Face_ID", "Status", "Type", "Walkin_Time", "Total_Persons_Visited", "Seconds"]
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

def get_face_id(face_box, timestamp):
    global known_faces, face_ids
    (x, y, w, h) = face_box
    center_x = x + w / 2
    center_y = y + h / 2
    for idx, (fx, fy, fw, fh) in enumerate(known_faces):
        if (fx <= center_x <= fx + fw) and (fy <= center_y <= fy + fh):
            known_faces[idx] = (x, y, w, h)
            return face_ids[idx], 'old user visit'
    new_id = str(uuid.uuid4())
    known_faces.append((x, y, w, h))
    face_ids.append(new_id)
    face_first_seen_times[new_id] = timestamp
    face_last_seen_times[new_id] = timestamp
    face_screenshot_taken[new_id] = False
    face_walkin_durations[new_id] = datetime.timedelta()
    return new_id, 'new user visit'

close_threshold = 150
current_proximity_status = {}
prev_proximity_status = {}
total_persons_visited = 0

previous_walkin_times = {}  # Dictionary to store previous walkin times for each face ID


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    current_time = datetime.datetime.now()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Run YOLOv8n object detection
    results = yolo_model(frame, stream=True)

    current_proximity_status.clear()
    person_detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_id, status = get_face_id((startX, startY, endX - startX, endY - startY), current_time)
            face_width = endX - startX
            is_close = face_width >= close_threshold
            current_proximity_status[face_id] = is_close

            # Check if there's a meaningful change or if it's a new detection
            if face_id not in prev_proximity_status or prev_proximity_status[face_id] != is_close:
                walkin_status = "walkin" if is_close else "glancing"
                
                walkin_time_seconds = round((current_time - face_first_seen_times[face_id]).total_seconds())
                # Calculate the difference between the current walkin time and the previous one
                if face_id in previous_walkin_times:
                    walkin_time_difference = walkin_time_seconds - previous_walkin_times[face_id]
                else:
                    walkin_time_difference = 0
                previous_walkin_times[face_id] = walkin_time_seconds

                with open(csv_file, 'a', newline='') as file:  # Open the file in append mode
                    writer = csv.writer(file)
                    # Append the walkin time difference to the first row
                    if len(previous_walkin_times) == 1:
                        writer.writerow([current_time.strftime("%Y-%m-%d %H:%M:%S"), face_id, status, walkin_status,
                                         walkin_time_seconds, total_persons_visited, walkin_time_difference])
                    else:
                        writer.writerow([current_time.strftime("%Y-%m-%d %H:%M:%S"), face_id, status, walkin_status,
                                         walkin_time_seconds, total_persons_visited, ""])

                total_persons_visited = len(set(face_ids))

            # Check if the face has been seen for at least 2 seconds before taking a screenshot
            if not face_screenshot_taken[face_id] and (current_time - face_first_seen_times[face_id]).total_seconds() >= 2 and is_close:
                screenshot_path = f'{screenshot_folder}/{face_id}_{current_time.strftime("%Y%m%d%H%M%S")}.jpg'
                cv2.imwrite(screenshot_path, frame)
                face_screenshot_taken[face_id] = True

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, face_id, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Check for person detections using YOLOv8n
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  # 'person' class has index 0
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)

                person_detected = True
                face_id_found = False
                for face_box in known_faces:
                    fx, fy, fw, fh = face_box
                    if fx <= x1 <= fx + fw and fy <= y1 <= fy + fh:
                        face_id_found = True
                        break

                if not face_id_found:
                    # Person detected but no face ID generated, add a row to the CSV with 'Passed By'
                    with open(csv_file, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([current_time.strftime("%Y-%m-%d %H:%M:%S"), '', '', 'Passed By', '', total_persons_visited, ''])

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    prev_proximity_status = current_proximity_status.copy()
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

# Update the CSV file with the total persons visited
with open(csv_file, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["", "", "", "", "", total_persons_visited])

# Load the CSV into a DataFrame
df = pd.read_csv('face_logs.csv')

# Calculate the differences between consecutive Walkin_Time values
df['Walkin_Time'] = df.groupby('Face_ID')['Walkin_Time'].diff().fillna(0)

# Correct the Walkin_Time for the last entry
df.loc[df.groupby('Face_ID').tail(1).index, 'Walkin_Time'] = 0

# Update 'Type' column based on 'Walkin_Time' threshold
df.loc[(df['Type'] == 'glancing') & (df['Walkin_Time'] > 2), 'Type'] = 'viewing'

# Drop 'Seconds' column
df = df.drop(columns=["Seconds"])

# Save updated DataFrame back to CSV
df.to_csv('face_logs.csv', index=False)

print("Updated face_logs.csv with corrected Walkin_Time and Type.")

# Function to classify age using ViT model
# Function to classify age using ViT model
def classify_age(image):
    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    transform = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
    inputs = transform(images=image, return_tensors='pt')
    output = model(**inputs)
    proba = output.logits.softmax(1)
    preds = proba.argmax(1)
    predicted_classes = preds.detach().numpy()
    predicted_probabilities = proba.max(1).values.detach().numpy()

    labels = [model.config.id2label[label] for label in predicted_classes]
    probabilities = predicted_probabilities

    return labels[0], probabilities[0]

# Load the gender classification model
gender_processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")

# Folder path where screenshots are stored
folder_path = "/home/gokul/Documents/outputs"
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
csv_file_path = "face_logs_gender.csv"

# Write the predicted age range and gender to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Face_ID', 'Predicted Age Range', 'Predicted Gender'])
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        base_name = image_file.split('_')[0]
        base_name = os.path.splitext(base_name)[0]
        predicted_age_label, predicted_age_probability = classify_age(image)
        predicted_age_range = predicted_age_label
        gender_inputs = gender_processor(images=image, return_tensors="pt")
        gender_outputs = gender_model(**gender_inputs)
        predicted_gender = torch.argmax(gender_outputs.logits).item()
        writer.writerow([base_name, predicted_age_range, 'Male' if predicted_gender == 1 else 'Female'])
        print(f"Image: {base_name}, Predicted Age Range: {predicted_age_range}, Predicted Gender: {'Male' if predicted_gender == 1 else 'Female'}")

print("Results saved to:", csv_file_path)


face_df = pd.read_csv('face_logs.csv')
gender_df = pd.read_csv('face_logs_gender.csv')

# Merge DataFrames based on 'Face_ID' with an 1outer join
merged_df = pd.merge(face_df, gender_df, on='Face_ID', how='outer')

# Fill NaN values in 'Predicted Age Range' and 'Predicted Gender' columns with empty strings
merged_df = merged_df.fillna({'Predicted Age Range': '', 'Predicted Gender': ''})


merged_df.to_csv('merged_data.csv', index=False)

print("Merged data saved to: merged_data.csv")
