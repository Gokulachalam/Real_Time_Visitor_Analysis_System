import os
import csv
import pandas as pd
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

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

# Merge DataFrames based on 'Face_ID' with an outer join
merged_df = pd.merge(face_df, gender_df, on='Face_ID', how='outer')

# Fill NaN values in 'Predicted Age Range' and 'Predicted Gender' columns with empty strings
merged_df = merged_df.fillna({'Predicted Age Range': '', 'Predicted Gender': ''})

merged_df.to_csv('merged_data.csv', index=False)

print("Merged data saved to: merged_data.csv")
