# Real-Time Visitor Analysis System

The Real-Time Visitor Analysis System is a powerful tool that combines face detection, object detection, and image classification techniques to analyze and track visitors in real-time. It leverages OpenCV, YOLOv8n, and Transformer models to provide valuable insights into visitor behavior and demographics. This system has the ability to generate unique IDs for each face and remember each face with its associated ID. (Done by Gokulachalam)

## Features

- **Real-Time Face Detection and Tracking**: Utilizes OpenCV and a pre-trained face detection model to detect and track individual faces in real-time video streams.
- **Object Detection (Person)**: Employs the YOLOv8n object detection model to detect and identify persons in the video feed.
- **Visitor Proximity Analysis**: Analyzes the proximity of visitors to the camera, classifying them as "walkin" (close) or "glancing" (far).
- **Visitor Duration Tracking**: Tracks the duration of each visitor's presence, including the time they first appeared and their total time in view.
- **Age Estimation**: Leverages a pre-trained Vision Transformer (ViT) model to estimate the age range of detected faces.
- **Gender Classification**: Utilizes a pre-trained Transformer model to classify the gender of detected faces.
- **Screenshot Capture**: Captures and saves screenshots of detected faces for further analysis or record-keeping.
- **Data Logging**: Logs visitor data, including timestamps, face IDs, status, type (walkin, glancing, viewing), duration, and total visitors, to a CSV file.
- **Data Merging**: Combines the visitor logs with the predicted age ranges and genders, generating a comprehensive merged dataset.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pandas
- Ultralytics YOLOv8n
- Pillow
- Transformers
- Torch

## Installation

1. Clone the repository: `git clone https://github.com/your-username/real-time-visitor-analysis.git`
2. Navigate to the project directory: `cd real-time-visitor-analysis`
3. Install the required dependencies using pip: `pip install -r requirements.txt`
4. Ensure that the necessary pre-trained models (face detection, YOLOv8n, ViT age classifier, and gender classification) are downloaded and placed in the appropriate directories.

## Usage

1. Run the main script: `python main.py`
2. The system will start capturing video from the default camera and perform real-time visitor analysis.
3. Detected faces, persons, and relevant information will be displayed on the video feed.
4. Visitor data, including timestamps, face IDs, status, type, duration, and total visitors, will be logged to a CSV file (`face_logs.csv`).
5. After the analysis is complete, the script will merge the visitor logs with the predicted age ranges and genders, generating a comprehensive merged dataset (`merged_data.csv`).

## Customization

You can customize various aspects of the system, such as:

- Changing the face detection model
- Adjusting the YOLOv8n model
- Using different age estimation or gender classification models
- Modifying the proximity detection threshold
- Adjusting the screenshot capture conditions

Please refer to the code documentation and comments for more details on customization options.

## Contributing

Contributions to the Real-Time Visitor Analysis System are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
