import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_trained_model(model_path):
    """
    Loads the trained TensorFlow model.

    Args:
        model_path (str): Path to the saved model (.h5 file).

    Returns:
        tf.keras.Model: Loaded TensorFlow model.
    """
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model

def preprocess_frame(frame, target_size=(128, 128)):
    """
    Preprocesses the input frame for model prediction.

    Args:
        frame (np.array): The original frame captured from the webcam.
        target_size (tuple): The size to which the frame should be resized.

    Returns:
        np.array: Preprocessed frame ready for prediction.
    """
    # Resize the frame to match the training input size
    resized_frame = cv2.resize(frame, target_size)
    
    # Normalize the pixel values to [0, 1]
    normalized_frame = resized_frame.astype('float32') / 255.0
    
    # Expand dimensions to match model's input shape (1, height, width, channels)
    input_frame = np.expand_dims(normalized_frame, axis=0)
    
    return input_frame

def predict_bounding_box(model, preprocessed_frame):
    """
    Predicts the bounding box coordinates using the trained model.

    Args:
        model (tf.keras.Model): The trained TensorFlow model.
        preprocessed_frame (np.array): The preprocessed frame.

    Returns:
        list: Predicted bounding box [x_center, y_center, width, height].
    """
    prediction = model.predict(preprocessed_frame)
    prediction = np.clip(prediction[0], 0, 1)  # Ensure predictions are within [0,1]
    return prediction.tolist()

def draw_bounding_box(frame, bbox, confidence_threshold=0.1):
    """
    Draws the bounding box on the frame if it meets the confidence threshold.

    Args:
        frame (np.array): The original frame.
        bbox (list): Predicted bounding box [x_center, y_center, width, height].
        confidence_threshold (float): Minimum width and height to consider as valid detection.

    Returns:
        np.array: Frame with bounding box drawn (if valid).
    """
    x_center, y_center, width, height = bbox

    # Check if the bounding box has non-zero dimensions
    if width < confidence_threshold or height < confidence_threshold:
        return frame  # No valid detection

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Convert normalized coordinates to absolute pixel values
    x_center_abs = x_center * frame_width
    y_center_abs = y_center * frame_height
    width_abs = width * frame_width
    height_abs = height * frame_height

    # Calculate top-left and bottom-right coordinates
    x_min = int(x_center_abs - (width_abs / 2))
    y_min = int(y_center_abs - (height_abs / 2))
    x_max = int(x_center_abs + (width_abs / 2))
    y_max = int(y_center_abs + (height_abs / 2))

    # Ensure coordinates are within frame boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_width - 1, x_max)
    y_max = min(frame_height - 1, y_max)

    # Draw the bounding box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Optionally, display the bounding box coordinates
    label = f"Rubik's Cube: [{x_center:.2f}, {y_center:.2f}, {width:.2f}, {height:.2f}]"
    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

    return frame

def main():
    # Path to the trained model
    # /Users/ludvigeriksonbrangstrup/Solving-Rubik-s-Cube-With-Robot-Manipulator/Neural network train masking/rubiks_cube_detector.h5
    # model_path = 'rubiks_cube_detector.h5'  # Update this path if necessary
    model_path = 'Neural network train masking/rubiks_cube_detector_v2.h5'
    # Load the trained model
    model = load_trained_model(model_path)

    # Initialize webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting live Rubik's Cube detection. Press 'q' to exit.")

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame.")
                break

            # Make a copy of the original frame for visualization
            output_frame = frame.copy()

            # Preprocess the frame
            preprocessed_frame = preprocess_frame(frame, target_size=(128, 128))

            # Predict the bounding box
            bbox = predict_bounding_box(model, preprocessed_frame)

            # Draw the bounding box on the original frame
            output_frame = draw_bounding_box(output_frame, bbox, confidence_threshold=0.05)

            # Display the resulting frame
            cv2.imshow('Rubik\'s Cube Detection', output_frame)

            # Exit condition: press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting live detection.")
                break

    except KeyboardInterrupt:
        # Allow graceful exit with Ctrl+C
        print("Interrupted by user. Exiting.")

    finally:
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()