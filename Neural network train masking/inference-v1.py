import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

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

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjusts brightness and contrast of the image.

    Args:
        image (np.array): Original image in BGR format.
        brightness (int): Value from -100 to 100 to adjust brightness.
        contrast (int): Value from -100 to 100 to adjust contrast.

    Returns:
        np.array: Adjusted image.
    """
    # Brightness adjustment
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        image = cv2.convertScaleAbs(image, alpha=alpha_b, gamma=gamma_b)
    
    # Contrast adjustment
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1 - f)
        image = cv2.convertScaleAbs(image, alpha=alpha_c, gamma=gamma_c)
    
    return image

def adjust_color_balance(image, blue=1.0, green=1.0, red=1.0):
    """
    Adjusts the color balance of the image by scaling B, G, R channels.

    Args:
        image (np.array): Original image in BGR format.
        blue (float): Scaling factor for Blue channel.
        green (float): Scaling factor for Green channel.
        red (float): Scaling factor for Red channel.

    Returns:
        np.array: Color-balanced image.
    """
    # Split the image into B, G, R channels
    B, G, R = cv2.split(image)
    
    # Scale each channel
    B = cv2.multiply(B, blue)
    G = cv2.multiply(G, green)
    R = cv2.multiply(R, red)
    
    # Merge back the channels
    balanced_image = cv2.merge([B, G, R])
    
    # Clip the values to [0,255] and convert to uint8
    balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)
    
    return balanced_image

def draw_center_point(frame, center_point, confidence_threshold=0.05):
    """
    Draws a large red dot at the center point on the frame if it meets the confidence threshold.

    Args:
        frame (np.array): The original frame.
        center_point (list): Predicted center point [x_center, y_center].
        confidence_threshold (float): Minimum confidence to consider as valid detection.

    Returns:
        np.array: Frame with the center point drawn (if valid).
    """
    x_center, y_center = center_point

    # Check if the center point has valid confidence
    if x_center < confidence_threshold or y_center < confidence_threshold:
        return frame  # No valid detection

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Convert normalized coordinates to absolute pixel values
    x_center_abs = int(x_center * frame_width)
    y_center_abs = int(y_center * frame_height)

    # Define the radius and color of the red dot
    radius = 10  # Radius of the red dot in pixels
    color = (0, 0, 255)  # Red color in BGR
    thickness = -1  # Solid circle

    # Draw the red dot on the frame
    cv2.circle(frame, (x_center_abs, y_center_abs), radius, color, thickness)

    return frame

def create_sliders(window_name):
    """
    Creates sliders for brightness, contrast, and color balance adjustments.

    Args:
        window_name (str): Name of the window where sliders will be displayed.
    """
    cv2.namedWindow(window_name)
    
    # Create Trackbars for Brightness and Contrast
    cv2.createTrackbar('Brightness', window_name, 100, 200, lambda x: None)  # Range: 0-200, default=100
    cv2.createTrackbar('Contrast', window_name, 100, 200, lambda x: None)    # Range: 0-200, default=100
    
    # Create Trackbars for Color Balance (Scaling factors from 0.0 to 3.0, represented as 0-300)
    cv2.createTrackbar('Blue', window_name, 100, 300, lambda x: None)         # Scale: 0.0-3.0
    cv2.createTrackbar('Green', window_name, 100, 300, lambda x: None)        # Scale: 0.0-3.0
    cv2.createTrackbar('Red', window_name, 100, 300, lambda x: None)          # Scale: 0.0-3.0

def get_trackbar_values(window_name):
    """
    Retrieves the current values of the sliders.

    Args:
        window_name (str): Name of the window containing the sliders.

    Returns:
        tuple: Brightness, Contrast, Blue, Green, Red values.
    """
    brightness = cv2.getTrackbarPos('Brightness', window_name) - 100  # Shift to range: -100 to +100
    contrast = cv2.getTrackbarPos('Contrast', window_name) - 100      # Shift to range: -100 to +100
    
    # Retrieve color balance scaling factors
    blue = cv2.getTrackbarPos('Blue', window_name) / 100.0          # Scale: 0.0-3.0
    green = cv2.getTrackbarPos('Green', window_name) / 100.0        # Scale: 0.0-3.0
    red = cv2.getTrackbarPos('Red', window_name) / 100.0            # Scale: 0.0-3.0
    
    return brightness, contrast, blue, green, red

def main():
    # Path to the trained model
    model_path = 'Neural network train masking/rubiks_cube_detector_v2.h5'  # Update this path if necessary

    # Load the trained model
    model = load_trained_model(model_path)

    # Initialize webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting live Rubik's Cube detection. Press 'q' to exit.")

    # Create sliders window
    sliders_window = 'Adjustments'
    create_sliders(sliders_window)

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame.")
                break

            # Make a copy of the original frame for visualization
            output_frame = frame.copy()

            # Retrieve slider values
            brightness, contrast, blue, green, red = get_trackbar_values(sliders_window)

            # Adjust brightness and contrast
            adjusted_frame = adjust_brightness_contrast(output_frame, brightness=brightness, contrast=contrast)

            # Adjust color balance
            adjusted_frame = adjust_color_balance(adjusted_frame, blue=blue, green=green, red=red)

            # Preprocess the frame for prediction
            preprocessed_frame = preprocess_frame(adjusted_frame, target_size=(128, 128))

            # Predict the center point
            bbox = predict_bounding_box(model, preprocessed_frame)

            # Draw the center point on the original frame
            output_frame = draw_center_point(output_frame, bbox, confidence_threshold=0.05)

            # Display the resulting frame
            cv2.imshow('Rubik\'s Cube Detection', output_frame)
            cv2.imshow(sliders_window, adjusted_frame)  # Show the adjusted frame in the sliders window

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