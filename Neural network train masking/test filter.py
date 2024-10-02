# import cv2
# import numpy as np

# def adjust_rgb(frame, r_factor, g_factor, b_factor):
#     """
#     Adjust the RGB strength of the frame.
    
#     Args:
#         frame (np.array): The original frame captured from the webcam.
#         r_factor (float): Factor to adjust the red channel.
#         g_factor (float): Factor to adjust the green channel.
#         b_factor (float): Factor to adjust the blue channel.
    
#     Returns:
#         np.array: Frame with adjusted RGB strength.
#     """
#     # Split the frame into its BGR components
#     b, g, r = cv2.split(frame)
    
#     # Adjust each channel by the respective factor
#     r = cv2.multiply(r, r_factor)
#     g = cv2.multiply(g, g_factor)
#     b = cv2.multiply(b, b_factor)
    
#     # Merge the channels back together
#     adjusted_frame = cv2.merge([b, g, r])
    
#     return adjusted_frame

# def on_trackbar(val):
#     pass

# def main():
#     # Initialize webcam (0 is the default camera)
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     # Create a window
#     cv2.namedWindow('Live Camera Feed')

#     # Create trackbars for RGB adjustment
#     cv2.createTrackbar('R', 'Live Camera Feed', 100, 200, on_trackbar)
#     cv2.createTrackbar('G', 'Live Camera Feed', 100, 200, on_trackbar)
#     cv2.createTrackbar('B', 'Live Camera Feed', 100, 200, on_trackbar)

#     print("Starting live camera feed. Press 'q' to exit.")

#     try:
#         while True:
#             # Capture frame-by-frame
#             ret, frame = cap.read()

#             if not ret:
#                 print("Failed to grab frame.")
#                 break

#             # Get the current positions of the trackbars
#             r_factor = cv2.getTrackbarPos('R', 'Live Camera Feed') / 100.0
#             g_factor = cv2.getTrackbarPos('G', 'Live Camera Feed') / 100.0
#             b_factor = cv2.getTrackbarPos('B', 'Live Camera Feed') / 100.0

#             # Adjust the RGB strength of the frame
#             adjusted_frame = adjust_rgb(frame, r_factor, g_factor, b_factor)

#             # Display the resulting frame
#             cv2.imshow('Live Camera Feed', adjusted_frame)

#             # Exit condition: press 'q' to quit
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("Exiting live camera feed.")
#                 break

#     except KeyboardInterrupt:
#         # Allow graceful exit with Ctrl+C
#         print("Interrupted by user. Exiting.")

#     finally:
#         # When everything is done, release the capture
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()




import cv2
import numpy as np

def adjust_rgb(frame, r_factor, g_factor, b_factor):
    """
    Adjust the RGB strength of the frame.
    
    Args:
        frame (np.array): The original frame captured from the webcam.
        r_factor (float): Factor to adjust the red channel.
        g_factor (float): Factor to adjust the green channel.
        b_factor (float): Factor to adjust the blue channel.
    
    Returns:
        np.array: Frame with adjusted RGB strength.
    """
    # Split the frame into its BGR components
    b, g, r = cv2.split(frame)
    
    # Adjust each channel by the respective factor
    r = cv2.multiply(r, r_factor)
    g = cv2.multiply(g, g_factor)
    b = cv2.multiply(b, b_factor)
    
    # Merge the channels back together
    adjusted_frame = cv2.merge([b, g, r])
    
    return adjusted_frame

def adjust_saturation_brightness_contrast(frame, saturation_factor, brightness_factor, contrast_factor):
    """
    Adjust the saturation, brightness, and contrast of the frame.
    
    Args:
        frame (np.array): The original frame captured from the webcam.
        saturation_factor (float): Factor to adjust the saturation.
        brightness_factor (float): Factor to adjust the brightness.
        contrast_factor (float): Factor to adjust the contrast.
    
    Returns:
        np.array: Frame with adjusted saturation, brightness, and contrast.
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Adjust the saturation
    hsv[..., 1] = cv2.multiply(hsv[..., 1], saturation_factor)
    
    # Convert back to BGR color space
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Adjust the brightness and contrast
    frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=brightness_factor)
    
    return frame

def on_trackbar(val):
    pass

def main():
    # Initialize webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a window
    cv2.namedWindow('Live Camera Feed')

    # Create trackbars for RGB adjustment
    cv2.createTrackbar('R', 'Live Camera Feed', 80, 200, on_trackbar)
    cv2.createTrackbar('G', 'Live Camera Feed', 80, 200, on_trackbar)
    cv2.createTrackbar('B', 'Live Camera Feed', 80, 200, on_trackbar)
    
    # Create trackbars for saturation, brightness, and contrast adjustment
    cv2.createTrackbar('Saturation', 'Live Camera Feed', 50, 200, on_trackbar)
    cv2.createTrackbar('Brightness', 'Live Camera Feed', 50, 100, on_trackbar)
    cv2.createTrackbar('Contrast', 'Live Camera Feed', 80, 200, on_trackbar)

    print("Starting live camera feed. Press 'q' to exit.")

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame.")
                break

            # Get the current positions of the trackbars
            r_factor = cv2.getTrackbarPos('R', 'Live Camera Feed') / 100.0
            g_factor = cv2.getTrackbarPos('G', 'Live Camera Feed') / 100.0
            b_factor = cv2.getTrackbarPos('B', 'Live Camera Feed') / 100.0
            saturation_factor = cv2.getTrackbarPos('Saturation', 'Live Camera Feed') / 100.0
            brightness_factor = cv2.getTrackbarPos('Brightness', 'Live Camera Feed') - 50
            contrast_factor = cv2.getTrackbarPos('Contrast', 'Live Camera Feed') / 100.0

            # Adjust the RGB strength of the frame
            adjusted_frame = adjust_rgb(frame, r_factor, g_factor, b_factor)
            
            # Adjust the saturation, brightness, and contrast of the frame
            adjusted_frame = adjust_saturation_brightness_contrast(adjusted_frame, saturation_factor, brightness_factor, contrast_factor)

            # Display the resulting frame
            cv2.imshow('Live Camera Feed', adjusted_frame)

            # Exit condition: press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting live camera feed.")
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