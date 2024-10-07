import cv2
import numpy as np
from sklearn.cluster import KMeans
import time
    
class Scanner:
    def __init__(self):
        # Define the RGB values for the colors
        self.colors_rgb = {
            'orange':  (216, 114, 95),
            'yellow': (173, 189, 106),
            'green': (0, 180, 0),
            'blue': (0, 0, 180),
            'white': (155, 155, 155),
            'red': (177, 48, 48)  # Added red for detection
        }

        # Convert RGB to HSV ranges
        self.colors_hsv = self._create_hsv_ranges()

        # Define color mappings for drawing (BGR format)
        self.color_map_bgr = {
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'white': (255, 255, 200),
        }

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Error: Could not open video capture.")

    def _convert_rgb_to_hsv(self, rgb):  # TODO : remove
        """Convert an RGB tuple to HSV."""
        rgb_array = np.uint8([[rgb]])
        hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        return hsv_array[0][0]

    def _create_hsv_ranges(self):
        """Create HSV ranges for each color based on RGB values."""
        hue_range = 30
        saturation_range = 70
        value_range = 100
        colors_hsv = {}

        for color_name, rgb in self.colors_rgb.items():
            rgb_array = np.uint8([[rgb]])
            hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
            hsv = hsv_array[0][0]

            # hsv = self._convert_rgb_to_hsv(rgb)
            lower = np.array([
                max(hsv[0] - hue_range, 0),
                max(hsv[1] - saturation_range, 0),
                max(hsv[2] - value_range, 0)
            ])
            upper = np.array([
                min(hsv[0] + hue_range, 180),
                min(hsv[1] + saturation_range, 255),
                min(hsv[2] + value_range, 255)
            ])
            colors_hsv[color_name] = [{'lower': lower, 'upper': upper}]

        # Add specific ranges for red and white
        colors_hsv['white'] = [
            {'lower': np.array([0, 0, 150]), 'upper': np.array([180, 50, 255])}
        ]
        colors_hsv['red'] = [
            {'lower': np.array([0, 150, 70]), 'upper': np.array([10, 255, 255])},
            {'lower': np.array([160, 150, 70]), 'upper': np.array([180, 255, 255])}
        ]

        return colors_hsv

    def _capture_frame(self): # TODO , remove
        """Capture a single frame from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("Error: Could not read frame from video capture.")
        # Optional: Crop the frame if necessary (adjust as per your camera setup)
        # Example: frame = frame[0:-1, 400:-1]
        return frame

    def _get_binary(self, frame):
        """
        Convert the captured frame to a binary mask based on color detection.

        :param frame: The captured BGR frame.
        :return: Binary mask highlighting areas of interest.
        """
        # Adjust brightness
        brightness_factor = 0.5  # Adjust as needed
        frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Extract the V (value) channel
        v_channel = hsv[:, :, 2]

        # Apply a threshold to the V channel to get a binary image
        _, binary = cv2.threshold(v_channel, 40, 255, cv2.THRESH_BINARY_INV)

        return binary
    
    def _angle_between_vectors(self, v1, v2):
        """Calculate the angle between two vectors in degrees."""
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)
    
    def _get_mask(self, frame, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(frame)
        for cnt in contours:
            # Approximate contour
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # If the contour has 4 sides, it could be a rectangle
            if len(approx) == 4 and cv2.isContourConvex(approx):
                # Compute area and check if it's large enough
                area = cv2.contourArea(approx)
                if area > 5000:  # Adjust this threshold as needed
                    # Check if the rectangle has roughly 90-degree corners
                    angles = []
                    for i in range(4):
                        p1 = approx[i][0]
                        p2 = approx[(i + 1) % 4][0]
                        p3 = approx[(i + 2) % 4][0]
                        v1 = p1 - p2
                        v2 = p3 - p2
                        angle = self._angle_between_vectors(v1, v2)
                        angles.append(angle)
                    if all(85 <= angle <= 95 for angle in angles):  # Adjust this threshold as needed
                        # Get the center of the rectangle
                        M = cv2.moments(approx)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            # Draw the contour and center
                            cv2.drawContours(mask, [approx], -1, (255, 255, 255), thickness=cv2.FILLED)
                            #cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)
                            found_cube = True
                            # show the amsk
                            # cv2.imshow('mask', mask)
                            return True, mask
        print("No cube found")
        return False, None


    def _apply_mask(self, frame, mask): #TODO remove, not used anymore
        """
        Apply the binary mask to the frame to isolate regions of interest.

        :param frame: The original BGR frame.
        :param mask: The binary mask.
        :return: Masked frame.
        """
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        return masked_frame

    def _find_squares(self, masked_frame, show_squares = False):
        """
        Detect colored squares in the masked frame.

        :param masked_frame: The masked BGR frame.
        :return: List of detected squares with their properties.
        """
        squares = []
        hsv_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

        for color_name, color_ranges in self.colors_hsv.items():
            mask = None
            for color_range in color_ranges:
                lower = color_range['lower']
                upper = color_range['upper']
                curr_mask = cv2.inRange(hsv_frame, lower, upper)
                if mask is None:
                    mask = curr_mask
                else:
                    mask = cv2.bitwise_or(mask, curr_mask)
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                # Approximate contour
                epsilon = 0.08 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                # If the contour has 4 sides, it could be a square
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    area = cv2.contourArea(approx)
                    if 200 < area < 15000:  # Adjust thresholds as needed
                        M = cv2.moments(approx)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            squares.append({'position': (cx, cy), 'color': color_name, 'area': area})
                            if True:
                                cv2.drawContours(masked_frame, [approx], -1, self.color_map_bgr[color_name], 2)
                                cv2.circle(masked_frame, (cx, cy), 5, (0, 0, 0), -1)
        return squares, masked_frame

    def _filter_squares(self, squares):
        """
        Filter the detected squares based on proximity to the mean position and area consistency.

        :param squares: List of detected squares.
        :return: Filtered list of squares.
        """
        if len(squares) < 4:
            raise ValueError("Not enough squares to filter 9 closest points")

        positions = np.array([s['position'] for s in squares])
        areas = np.array([s['area'] for s in squares])

        mean_point = np.mean(positions, axis=0)
        mean_area = np.mean(areas)

        filtered_points = []
        for square in squares:
            distance = np.linalg.norm(np.array(square['position']) - mean_point)
            area_deviation = abs(square['area'] - mean_area) / mean_area
            if distance <= 200 and area_deviation <= 0.2:
                filtered_points.append(square)

        if len(filtered_points) >= 4:
            return filtered_points

        raise ValueError("No suitable points found within the distance and area criteria")

    def _squares_to_grid(self, squares):
        """
        Convert the list of 9 squares into a 3x3 grid based on their positions.

        :param squares: List of 9 filtered squares.
        :return: 3x3 list representing the colors of the Rubik's cube face.
        """
        grid_array = [['' for _ in range(3)] for _ in range(3)]
        positions = np.array([s['position'] for s in squares])
        colors_list = [s['color'] for s in squares]

        # Perform KMeans clustering on x and y positions to determine grid layout
        kmeans_x = KMeans(n_clusters=3, random_state=0).fit(positions[:, 0].reshape(-1, 1))
        x_labels = kmeans_x.labels_
        kmeans_y = KMeans(n_clusters=3, random_state=0).fit(positions[:, 1].reshape(-1, 1))
        y_labels = kmeans_y.labels_

        # Order the clusters
        x_order = np.argsort(kmeans_x.cluster_centers_.flatten())
        y_order = np.argsort(kmeans_y.cluster_centers_.flatten())

        x_label_to_grid = {label: idx for idx, label in enumerate(x_order)}
        y_label_to_grid = {label: idx for idx, label in enumerate(y_order)}

        # Populate the grid array
        for i in range(len(squares)):
            x_grid = x_label_to_grid[x_labels[i]]
            y_grid = y_label_to_grid[y_labels[i]]
            grid_array[y_grid][x_grid] = colors_list[i]

        return grid_array


    def scan_face(self):
        """
        Process frames to detect Rubik's cube face and return a 3x3 array of colors.
        Tries for 10 seconds before returning None if not detected.
    
        :return: 3x3 list representing the colors of the Rubik's cube face or None if not detected.
        """
        start_time = time.time()
        timeout = 10  # seconds
    
        while time.time() - start_time < timeout:
            if cv2.waitKey(1) == ord('q'):
                break
            try:
                # frame = self._capture_frame()
                ret, frame = self.cap.read()
                if not ret:
                    raise IOError("Error: Could not read frame from video capture.")
                
                
                cv2.imshow('frame', frame)
                binary = self._get_binary(frame)
                found_face, mask = self._get_mask(frame, binary)
                masked_frame = frame
                if found_face:
                    cv2.imshow('Mask', mask)
                    masked_frame = cv2.bitwise_and(frame, mask)
                    cv2.imshow('masked_frame', masked_frame)
                    #masked_frame = self._apply_mask(frame, binary_mask)
                    squares, masked_frame = self._find_squares(masked_frame, show_squares = True)
                    cv2.imshow('Frame', masked_frame)

                    print(squares)
    
                    # Filter squares based on proximity and area
                    try:
                        filtered_squares = self._filter_squares(squares)
                    except ValueError:
                        # Not enough squares detected in this frame, continue to next iteration
                        continue

                   
        
                    # Convert filtered squares to a 3x3 grid
                    grid = self._squares_to_grid(squares)
                    return grid
    
            except IOError as e:
                print(f"IOError encountered: {e}")
                continue

    
        print("Failed to detect Rubik's cube face within 10 seconds.")
        return None

    def close(self):
        """Release the video capture and close any OpenCV windows."""
        self.cap.release()
        cv2.destroyAllWindows()



scanner = Scanner()
face = scanner.scan_face()
print(face)
scanner.close




