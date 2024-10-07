import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy import ndimage

# Define color ranges in HSV
# colors = {
#     'red': [
#         {'lower': np.array([0, 70, 50]), 'upper': np.array([10, 255, 255])},
#         {'lower': np.array([170, 70, 50]), 'upper': np.array([180, 255, 255])}
#     ],
#     'orange': [{'lower': np.array([11, 70, 50]), 'upper': np.array([25, 255, 255])}],
#     'yellow': [{'lower': np.array([26, 70, 50]), 'upper': np.array([35, 255, 255])}],
#     'green': [{'lower': np.array([36, 70, 50]), 'upper': np.array([85, 255, 255])}],
#     'blue': [{'lower': np.array([86, 70, 50]), 'upper': np.array([125, 255, 255])}],
#     'white': [{'lower': np.array([0, 0, 200]), 'upper': np.array([180, 25, 255])}],
# }

# colors = {
#     'red': [{'lower': np.array([0, 50, 50]), 'upper': np.array([15, 255, 255])},
#             {'lower': np.array([165, 50, 50]), 'upper': np.array([180, 255, 255])}],
#     'orange': [{'lower': np.array([11, 70, 50]), 'upper': np.array([25, 255, 255])}],
#     'yellow': [{'lower': np.array([26, 70, 50]), 'upper': np.array([32, 255, 255])}],  # More green tone
#     'green': [{'lower': np.array([36, 70, 50]), 'upper': np.array([85, 255, 255])}],
#     'blue': [{'lower': np.array([86, 70, 50]), 'upper': np.array([125, 255, 255])}],
#     'white': [{'lower': np.array([0, 0, 180]), 'upper': np.array([180, 50, 255])}],  # Adjusted for bluish white
# }

# colors = {
#     'red': [{'lower': np.array([170, 150, 150]), 'upper': np.array([180, 255, 255])},
#             {'lower': np.array([0, 150, 150]), 'upper': np.array([10, 255, 255])}],  # Red spans across 0 and 180
#     'orange': [{'lower': np.array([10, 80, 130]), 'upper': np.array([30, 200, 255])}],
#     'yellow': [{'lower': np.array([30, 70, 150]), 'upper': np.array([40, 140, 255])}],  # More green tone
#     'green': [{'lower': np.array([50, 70, 50]), 'upper': np.array([85, 255, 255])}],
#     'blue': [{'lower': np.array([100, 70, 70]), 'upper': np.array([120, 255, 255])}],
#     'white': [{'lower': np.array([0, 0, 200]), 'upper': np.array([180, 100, 255])}],  # Adjusted for bluish white
# }


# Define the RGB values for the colors
# colors_rgb = {
#     'orange': (255, 158, 99),
#     'yellow': (250, 224, 141),
#     'green': (80, 140, 80),
#     'blue': (16, 48, 243),
#     'white': (170, 160, 200)
# }


# colors_rgb = {
#     'orange':  (255, 158, 99),
#     'yellow': (255, 255, 0),
#     'green': (0, 128, 0),
#     'blue': (0, 0, 255),
#     'white': (255, 255, 255)
# }


# colors_rgb = {
#     'orange':  (216, 114, 95),
#     'yellow': (173, 189, 106),
#     'green': (0, 128, 0),
#     'blue': (29, 75, 128),
#     'white': (191, 193, 184)
# }


colors_rgb = {
    'orange':  (216, 114, 95),
    'yellow': (173, 189, 106),
    'green': (0, 180, 0),
    'blue': (0, 0, 180),
    'white': (155, 155, 155)
}





# Convert RGB to HSV
def convert_rgb_to_hsv(rgb):
    rgb_array = np.uint8([[rgb]])
    hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    return hsv_array[0][0]

# Define the fixed ± range for HSV
hue_range = 30
saturation_range = 70
value_range = 100

# Create the HSV ranges
colors_hsv = {}
for color_name, rgb in colors_rgb.items():
    hsv = convert_rgb_to_hsv(rgb)
    lower = np.array([max(hsv[0] - hue_range, 0), max(hsv[1] - saturation_range, 0), max(hsv[2] - value_range, 0)])
    upper = np.array([min(hsv[0] + hue_range, 180), min(hsv[1] + saturation_range, 255), min(hsv[2] + value_range, 255)])
    colors_hsv[color_name] = [{'lower': lower, 'upper': upper}]

rgb_color = np.uint8([[[177, 48, 48]]])

# # Add the predefined red ranges
# colors_hsv['red'] = [{'lower': np.array([140, 20, 20]), 'upper': np.array([200, 70, 70])} ,
#                      {'lower': np.array([0, 20, 20]), 'upper': np.array([20,  70, 70])}]

# colors_hsv['red'] = [{'lower': np.array([130, 0, 0]), 'upper': np.array([255, 80, 80])} ,
#                      {'lower': np.array([0, 0, 0]), 'upper': np.array([50,  80, 80])}]


colors_hsv['white'] = [
    {'lower': np.array([0, 0, 150]), 'upper': np.array([180, 50, 255])}  # White/light gray range
]

colors_hsv['red'] = [
    {'lower': np.array([0, 150, 70]), 'upper': np.array([10, 255, 255])},  # Lower red range
    {'lower': np.array([160, 150, 70]), 'upper': np.array([180, 255, 255])}  # Upper red range
]
# Print the HSV ranges
for color_name, hsv_range in colors_hsv.items():
    print(f"{color_name}: {hsv_range}")

# The final colors dictionary
colors = colors_hsv

# Map color names to BGR colors for drawing
color_map_bgr = {
    'red': (0, 0, 255),
    'orange': (0, 165, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'white': (255, 255, 200),
}

# Map color names to matplotlib colors
color_map_matplotlib = {
    'red': 'red',
    'orange': 'orange',
    'yellow': 'yellow',
    'green': 'green',
    'blue': 'blue',
    'white': 'lightgray',
}

# Start video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# # Disable auto-exposure
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.05)  # 0.25 to enable manual exposure control

# # Set a fixed brightness level (value between 0 and 1)
# fixed_brightness = 0.05  # Adjust this value as needed
# cap.set(cv2.CAP_PROP_BRIGHTNESS, fixed_brightness)


# Initialize matplotlib
plt.ion()
fig, ax = plt.subplots()



def crop_center(frame, crop_size):
    """
    Crop the center of the frame to the specified crop size.
    
    :param frame: The input frame (image) to be cropped.
    :param crop_size: The size of the cropped region (width and height).
    :return: The cropped frame.
    """
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    half_crop_size = crop_size // 2
    
    start_x = max(center_x - half_crop_size, 0)
    start_y = max(center_y - half_crop_size, 0)
    end_x = min(center_x + half_crop_size, width)
    end_y = min(center_y + half_crop_size, height)
    
    cropped_frame = frame[start_y:end_y, start_x:end_x]
    return cropped_frame


while True:
    # Capture frame-by-frame
    found_cube = False
    ret, frame = cap.read()
    frame = frame[0:-1, 400:-1]

    # brightness_factor = 1 # Adjust this value to reduce brightness
    # frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Resize frame if necessary
    # frame = cv2.resize(frame, (640, 480))

    height, width, _ = frame.shape

    frame

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray image', gray)

    # Apply a threshold to get a binary image
    # _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    # _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    # 
    #_, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 20, 5)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract the V (value) channel
    v_channel = hsv[:, :, 2]

    # Apply a threshold to the V channel to get a binary image
    _, binary = cv2.threshold(v_channel, 120, 255, cv2.THRESH_BINARY_INV)

    # Display the filtered image in a separate window for debugging
    cv2.imshow('Filtered Image', binary)

    # # Apply morphological operations to clean up the binary image
    # kernel = np.ones((3, 3), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Morphed Image', binary)




    def angle_between_vectors(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)
    
    # Find contours
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
                    angle = angle_between_vectors(v1, v2)
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
                        cv2.imshow('mask', mask)


    frame = cv2.bitwise_and(frame, mask)
    if found_cube:
        brightness_factor = 0.8 # Adjust this value to reduce brightness
        frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        squares = []

        for color_name, color_ranges in colors.items():
            mask = None
            for color_range in color_ranges:
                lower = color_range['lower']
                upper = color_range['upper']
                curr_mask = cv2.inRange(hsv, lower, upper)
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
                    # Compute area and check if it's large enough
                    area = cv2.contourArea(approx)
                    if area > 200 and area < 30000:  # Adjust this threshold as needed
                        # Get the center of the square
                        M = cv2.moments(approx)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            squares.append({'position': (cx, cy), 'color': color_name, 'area': area})
                            # Draw the contour and center
                            cv2.drawContours(frame, [approx], -1, color_map_bgr[color_name], 2)
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)

        # Plot all detected centers before filtering



        

        
        # def filter_closest_points(squares):
        #     # Check if there are at least 6 squares
        #     if len(squares) < 6:
        #         raise ValueError("Not enough squares to filter 6-9 closest points")
        
        #     # Extract positions from the squares dictionary
        #     positions = np.array([s['position'] for s in squares])
        
        #     # Determine the number of clusters (min 6, max 9)
        #     n_clusters = min(10, len(squares))
        
        #     # Perform KMeans clustering to find clusters
        #     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(positions)
        
        #     # Get the cluster centers
        #     centers = kmeans.cluster_centers_
        
        #     # Find the closest point in the original positions to each cluster center
        #     closest_points = []
        #     for center in centers:
        #         distances = np.linalg.norm(positions - center, axis=1)
        #         closest_index = np.argmin(distances)
        #         closest_points.append(squares[closest_index])
        
        #     # Check if the points are within 300 units of each other
        #     def are_points_within_distance(points, max_distance=400):
        #         positions = np.array([p['position'] for p in points])
        #         distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
        #         return np.all(distances <= max_distance)
        
        #     # Filter points that are within 300 units of each other
        #     filtered_points = []
        #     for i in range(len(closest_points)):
        #         for j in range(i + 1, len(closest_points)):
        #             if are_points_within_distance(closest_points[i:j+1]):
        #                 filtered_points = closest_points[i:j+1]
        #                 if len(filtered_points) >= 6:
        #                     return filtered_points
        
        #     raise ValueError("No suitable points found within the distance criteria")
        
        # # Example usage
        # try:
        #     squares = filter_closest_points(squares)
        # except ValueError as e:
        #     squares = {}
        #     print(e)


        # def filter_rubiks_cube_face(squares):
        #     # Check if there are at least 9 squares
        #     if len(squares) < 9:
        #         raise ValueError("Not enough squares to identify a Rubik's cube face")
            
        #     # Extract positions from the squares list
        #     positions = np.array([s['position'] for s in squares], dtype=float)  # Shape (n, 2)
            
        #     # Center the positions
        #     mean_position = np.mean(positions, axis=0)
        #     positions_centered = positions - mean_position  # Shape (n, 2)

        #     # Step 1: Perform PCA to find principal axes in 2D
        #     pca = PCA(n_components=2)
        #     pca.fit(positions_centered)
        #     principal_axes = pca.components_  # Shape (2, 2)

        #     # Step 2: Rotate points so that principal axes align with coordinate axes
        #     rotation_matrix = principal_axes.T  # Shape (2, 2)
        #     rotated_positions = np.dot(positions_centered, rotation_matrix)  # Shape (n, 2)

        #     # Step 3: Determine grid spacing
        #     sorted_x = np.unique(np.sort(rotated_positions[:, 0]))
        #     sorted_y = np.unique(np.sort(rotated_positions[:, 1]))
        #     if len(sorted_x) < 2 or len(sorted_y) < 2:
        #         raise ValueError("Not enough unique positions to determine grid spacing")
            
        #     grid_spacing_x = np.median(np.diff(sorted_x))
        #     grid_spacing_y = np.median(np.diff(sorted_y))
        #     grid_spacing = (grid_spacing_x + grid_spacing_y) / 2.0

        #     if grid_spacing == 0:
        #         raise ValueError("Grid spacing calculated as zero")

        #     # Step 4: Quantize coordinates to a grid
        #     tolerance = grid_spacing * 0.2  # Allow some tolerance due to noise
        #     quantized_positions = np.round(rotated_positions / grid_spacing)

        #     # Step 5: Group points based on quantized coordinates
        #     grid_dict = {}
        #     for idx, qp in enumerate(quantized_positions):
        #         key = (int(qp[0]), int(qp[1]))
        #         if key in grid_dict:
        #             grid_dict[key].append(idx)
        #         else:
        #             grid_dict[key] = [idx]

        #     # Step 6: Find the 3x3 grid
        #     grid_keys = np.array(list(grid_dict.keys()))
        #     if grid_keys.size == 0:
        #         raise ValueError("No grid keys found")
        #     unique_x = np.unique(grid_keys[:, 0])
        #     unique_y = np.unique(grid_keys[:, 1])

        #     if len(unique_x) >= 3 and len(unique_y) >= 3:
        #         # Find the most frequent x and y coordinates to form a 3x3 grid
        #         x_counts = {x: np.sum(grid_keys[:, 0] == x) for x in unique_x}
        #         y_counts = {y: np.sum(grid_keys[:, 1] == y) for y in unique_y}
        #         top_x = sorted(x_counts, key=x_counts.get, reverse=True)[:3]
        #         top_y = sorted(y_counts, key=y_counts.get, reverse=True)[:3]

        #         # Collect the points in the 3x3 grid
        #         grid_points = []
        #         for x in top_x:
        #             for y in top_y:
        #                 key = (x, y)
        #                 if key in grid_dict:
        #                     idx_list = grid_dict[key]
        #                     grid_center = np.array([x, y]) * grid_spacing
        #                     distances = [np.linalg.norm(rotated_positions[idx] - grid_center) for idx in idx_list]
        #                     min_idx = np.argmin(distances)
        #                     selected_idx = idx_list[min_idx]
        #                     grid_points.append(squares[selected_idx])

        #         if len(grid_points) == 9:
        #             return grid_points

        #     raise ValueError("No suitable 3x3 grid found forming a Rubik's cube face")

        # # Example usage
        # try:
        #     squares = filter_rubiks_cube_face(squares)
        # except ValueError as e:
        #     squares = []
        #     print(e)




        # def filter_rubiks_cube_face(squares):
        #     # Check if there are at least 9 squares
        #     if len(squares) < 9:
        #         raise ValueError("Not enough squares to identify a Rubik's cube face")
            
        #     # Extract positions from the squares list
        #     positions = np.array([s['position'] for s in squares], dtype=float)  # Shape (n, 2)

        #     # Center the positions
        #     mean_position = np.mean(positions, axis=0)
        #     positions_centered = positions - mean_position  # Shape (n, 2)
            
        #     # Normalize positions for Hough Transform
        #     max_dim = np.max(np.abs(positions_centered))
        #     positions_normalized = positions_centered / max_dim  # Scale to -1 to 1
            
        #     # Create an image representation of the points
        #     image_size = 200  # Adjust as needed
        #     img = np.zeros((image_size, image_size))
        #     for x, y in positions_normalized:
        #         xi = int((x + 1) / 2 * (image_size - 1))
        #         yi = int((y + 1) / 2 * (image_size - 1))
        #         img[yi, xi] = 1  # Note that image coordinates are (row, col)
            
        #     # Apply Gaussian filter to smooth the image
        #     img_blurred = ndimage.gaussian_filter(img, sigma=2)
            
        #     # Perform Hough Transform
        #     from skimage.transform import hough_line, hough_line_peaks
        #     hspace, angles, distances = hough_line(img_blurred)
        #     accum, angle_peaks, dist_peaks = hough_line_peaks(hspace, angles, distances, num_peaks=6)
            
        #     # Extract lines in parameter space
        #     lines = []
        #     for angle, dist in zip(angle_peaks, dist_peaks):
        #         lines.append((angle, dist))
            
        #     # Find intersections of lines
        #     intersections = []
        #     for i in range(len(lines)):
        #         angle1, dist1 = lines[i]
        #         for j in range(i+1, len(lines)):
        #             angle2, dist2 = lines[j]
        #             # Compute intersection point
        #             sin_a1, cos_a1 = np.sin(angle1), np.cos(angle1)
        #             sin_a2, cos_a2 = np.sin(angle2), np.cos(angle2)
        #             determinant = cos_a1 * sin_a2 - sin_a1 * cos_a2
        #             if np.abs(determinant) < 1e-10:
        #                 continue  # Lines are parallel
        #             x = (sin_a2 * dist1 - sin_a1 * dist2) / determinant
        #             y = (-cos_a2 * dist1 + cos_a1 * dist2) / determinant
        #             intersections.append((x, y))
            
        #     if len(intersections) < 9:
        #         raise ValueError("Not enough intersections found to form a grid")
            
        #     # Convert intersections back to original coordinate system
        #     intersections = np.array(intersections)
        #     intersections = intersections * max_dim  # Scale back
        #     intersections = intersections + mean_position  # Translate back

        #     # Match the original points to the grid intersections
        #     distances = cdist(positions, intersections)
        #     min_dist_indices = np.argmin(distances, axis=1)
        #     grid_points_indices = []
        #     for idx in range(len(intersections)):
        #         matched_points = np.where(min_dist_indices == idx)[0]
        #         if len(matched_points) > 0:
        #             # Select the closest point
        #             point_idx = matched_points[np.argmin(distances[matched_points, idx])]
        #             grid_points_indices.append(point_idx)
            
        #     # Ensure we have 9 points
        #     if len(grid_points_indices) >= 9:
        #         selected_indices = grid_points_indices[:9]
        #         grid_points = [squares[idx] for idx in selected_indices]
        #         return grid_points
        #     else:
        #         raise ValueError("No suitable 3x3 grid found forming a Rubik's cube face")

        # # Example usage
        # try:
        #     squares = filter_rubiks_cube_face(squares)
        # except ValueError as e:
        #     squares = []
        #     print(e)
















        
        # def filter_square_points(squares):
        #     # Check if there are at least 8 squares
        #     if len(squares) < 8:
        #         raise ValueError("Not enough squares to filter 8 points forming a square")
            
        #     # Extract positions from the squares dictionary
        #     positions = np.array([s['position'] for s in squares])
            
        #     # Determine the number of clusters (min 8, max 9)
        #     n_clusters = min(10, len(squares))
            
        #     # Perform KMeans clustering to find clusters
        #     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(positions)
            
        #     # Get the cluster centers
        #     centers = kmeans.cluster_centers_
            
        #     # Find the closest point in the original positions to each cluster center
        #     closest_points = []
        #     for center in centers:
        #         distances = np.linalg.norm(positions - center, axis=1)
        #         closest_index = np.argmin(distances)
        #         closest_points.append(squares[closest_index])
            
        #     # Function to check if three points are collinear
        #     def are_collinear(p1, p2, p3, tolerance=1e-2):
        #         return abs((p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) < tolerance
            
        #     # Function to check if points form a square
        #     def form_square(points, tolerance=0.1):
        #         if len(points) != 8:
        #             return False
                
        #         # Calculate pairwise distances
        #         distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
                
        #         # Get unique distances
        #         unique_distances = np.unique(distances)
                
        #         # Check if there are exactly two unique distances (side and diagonal)
        #         if len(unique_distances) != 2:
        #             return False
                
        #         side, diagonal = sorted(unique_distances)
                
        #         # Check if the diagonal is roughly sqrt(2) times the side
        #         return abs(diagonal - np.sqrt(2) * side) < tolerance
            
        #     # Check if the points form a square
        #     for i in range(len(closest_points)):
        #         for j in range(i + 1, len(closest_points)):
        #             subset = closest_points[i:j+1]
        #             positions_subset = np.array([p['position'] for p in subset])
        #             if form_square(positions_subset):
        #                 return subset
            
        #     raise ValueError("No suitable points found forming a square")
        
        # # Example usage
        # try:
        #     squares = filter_square_points(squares)
        # except ValueError as e:
        #     squares = {}
        #     print(e)



        # def filter_closest_points(squares):
        #     # Check if there are at least 6 squares
        #     if len(squares) < 9:
        #         raise ValueError("Not enough squares to filter 6-9 closest points")
        
        #     # Extract positions from the squares dictionary
        #     positions = np.array([s['position'] for s in squares])
        
        #     # Calculate the mean point
        #     mean_point = np.mean(positions, axis=0)
        
        #     # Filter out points that are further than 400 units from the mean point
        #     filtered_points = []
        #     for square in squares:
        #         distance = np.linalg.norm(np.array(square['position']) - mean_point)
        #         if distance <= 200:
        #             filtered_points.append(square)
        
        #     # Check if there are at least 6 points within the distance criteria
        #     if len(filtered_points) >= 6:
        #         return filtered_points
        
        #     raise ValueError("No suitable points found within the distance criteria")
        
        # # Example usage
        # try:
        #     squares = filter_closest_points(squares)
        # except ValueError as e:
        #     squares = []
        #     print(e)




        
        def filter_closest_points(squares):
            # Check if there are at least 9 squares
            if len(squares) < 4:
                raise ValueError("Not enough squares to filter 6-9 closest points")
        
            # Extract positions and areas from the squares dictionary
            positions = np.array([s['position'] for s in squares])
            areas = np.array([s['area'] for s in squares])

        
            # Calculate the mean point and mean area
            mean_point = np.mean(positions, axis=0)
            mean_area = np.mean(areas)
        
            # Filter out points that are further than 200 units from the mean point
            # and have an area within 20% of the mean area
            filtered_points = []
            for square in squares:
                distance = np.linalg.norm(np.array(square['position']) - mean_point)
                area_deviation = abs(square['area'] - mean_area) / mean_area
                if distance <= 100000: #and area_deviation <= 0.2:
                    filtered_points.append(square)
        
            # Check if there are at least 6 points within the distance and area criteria
            if len(filtered_points) == 4:
                return filtered_points
        
            raise ValueError("No suitable points found within the distance and area criteria")
        
        # # Example usage
        # try:
        #     squares = filter_closest_points(squares)
        # except ValueError as e:
        #     squares = []
        #     print(e)
        
        




        ax.clear()
        for square in squares:
            cx, cy = square['position']
            color = color_map_matplotlib[square['color']]
            ax.scatter(cx, cy, c=color, s=100, marker='o')


        # Figur 1


        # Set plot limits and grid
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates
        ax.grid(True)
        plt.draw()
        plt.pause(0.001)


        # Tar ut ut färgerna och plottar dom i guit som heter Rubricks cube grid


        # If we have enough squares, perform clustering
        if len(squares) >= 6:
            x_positions = np.array([s['position'][0] for s in squares])
            y_positions = np.array([s['position'][1] for s in squares])
            colors_list = [s['color'] for s in squares]
        
            # Perform KMeans clustering on x and y positions
            kmeans_x = KMeans(n_clusters=3, random_state=0).fit(x_positions.reshape(-1, 1))
            x_labels = kmeans_x.labels_
            kmeans_y = KMeans(n_clusters=3, random_state=0).fit(y_positions.reshape(-1, 1))
            y_labels = kmeans_y.labels_
        
            # Map cluster labels to grid positions
            x_centers = kmeans_x.cluster_centers_.flatten()
            y_centers = kmeans_y.cluster_centers_.flatten()
            x_order = np.argsort(x_centers)
            y_order = np.argsort(y_centers)
        
            x_label_to_grid = {label: idx for idx, label in enumerate(x_order)}
            y_label_to_grid = {label: idx for idx, label in enumerate(y_order)}
        
            grid_positions = []
            for i in range(len(squares)):
                x_grid = x_label_to_grid[x_labels[i]]
                y_grid = y_label_to_grid[y_labels[i]]
                grid_positions.append((x_grid, y_grid))
        
            # Draw the grid on a separate image
            grid_image = np.zeros((300, 300, 3), dtype=np.uint8)
            for i in range(len(squares)):
                x_grid, y_grid = grid_positions[i]
                color = color_map_bgr[colors_list[i]]
                # Invert y_grid to match image coordinate system
                y_grid_inv = y_grid  # Because y increases downward in images
                cv2.rectangle(
                    grid_image,
                    (x_grid * 100, y_grid_inv * 100),
                    ((x_grid + 1) * 100, (y_grid_inv + 1) * 100),
                    color,
                    -1,
                )
                # Optional: Draw grid lines
                cv2.rectangle(
                    grid_image,
                    (x_grid * 100, y_grid_inv * 100),
                    ((x_grid + 1) * 100, (y_grid_inv + 1) * 100),
                    (0, 0, 0),
                    2,
                )
        
            # Show the grid image
            cv2.imshow('Rubik\'s Cube Grid', grid_image)
        
            # Show the grid image
            cv2.imshow('Rubik\'s Cube Grid', grid_image)

            # Plot the centers on a matplotlib grid for debugging
            # ax.clear()
            # for i in range(len(squares)):
            #     x_grid, y_grid = grid_positions[i]
            #     color = color_map_matplotlib[colors_list[i]]
            #     # Invert y_grid to match standard plotting coordinates
            #     y_grid_plot = y_grid  # y increases upward in plots
            #     ax.scatter(x_grid, y_grid_plot, c=color, s=1000, marker='o')
            #     # Commented out the part that plots squares
            #     # ax.scatter(x_grid, y_grid_plot, c=color, s=1000, marker='s')

            # # Set plot limits and grid
            # ax.set_xlim(-0.5, 2.5)
            # ax.set_ylim(-0.5, 2.5)
            # ax.set_xticks([0, 1, 2])
            # ax.set_yticks([0, 1, 2])
            ax.grid(True)
            plt.draw()
            plt.pause(0.001)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()