import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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

colors_rgb = {
    'orange':  (216, 114, 95),
    'yellow': (173, 189, 106),
    'green': (0, 128, 0),
    'blue': (29, 75, 128),
    'white': (191, 193, 184)
}



# Convert RGB to HSV
def convert_rgb_to_hsv(rgb):
    rgb_array = np.uint8([[rgb]])
    hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    return hsv_array[0][0]

# Define the fixed ± range for HSV
hue_range = 30
saturation_range = 70
value_range = 70

# Create the HSV ranges
colors_hsv = {}
for color_name, rgb in colors_rgb.items():
    hsv = convert_rgb_to_hsv(rgb)
    lower = np.array([max(hsv[0] - hue_range, 0), max(hsv[1] - saturation_range, 0), max(hsv[2] - value_range, 0)])
    upper = np.array([min(hsv[0] + hue_range, 180), min(hsv[1] + saturation_range, 255), min(hsv[2] + value_range, 255)])
    colors_hsv[color_name] = [{'lower': lower, 'upper': upper}]

rgb_color = np.uint8([[[177, 48, 48]]])

# Add the predefined red ranges
colors_hsv['red'] = [{'lower': np.array([140, 20, 20]), 'upper': np.array([200, 70, 70])} ,
                     {'lower': np.array([0, 20, 20]), 'upper': np.array([20,  70, 70])}]

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

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize matplotlib
plt.ion()
fig, ax = plt.subplots()

while True:
    # Capture frame-by-frame
    found_cube = False
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Resize frame if necessary
    # frame = cv2.resize(frame, (640, 480))

    height, width, _ = frame.shape

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to get a binary image
    # _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    # _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract the V (value) channel
    v_channel = hsv[:, :, 2]

    # Apply a threshold to the V channel to get a binary image
    _, binary = cv2.threshold(v_channel, 60, 255, cv2.THRESH_BINARY_INV)

    # Display the filtered image in a separate window for debugging
    cv2.imshow('Filtered Image', binary)

    # Apply morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)




    def angle_between_vectors(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(frame)
    for cnt in contours:
        # Approximate contour
        # ============================================================
        epsilon = 0.05* cv2.arcLength(cnt, True)   
         # ============================================================
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
                if all(70 <= angle <= 110 for angle in angles):  # Adjust this threshold as needed
                    # Get the center of the rectangle
                    M = cv2.moments(approx)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # Draw the contour and center
                        cv2.drawContours(mask, [approx], -1, (255, 255, 255), thickness=cv2.FILLED)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)
                        found_cube = True
    frame = cv2.bitwise_and(frame, mask)
    if found_cube:
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
                epsilon = 0.05 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                # If the contour has 4 sides, it could be a square
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    # Compute area and check if it's large enough
                    area = cv2.contourArea(approx)
                    if area > 1000 and area < 15000:  # Adjust this threshold as needed
                        # Get the center of the square
                        M = cv2.moments(approx)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            squares.append({'position': (cx, cy), 'color': color_name})
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



        def filter_closest_points(squares):
            # Check if there are at least 6 squares
            if len(squares) < 6:
                raise ValueError("Not enough squares to filter 6-9 closest points")
        
            # Extract positions from the squares dictionary
            positions = np.array([s['position'] for s in squares])
        
            # Calculate the mean point
            mean_point = np.mean(positions, axis=0)
        
            # Filter out points that are further than 400 units from the mean point
            filtered_points = []
            for square in squares:
                distance = np.linalg.norm(np.array(square['position']) - mean_point)
                if distance <= 300:
                    filtered_points.append(square)
        
            # Check if there are at least 6 points within the distance criteria
            if len(filtered_points) >= 6:
                return filtered_points
        
            raise ValueError("No suitable points found within the distance criteria")
        
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

        # Set plot limits and grid
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates
        ax.grid(True)
        plt.draw()
        plt.pause(0.001)

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