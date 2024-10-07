from Rubiks_cube_class.rubickscube import RubiksCube

rubiks_cube = RubiksCube()

# Initialize each face with a color (e.g., 'W' for white, 'R' for red, etc.)
# cube.update_face('U', [['w']*3 for _ in range(3)])
# cube.update_face('L', [['o']*3 for _ in range(3)])
# cube.update_face('F', [['g']*3 for _ in range(3)])
# cube.update_face('R', [['r']*3 for _ in range(3)])
# cube.update_face('B', [['b']*3 for _ in range(3)])
# cube.update_face('D', [['y']*3 for _ in range(3)])

# # Print the current matrix
# cube.print_matrix()



# new_front = [
#     ['r', 'r', 'g'],
#     ['r', 'r', 'r'],
#     ['b', 'r', 'r']
# ]

# rubiks_cube.update_face('F', new_front)

# new_up = [
#     ['w', 'w', 'o'],
#     ['w', 'w', 'w'],
#     ['w', 'b', 'w']
# ]

# rubiks_cube.update_face('U', new_up)

# new_right = [
#     ['b', 'b', 'w'],
#     ['b', 'b', 'b'],
#     ['r', 'b', 'b']
# ]

# rubiks_cube.update_face('R', new_right)

# new_back = [
#     ['o', 'o', 'b'],
#     ['o', 'o', 'o'],
#     ['o', 'w', 'o']
# ]














# # Update a specific face
new_front = [
    ['r', 'r', 'g'],
    ['r', 'g', 'r'],
    ['b', 'r', 'r']
]
rubiks_cube.update_face('U', new_front)


new_front = [
    ['r', 'r', 'g'],
    ['r', 'r', 'r'],
    ['b', 'r', 'r']
]

# Update the Front face
rubiks_cube.update_face('F', new_front)

# Define a new 3x3 matrix for the Up ('U') face
new_up = [
    ['w', 'w', 'o'],
    ['w', 'w', 'w'],
    ['w', 'b', 'w']
]

# Update the Up face
rubiks_cube.update_face('R', new_up)

# Define a new 3x3 matrix for the Left ('L') face
new_left = [
    ['g', 'g', 'r'],
    ['g', 'g', 'g'],
    ['g', 'y', 'g']
]

# Update the Left face
rubiks_cube.update_face('B', new_left)

# Define a new 3x3 matrix for the Right ('R') face
new_right = [
    ['b', 'b', 'w'],
    ['b', 'b', 'b'],
    ['r', 'b', 'b']
]

# Update the Right face
rubiks_cube.update_face('L', new_right)

# Define a new 3x3 matrix for the Back ('B') face
new_back = [
    ['o', 'o', 'b'],
    ['o', 'o', 'o'],
    ['o', 'w', 'o']
]



# Update the Back face
rubiks_cube.update_face('D', new_back)




# Initialize each face with a color (e.g., 'W' for white, 'R' for red, etc.)
# rubiks_cube.update_face('U', [['w']*3 for _ in range(3)])
# rubiks_cube.update_face('L', [['o']*3 for _ in range(3)])
# rubiks_cube.update_face('F', [['r']*3 for _ in range(3)])
# rubiks_cube.update_face('R', [['g']*3 for _ in range(3)])
# rubiks_cube.update_face('B', [['b']*3 for _ in range(3)])
# rubiks_cube.update_face('D', [['y']*3 for _ in range(3)])


rubiks_cube.print_matrix()

# Update and print face notation matrix
rubiks_cube.update_face_notation_matrix()
rubiks_cube.print_output_matrix()

# Get face notation string
notation_str = rubiks_cube.get_face_notation_string()
print(f"Face Notation String: {notation_str}")

class Robot:
    def rotatecube_up(self):
        pass

    def rotatecube_down(self):
        pass

    def rotatecube_right(self):
        pass

class Scanner:
    def scan_face(self) -> list:
        return [
            ['w', 'w', 'w'],
            ['w', 'w', 'w'],
            ['w', 'w', 'w']
        ]


scanner = Scanner()

robot = Robot()

def read_cube():
    # Scan the upper face
    upp_face = scanner.scan_face()
    hej = scanner.scan_face()
    rubiks_cube.update_face('U', upp_face)

    # Rotate the cube to scan the front face
    robot.rotatecube_down()
    front_face = scanner.scan_face()
    rubiks_cube.update_face('F', front_face)

    # Rotate the cube to scan the right face
    robot.rotatecube_right()
    right_face = scanner.scan_face()
    rubiks_cube.update_face('R', right_face)

    # Rotate the cube to scan the back face
    robot.rotatecube_right()
    back_face = scanner.scan_face()
    rubiks_cube.update_face('B', back_face)

    # Rotate the cube to scan the left face
    robot.rotatecube_right()
    left_face = scanner.scan_face()
    rubiks_cube.update_face('L', left_face)

    # Rotate the cube to scan the down face
    robot.rotatecube_right()
    robot.rotatecube_down()
    down_face = scanner.scan_face()
    rubiks_cube.update_face('D', down_face)

    # Rotate the cube so front face is facing up
    robot.rotatecube_up()









# UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
# UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB