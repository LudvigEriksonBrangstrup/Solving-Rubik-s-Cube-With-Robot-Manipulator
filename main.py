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
rubiks_cube.update_face('F', new_front)


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
rubiks_cube.update_face('U', new_up)

# Define a new 3x3 matrix for the Left ('L') face
new_left = [
    ['g', 'g', 'r'],
    ['g', 'g', 'g'],
    ['g', 'y', 'g']
]

# Update the Left face
rubiks_cube.update_face('L', new_left)

# Define a new 3x3 matrix for the Right ('R') face
new_right = [
    ['b', 'b', 'w'],
    ['b', 'b', 'b'],
    ['r', 'b', 'b']
]

# Update the Right face
rubiks_cube.update_face('R', new_right)

# Define a new 3x3 matrix for the Back ('B') face
new_back = [
    ['o', 'o', 'b'],
    ['o', 'o', 'o'],
    ['o', 'w', 'o']
]



# Update the Back face
rubiks_cube.update_face('B', new_back)


new_down = [
    ['y', 'y', 'g'],
    ['y', 'y', 'y'],
    ['y', 'r', 'y']
]

# Update the Down face
rubiks_cube.update_face('D', new_down)


# # Initialize each face with a color (e.g., 'W' for white, 'R' for red, etc.)
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



# UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
# UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
