import numpy as np

class RubiksCube:
    def __init__(self):
        """
        Initialize the Rubik's Cube with each face represented as a 3x3 matrix.
        All elements are set to None initially.
        """
        # Define the six faces of the cube
        self.faces = {
            'U': np.full((3, 3), None, dtype=object),
            'L': np.full((3, 3), None, dtype=object),
            'F': np.full((3, 3), None, dtype=object),
            'R': np.full((3, 3), None, dtype=object),
            'B': np.full((3, 3), None, dtype=object),
            'D': np.full((3, 3), None, dtype=object)
        }

        self.face_positions = {
            'U': (0, 3),
            'L': (3, 0),
            'F': (3, 3),
            'R': (3, 6),
            'B': (3, 9),
            'D': (6, 3)
        }
        
        # Initialize a separate face notation matrix
        self.face_notation_matrix = np.full((9, 12), None, dtype=object)
    
    def print_matrix(self) -> None:
        """
        Print the current state of the Rubik's Cube in a 9x12 matrix format.
        """
        print("Rubik's Cube Matrix (9x12):")
        # Map each face to its position in the 9x12 matrix
        
        # Create a 9x12 matrix filled with "None"
        display_matrix = np.full((9, 12), "None", dtype=object)
        
        # Populate the display_matrix with face colors
        for face, (start_row, start_col) in self.face_positions.items():
            face_matrix = self.faces[face]
            for i in range(3):
                for j in range(3):
                    color = face_matrix[i][j]
                    display_matrix[start_row + i][start_col + j] = color if color is not None else "None"
        
        # Print the matrix
        for row in display_matrix:
            print("\t".join(row))
        print("\n")
    
    def update_face(self, side: str, new_matrix: list) -> None:
        """
        Update a specific face of the Rubik's Cube with a new 3x3 matrix.
        
        :param side: A string key representing the face ('U', 'L', 'F', 'R', 'B', 'D').
        :param new_matrix: A 3x3 list of lists representing the new state of the face.
        """
        # Validate the side key
        if side not in self.faces:
            raise ValueError(f"Invalid side key '{side}'. Valid keys are 'U', 'L', 'F', 'R', 'B', 'D'.")
        
        # Validate the new_matrix dimensions
        if not (isinstance(new_matrix, list) and len(new_matrix) == 3 and all(isinstance(row, list) and len(row) == 3 for row in new_matrix)):
            raise ValueError("new_matrix must be a 3x3 list of lists.")
        
        # Update the face with the new values
        self.faces[side] = np.array(new_matrix, dtype=object)
    
    def get_matrix(self) -> dict:
        """
        Get a copy of the current state of the Rubik's Cube faces.
        
        :return: A dictionary containing copies of each face's 3x3 matrix.
        """
        return {face: self.faces[face].copy() for face in self.faces}
    
    def update_face_notation_matrix(self) -> None:
        """
        Convert the color notation to the face notation based on the color in the middle position of each face.
        """
        # Map each color to its corresponding face based on the center color
        color_to_face = {}
        for face, matrix in self.faces.items():
            middle_color = matrix[1][1]
            if middle_color is not None:
                color_to_face[middle_color] = face
    
        # Reset the face_notation_matrix
        self.face_notation_matrix[:] = "None"
    
        # Populate the face_notation_matrix with face notations
        for face, (start_row, start_col) in self.face_positions.items():
            for i in range(3):
                for j in range(3):
                    color = self.faces[face][i][j]
                    if color is not None:
                        self.face_notation_matrix[start_row + i][start_col + j] = color_to_face.get(color, color)
                    else:
                        self.face_notation_matrix[start_row + i][start_col + j] = "None"
    
    def print_output_matrix(self) -> None:
        """
        Print the output matrix which contains the face notation.
        """
        print("Output Matrix (Face Notation):")
        for row in self.face_notation_matrix:
            print("\t".join([elem if elem is not None else "None" for elem in row]))
        print("\n")
    
    def get_face_notation_string(self) -> str:
        """
        Get the face notation string in the format UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB.
        
        :return: A string representing the face notation.
        """
        face_order = ['U', 'R', 'F', 'D', 'L', 'B']
        face_notation_string = ""
        
        for face in face_order:
            matrix = self.faces[face]
            for row in matrix:
                for color in row:
                    face_notation_string += color if color is not None else 'None'
        
        return face_notation_string

# Example Usage
if __name__ == "__main__":
    cube = RubiksCube()
    
    # Initialize each face with a color (e.g., 'W' for white, 'R' for red, etc.)
    # cube.update_face('U', [['w']*3 for _ in range(3)])
    # cube.update_face('L', [['o']*3 for _ in range(3)])
    # cube.update_face('F', [['g']*3 for _ in range(3)])
    # cube.update_face('R', [['r']*3 for _ in range(3)])
    # cube.update_face('B', [['b']*3 for _ in range(3)])
    # cube.update_face('D', [['y']*3 for _ in range(3)])

    # # Print the current matrix
    # cube.print_matrix()
    
    # # Update a specific face
    new_front = [
        ['r', 'r', 'g'],
        ['r', 'g', 'r'],
        ['b', 'r', 'r']
    ]
    cube.update_face('F', new_front)


    new_front = [
        ['r', 'r', 'g'],
        ['r', 'r', 'r'],
        ['b', 'r', 'r']
    ]
    
    # Update the Front face
    cube.update_face('F', new_front)
    
    # Define a new 3x3 matrix for the Up ('U') face
    new_up = [
        ['w', 'w', 'o'],
        ['w', 'w', 'w'],
        ['w', 'b', 'w']
    ]
    
    # Update the Up face
    cube.update_face('U', new_up)
    
    # Define a new 3x3 matrix for the Left ('L') face
    new_left = [
        ['g', 'g', 'r'],
        ['g', 'g', 'g'],
        ['g', 'y', 'g']
    ]
    
    # Update the Left face
    cube.update_face('L', new_left)
    
    # Define a new 3x3 matrix for the Right ('R') face
    new_right = [
        ['b', 'b', 'w'],
        ['b', 'b', 'b'],
        ['r', 'b', 'b']
    ]
    
    # Update the Right face
    cube.update_face('R', new_right)
    
    # Define a new 3x3 matrix for the Back ('B') face
    new_back = [
        ['o', 'o', 'b'],
        ['o', 'o', 'o'],
        ['o', 'w', 'o']
    ]
    
    # Update the Back face
    cube.update_face('B', new_back)

    new_down = [
        ['y', 'y', 'g'],
        ['y', 'y', 'y'],
        ['y', 'r', 'y']
    ]

    # Update the Down face
    cube.update_face('D', new_down)
    
    # Update and print face notation matrix
    cube.update_face_notation_matrix()
    cube.print_output_matrix()
    
    # Get face notation string
    notation_str = cube.get_face_notation_string()
    print(f"Face Notation String: {notation_str}")