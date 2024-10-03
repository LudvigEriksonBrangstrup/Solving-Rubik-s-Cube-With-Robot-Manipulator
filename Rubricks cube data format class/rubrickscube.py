import numpy as np

class RubiksCube:
    def __init__(self):
        """
        Initialize a 9x12 matrix representing the Rubik's Cube.
        All elements are set to None.
        """
        # Define face positions with corrected 'B' position
        self.face_positions = {
            'U': (0, 3),
            'L': (3, 0),
            'F': (3, 3),
            'R': (3, 6),
            'B': (3, 9),  # Adjusted to fit within 9 rows
            'D': (6, 3)
        }
        
        # Initialize a 9x12 matrix filled with None
        self.color_notation_matrix = np.full((9, 12), None, dtype=object)
        self.face_notation_matrix = np.full((9, 12), None, dtype=object)
    
    def print_matrix(self):
        """
        Print the current state of the Rubik's Cube matrix.
        """
        print("Rubik's Cube Matrix (9x12):")
        for row in self.color_notation_matrix:
            formatted_row = ""
            for elem in row:
                if elem is None:
                    formatted_row += "None\t"
                else:
                    formatted_row += f"{elem}\t"
            print(formatted_row)
        print("\n")
    
    def update_face(self, side, new_matrix):
        """
        Update a specific face of the Rubik's Cube with a new 3x3 matrix.
        
        :param side: A string key representing the face ('U', 'L', 'F', 'R', 'B', 'D').
        :param new_matrix: A 3x3 list of lists representing the new state of the face.
        """
        # Validate the side key
        if side not in self.face_positions:
            raise ValueError(f"Invalid side key '{side}'. Valid keys are 'U', 'L', 'F', 'R', 'B', 'D'.")
        
        # Validate the new_matrix dimensions
        if not (isinstance(new_matrix, list) and len(new_matrix) == 3):
            raise ValueError("new_matrix must be a 3x3 list of lists.")
        for row in new_matrix:
            if not (isinstance(row, list) and len(row) == 3):
                raise ValueError("new_matrix must be a 3x3 list of lists.")
        
        # Get the starting position for the specified face
        start_row, start_col = self.face_positions[side]
        
        # Update the matrix with the new face values
        for i in range(3):
            for j in range(3):
                self.color_notation_matrix[start_row + i][start_col + j] = new_matrix[i][j]

    def get_matrix(self):
        """
        Get the current state of the Rubik's Cube matrix.
        
        :return: A copy of the 9x12 Rubik's Cube matrix.
        """
        return self.color_notation_matrix.copy()
    
    def update_face_notation_matrix(self):
        """
        Convert the color notation to the face notation based on the color in the middle position of each face.
        """
        # Create a dictionary to map colors to face notations
        color_to_face = {}
        for face, (start_row, start_col) in self.face_positions.items():
            middle_color = self.color_notation_matrix[start_row + 1][start_col + 1]
            if middle_color is not None:  # Ensure the center piece is not None
                color_to_face[middle_color] = face
        
        # Convert the matrix to face notation
        for row in range(9):
            for col in range(12):
                color = self.color_notation_matrix[row][col]
                if color in color_to_face:
                    self.face_notation_matrix[row][col] = color_to_face[color]
                else:
                    self.face_notation_matrix[row][col] = color
    
    def print_output_matrix(self):
        """
        Print the output matrix which contains the face notation.
        """
        print("Output Matrix (Face Notation):")
        for row in self.face_notation_matrix:
            formatted_row = ""
            for elem in row:
                if elem is None:
                    formatted_row += "None\t"
                else:
                    formatted_row += f"{elem}\t"
            print(formatted_row)
        print("\n")

        
    def get_face_notation_string(self):
        """
        Get the face notation string in the format UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB.
        
        :return: A string representing the face notation.
        """
        face_order = ['U', 'R', 'F', 'D', 'L', 'B']
        face_notation_string = ""
        
        for face in face_order:
            start_row, start_col = self.face_positions[face]
            for i in range(3):
                for j in range(3):
                    face_notation_string += self.face_notation_matrix[start_row + i][start_col + j] or 'None'
        
        return face_notation_string

def main():
    # Initialize the Rubik's Cube
    cube = RubiksCube()
    
    # Define a new 3x3 matrix for the Front ('F') face
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
    
    # Define a new 3x3 matrix for the Down ('D') face
    new_down = [
        ['y', 'y', 'g'],
        ['y', 'y', 'y'],
        ['y', 'r', 'y']
    ]
    
    # Update the Down face
    cube.update_face('D', new_down)


    # new_front = [
    #     ['r', 'r', 'r'],
    #     ['r', 'r', 'r'],
    #     ['r', 'r', 'r']
    # ]
    
    # # Update the Front face
    # cube.update_face('F', new_front)
    
    # # Define a new 3x3 matrix for the Up ('U') face
    # new_up = [
    #     ['w', 'w', 'w'],
    #     ['w', 'w', 'w'],
    #     ['w', 'w', 'w']
    # ]
    
    # # Update the Up face
    # cube.update_face('U', new_up)
    
    # # Define a new 3x3 matrix for the Left ('L') face
    # new_left = [
    #     ['g', 'g', 'g'],
    #     ['g', 'g', 'g'],
    #     ['g', 'g', 'g']
    # ]
    
    # # Update the Left face
    # cube.update_face('L', new_left)
    
    # # Define a new 3x3 matrix for the Right ('R') face
    # new_right = [
    #     ['b', 'b', 'b'],
    #     ['b', 'b', 'b'],
    #     ['b', 'b', 'b']
    # ]
    
    # # Update the Right face
    # cube.update_face('R', new_right)
    
    # # Define a new 3x3 matrix for the Back ('B') face
    # new_back = [
    #     ['o', 'o', 'o'],
    #     ['o', 'o', 'o'],
    #     ['o', 'o', 'o']
    # ]
    
    # # Update the Back face
    # cube.update_face('B', new_back)
    
    # # Define a new 3x3 matrix for the Down ('D') face
    # new_down = [
    #     ['y', 'y', 'y'],
    #     ['y', 'y', 'y'],
    #     ['y', 'y', 'y']
    # ]
    
    # # Update the Down face
    # cube.update_face('D', new_down)
    
    
    # Print the updated matrix
    print("Rubik's Cube Matrix after updating all faces:")
    cube.print_matrix()

    # Convert color notation to face notation
    cube.update_face_notation_matrix()

    # Print the output matrix
    cube.print_output_matrix()

    print("Face Notation String:")
    print(cube.get_face_notation_string())

if __name__ == "__main__":
    main()



