import pygame
import numpy as np
import bvhio

# Load the BVH file
bvh = bvhio.readAsBvh('data/aiming1_subject1.bvh')

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 640, 480
BLACK = (0, 0, 0)

# Create the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Define a rotation matrix
def rotation_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with black
    screen.fill(BLACK)

    # Iterate through joints
    for joint, index, depth in bvh.Root.layout():
        # Get the position of the joint
        x, y, z = joint.Channels

        # Rotate the point
        x, y, z = np.dot([x, y, z], rotation_matrix(0.001))

        # Project the point onto the 2D screen
        z += 4  # Add a constant to prevent division by zero and keep points on the screen
        x /= z
        y /= z
        projected_point = (WIDTH / 2 + x * WIDTH / 2, HEIGHT / 2 + y * HEIGHT / 2)

        # Draw the point
        pygame.draw.circle(screen, (255, 255, 255), projected_point, 5)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
