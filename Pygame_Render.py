import pygame
import numpy as np
from anim import bvh
import sys
from anim.animation import Animation

# Load the BVH file
anim = bvh.load("data/dance1_subject2.bvh")

# Create a mapping from joint names to indices
joint_name_to_idx = {joint.name: i for i, joint in enumerate(anim.skel.joints)}

# Define the dimensions of the window
WIDTH, HEIGHT = 800, 600

# Initialize Pygame
pygame.init()

# Create a window with the defined dimensions
window = pygame.display.set_mode((WIDTH, HEIGHT))

# Function to project 3D coordinates to 2D
def project_3d_to_2d(positions):
    # Calculate the center of the character
    center = np.mean(positions, axis=0)

    # Subtract the center from each position, this is to move the character to the center of the screen
    positions -= center

    fov = 500
    factor = fov / (fov + positions[:, 2])
    x2d = positions[:, 0] * factor + WIDTH / 2
    y2d = -positions[:, 1] * factor + HEIGHT / 2
    return np.column_stack([x2d, y2d])

def render_scene():
    # Main game loop
    for i in range(len(anim)):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Clear the window
        window.fill((0, 0, 0))

        # Get the positions of the joints for this frame
        positions = project_3d_to_2d(anim.positions[i])

        # Draw each joint
        for position in positions:
            pygame.draw.circle(window, (255, 255, 255), position, 5)

        # Draw each "bone"
        for joint in anim.skel.joints:
            if joint.parent is not None and joint.parent != -1:  # Check if the joint has a valid parent and the parent is not -1
                start_idx = joint_name_to_idx[joint.name]
                end_idx = joint.parent  # Use the parent index directly
                start = positions[start_idx]
                end = positions[end_idx]
                pygame.draw.line(window, (255, 0, 0), start, end, 2)  # The last parameter '2' is the thickness of the line

        # Update the window
        pygame.display.flip()

        # Delay to control the speed of the animation
        pygame.time.delay(1000 // anim.fps)

if __name__ == "__main__":
    render_scene()