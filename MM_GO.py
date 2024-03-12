import pygame
import numpy as np
from KNN_MODEL import KNNModel, extract_features
from Pygame_Render import project_3d_to_2d, joint_name_to_idx, window
import os
from anim import bvh
import sys

# Load all BVH files and extract features
# anim = []
# for file in os.listdir("data"):
#     if file.endswith(".bvh"):
#         anim += bvh.load(os.path.join("data", file))
#         features.extend(extract_features(anim))

anim = bvh.load("data/dance1_subject2.bvh")
features = extract_features(anim)
# Initialize and fit the KNN model
knn_model = KNNModel()
# Reshape features from (3945, 22, 15) to (3945, 22*15)
features_2d = features.reshape(features.shape[0], -1)

# Fit the model
knn_model.fit(features_2d)

# Main game loop
for i in range(len(anim)):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            control_signal = None
            if event.key == pygame.K_RIGHT:
                # User input for "turn right"
                control_signal = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]])
            elif event.key == pygame.K_LEFT:
                # User input for "turn left"
                control_signal = np.array([[1, 0, -1], [0, 1, -1], [0, 0, 0]])
            elif event.key == pygame.K_UP:
                # User input for "move forward"
                control_signal = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
            elif event.key == pygame.K_DOWN:
                # User input for "move backward"
                control_signal = np.array([[-1, -1, 0], [-1, -1, 0], [0, 0, 0]])
            else:
                control_signal = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

            if control_signal is not None:
                # Find the most similar frame using the KNN model
                similar_frame_idx = knn_model.find_similar_frame(features_2d[i], control_signal)
                print("how does it look like here?", similar_frame_idx)
                # Update the current frame with the most similar frame
                anim.positions[i] = anim.positions[similar_frame_idx]

    ## Clear the window
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