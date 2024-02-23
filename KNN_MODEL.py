# 1. Extract the feature from all bvh files from the folder "data", each file has around 1000 frames,
# we will now focus on one bvh file or a series of BVH files that belong to a same category
# each joint: 2*3 matrix (this is the first two columns of the first three rows in local transform)
# +3*3 (this should be the AB^(-1) matrix of two 3*3 matrix extracted from the current frame and the next natural
# frame, which is, the first two columns of the first three rows in local transform, plus the fourth column of the first
# three rows in local transform)
# distance with the next natural frame, the second part belongs to control signal, first part
# is always fixed for each joint in each frame
# the final feature should be 1000 (num of frame) * 22 (num of joint) * (6+9) (per joint) (notice that
# the number here may not be the exact one, just for illustration), which is a 2D list (length is the number of frame)
# 2. Write a "KNN" model (where K is equal to 1) that keeps "picking" the next most "similar" frame,
# a special case is when user "input" something,
# at this point we only allow a 2D displacement input, that is plus/minus one unit for X-axis and Y-axis
# How to pick the next most similar frame? the current frame should be 22 * (6+9), it is supposed to stay the same
# since without user input (external affect), it should always pick itself which is most similar to it.
# if there is a user input, for each joint, the last 9 number (the control signal part, not the first 6 number),
# we need to change it in a way such as:
# 1, 0, 1
# 0, 1, 1
# 0, 0, 0
# the matrix above should be the matrix that we need to try adding to the last 9 control signal numbers (one by one)
# when the user press "turn right" and "move forward"
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from anim import bvh, animation

# Feature Extraction
def extract_features(anim):
    num_frames = len(anim)
    num_joints = len(anim.skel.joints)

    # Extract the first two columns of the first three rows in local transform for all frames and joints
    joint_features = anim.local_transform[:num_frames, :num_joints, :3, :2].reshape(num_frames, num_joints, -1)

    # Compute the AB^(-1) matrix for all frames and joints
    current_transform = np.hstack((anim.local_transform[:num_frames, :num_joints, :3, :2], anim.local_transform[:num_frames, :num_joints, :3, 3:4]))
    next_transform = np.roll(current_transform, -1, axis=0)
    next_transform[-1] = current_transform[-1]  # Use the same transform for the last frame
    ab_inv = np.matmul(current_transform, np.linalg.inv(next_transform)).reshape(num_frames, num_joints, -1)

    # Combine the joint features and the AB^(-1) matrix
    features = np.hstack((joint_features, ab_inv))

    return features.tolist()

# KNN Model
class KNNModel:
    def __init__(self, k=1):
        self.knn = NearestNeighbors(n_neighbors=k)

    def fit(self, X):
        self.knn.fit(X)

    def find_similar_frame(self, current_frame, control_signal):
        # Adjust the control signal part of the current frame based on user input
        for i in range(0, len(current_frame), 15):
            current_frame[i+6:i+15] += control_signal.flatten()
        # Find the most similar frame to the current (adjusted) frame
        distances, indices = self.knn.kneighbors([current_frame])
        return indices[0][0]