import pyvista as pv
import numpy as np
import utils.config as CONFIG
lines = CONFIG.lines

# rotation matrix
def rotation_matrix(pitch, roll, yaw):
    """
    Returns a 3D rotation matrix (intrinsic) for right-multiplication with row vectors.

    Parameters:
    pitch (float): Rotation around the x-axis in radians (elevation).
    roll (float): Rotation around the y-axis in radians (roll).
    yaw (float): Rotation around the z-axis in radians (azimuth).

    Returns:
    np.ndarray: A 3x3 rotation matrix.
    """
    # Rotation around X-axis (pitch/Pitch)
    Rx = np.array([
        [1,  0,                 0],
        [0,  np.cos(pitch), -np.sin(pitch)],
        [0,  np.sin(pitch), np.cos(pitch)]
    ])

    # Rotation around Y-axis (Roll)
    Ry = np.array([
        [np.cos(roll),  0, np.sin(roll)],
        [0,             1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])

    # Rotation around Z-axis (yaw/Yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw),  0],
        [0,               0,                1]
    ])

    # Right multiplication: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def plot_animation(coordinates:np.ndarray, num_frames:int=180, saving_path:str="animation.mp4"):
    """
    Plots an animation of the subject's movements using the provided body keypoint coordinates.

    Parameters:
    coordinates (np.ndarray): body keypoint coordinates, with shape (T, B, 3).

    Returns:
    pyvista plotter object.
    """
    coordinates=coordinates.transpose(1,0,2)# To shape: (num_groups, num_frames, 3)
    # visualize movements of the subject using the body kepoint coordinates
    sampling_rate = 60
    #num_frames = 20 * sampling_rate # video frames
    pitch, roll, yaw = 260, 90, 0  # in degrees
    d_to_r = np.pi/180
    pitch, roll, yaw = pitch*d_to_r, roll*d_to_r, yaw*d_to_r# convert to radian
    R = rotation_matrix(pitch, roll, yaw)
    rotated_coordinates = coordinates[:,:num_frames] @ R
    num_points = 26

    # Convert lines to PyVista format
    line_segments = []
    for line in lines:
        line_segments.append(len(line))  # Number of points in the line (always 2 here)
        line_segments.extend(line)      # Indices of points to connect

    line_segments = np.array(line_segments)

    # Initialize PyVista plotter
    plotter = pv.Plotter()

    # Create a point cloud
    points = rotated_coordinates[:, 0, :]
    point_cloud = pv.PolyData(rotated_coordinates[:, 0, :])
    plotter.add_points(point_cloud, color="blue", point_size=10)

    # Add lines connecting the points
    poly_data = pv.PolyData()
    poly_data.points = points
    poly_data.lines = line_segments
    plotter.add_mesh(poly_data, color="red", line_width=2)

    # Update function for animation
    def update(frame):
        # Update point positions
        point_cloud.points = rotated_coordinates[:, frame, :]  # Update point positions
        plotter.update()
        # Update line positions
        poly_data.points = rotated_coordinates[:, frame, :]
        plotter.update()

    # Start animation
    plotter.open_movie(saving_path, 54)  # Optional: Save as MP4
    plotter.show(auto_close=False)
    for frame in range(num_frames):
        update(frame)
        plotter.write_frame()  # Write each frame to the movie
    plotter.close()