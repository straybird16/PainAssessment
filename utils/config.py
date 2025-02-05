# configuration parameters for Miscellaneous EmoPain dataset usage

# activity columns
activity_cols = ['One Leg Stand 1',
 'One Leg Stand 2',
 'One Leg Stand 3',
 'One Leg Stand 4',
 'One Leg Stand 5',
 'One Leg Stand 6',
 'Sitting Still',
 'Reach Forward 2',
 'Reach Forward 1',
 'Sit to Stand Instructed 1',
 'Sit to Stand Instructed 2',
 'Sit to Stand Instructed 3',
 'Stand to Sit Instructed 1',
 'Stand to Sit Instructed 2',
 'Stand to Sit Instructed 3',
 'Standing Still',
 'Sit to Stand Not Instructed',
 'Stand to Sit Not Instructed',
 'Bend 2',
 'Bend 1',
 'Walk',
 'Other Major: Bend to pick up',
 'Sit to Stand Instructed 4',
 'Stand to Sit Instructed 4']

_joint_to_index_mapping = p = {'Hip': 0,
 'LeftUpperLeg': 1,
 'LeftLowerLeg': 2,
 'LeftAnkle': 3,
 'LeftHeel': 4,
 'LeftToes': 5,
 'RightUpperLeg': 6,
 'RightLowerLeg': 7,
 'RightAnkle': 8,
 'RightHeel': 9,
 'RightToes': 10,
 'Spine': 11,
 'Spine 1': 12,
 'LeftShoulder': 13,
 'LeftUpperArm': 14,
 'LeftLowerArm': 15,
 'LeftWrist': 16,
 'LeftFingertip': 17,
 'RightShoulder': 18,
 'RightUpperArm': 19,
 'RightArm': 20,
 'RightWrist': 21,
 'RightFingertip': 22,
 'Neck': 23,
 'Head': 24,
 'Crown': 25,}

# Define lines to connect points (indices of the points to connect)
lines = [
    # Head and Neck
    (p['Crown'], p['Head']),
    (p['Head'], p['Neck']),

    # Spine
    (p['Neck'], p['Spine 1']),
    (p['Spine 1'], p['Spine']),
    (p['Spine'], p['Hip']),

    # Left Arm
    (p['Neck'], p['LeftShoulder']),
    (p['LeftShoulder'], p['LeftUpperArm']),
    (p['LeftUpperArm'], p['LeftLowerArm']),
    (p['LeftLowerArm'], p['LeftWrist']),
    (p['LeftWrist'], p['LeftFingertip']),

    # Right Arm
    (p['Neck'], p['RightShoulder']),
    (p['RightShoulder'], p['RightUpperArm']),
    (p['RightUpperArm'], p['RightArm']),
    (p['RightArm'], p['RightWrist']),
    (p['RightWrist'], p['RightFingertip']),

    # Left Leg
    (p['Hip'], p['LeftUpperLeg']),
    (p['LeftUpperLeg'], p['LeftLowerLeg']),
    (p['LeftLowerLeg'], p['LeftAnkle']),
    (p['LeftAnkle'], p['LeftHeel']),
    (p['LeftHeel'], p['LeftToes']),

    # Right Leg
    (p['Hip'], p['RightUpperLeg']),
    (p['RightUpperLeg'], p['RightLowerLeg']),
    (p['RightLowerLeg'], p['RightAnkle']),
    (p['RightAnkle'], p['RightHeel']),
    (p['RightHeel'], p['RightToes'])
]