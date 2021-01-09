import torch
import numpy as np
BONE_LEVEL = ['carpal', 'met', 'pro', 'int', 'dis', 'end']

BONE_TO_JOINT_NAME = {
    0: "carpal",

    1: "met1",
    2: "pro1",
    3: "dis1",

    4: "met2",
    5: "pro2",
    6: "int2",
    7: "dis2",

    8: "met3",
    9: "pro3",
    10: "int3",
    11: "dis3",

    12: "met4",
    13: "pro4",
    14: "int4",
    15: "dis4",

    16: "met5",
    17: "pro5",
    18: "int5",
    19: "dis5",
}

JOINT_ID_NAME_DICT = {
    0: "carpal",
    1: "met1",
    2: "pro1",
    3: "dis1",
    4: "dis1_end",

    5: "met2",
    6: "pro2",
    7: "int2",
    8: "dis2",
    9: "dis2_end",

    10: "met3",
    11: "pro3",
    12: "int3",
    13: "dis3",
    14: "dis3_end",

    15: "met4",
    16: "pro4",
    17: "int4",
    18: "dis4",
    19: "dis4_end",

    20: "met5",
    21: "pro5",
    22: "int5",
    23: "dis5",
    24: "dis5_end"
}

JOINT_NAME_ID_DICT = {
    "carpal": 0,
    "met1": 1,
    "pro1": 2,
    "dis1": 3,
    "dis1_end": 4,

    "met2": 5,
    "pro2": 6,
    "int2": 7,
    "dis2": 8,
    "dis2_end": 9,

    "met3": 10,
    "pro3": 11,
    "int3": 12,
    "dis3": 13,
    "dis3_end": 14,

    "met4": 15,
    "pro4": 16,
    "int4": 17,
    "dis4": 18,
    "dis4_end": 19,

    "met5": 20,
    "pro5": 21,
    "int5": 22,
    "dis5": 23,
    "dis5_end": 24
}

TEMPLATE_BONE_NAME_ID_DICT = {
    "carpal": 0,
    "thumb1": 1,
    "thumb2": 2,
    "thumb3": 3,
    "index1": 4,
    "index2": 5,
    "index3": 6,
    "index4": 7,
    "middle1": 8,
    "middle2": 9,
    "middle3": 10,
    "middle4": 11,
    "ring1": 12,
    "ring2": 13,
    "ring3": 14,
    "ring4": 15,
    "pinky1": 16,
    "pinky2": 17,
    "pinky3": 18,
    "pinky4": 19,
}

JOINT_LINKS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8, 9],
    [0, 10, 11, 12, 13, 14],
    [0, 15, 16, 17, 18, 19],
    [0, 20, 21, 22, 23, 24]
]

JOINT_PARENT_ID_DICT = {
    0: -1,
    1: 0,
    2: 1,
    3: 2,
    4: 3,

    5: 0,
    6: 5,
    7: 6,
    8: 7,
    9: 8,

    10: 0,
    11: 10,
    12: 11,
    13: 12,
    14: 13,

    15: 0,
    16: 15,
    17: 16,
    18: 17,
    19: 18,

    20: 0,
    21: 20,
    22: 21,
    23: 22,
    24: 23
}

JOINT_CHILD_ID_DICT = {
    0: [1, 5, 10, 15, 20],
    1: [2],
    2: [3],
    3: [4],
    4: [],

    5: [6],
    6: [7],
    7: [8],
    8: [9],
    9: [],

    10: [11],
    11: [12],
    12: [13],
    13: [14],
    14: [],

    15: [16],
    16: [17],
    17: [18],
    18: [19],
    19: [],

    20: [21],
    21: [22],
    22: [23],
    23: [24],
    24: []
}

STATIC_JOINT_NUM = len(JOINT_ID_NAME_DICT.keys())
STATIC_BONE_NUM = len(BONE_TO_JOINT_NAME.keys())

STATIC_AXIS_NUM = 23

BONE_ID_AXIS_DICT = {
    0: [],
    1: [0, 1],
    2: [2, 3],
    3: [4],
    4: [],
    5: [5, 6],
    6: [7],
    7: [8],
    8: [],
    9: [9, 10],
    10: [11],
    11: [12],
    12: [13],
    13: [14, 15],
    14: [16],
    15: [17],
    16: [18],
    17: [10, 20],
    18: [21],
    19: [22],
}


DEVICE = torch.device(type='cpu')

# BONE_TO_JOINT_ID = {}
# for key in BONE_TO_JOINT_NAME:
#     BONE_TO_JOINT_ID[key] = JOINT_NAME_ID_DICT[BONE_TO_JOINT_NAME[key]]

JOINT_ID_BONE_DICT = {}
JOINT_ID_BONE = np.zeros(STATIC_BONE_NUM)
for key in JOINT_ID_NAME_DICT:
    value = JOINT_ID_NAME_DICT[key]
    for key_b in BONE_TO_JOINT_NAME:
        if BONE_TO_JOINT_NAME[key_b] == value:
            JOINT_ID_BONE_DICT[key] = key_b
            JOINT_ID_BONE[key_b] = key

if __name__ == "__main__":
    # print(JOINT_PARENT_ID_DICT)
    # print(JOINT_CHILD_ID_DICT)

    print(JOINT_ID_BONE_DICT)
    print(JOINT_ID_BONE)
