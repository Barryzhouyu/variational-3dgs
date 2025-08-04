import numpy as np
import json

####################hope_2#######################
# R = np.array([
#     [  0.9999999573921126,  0.0002306120443263246,     0.00017898004986157168 ],
#     [ -0.00023060012195533902,  0.9999999711919711,  -0.0000666306267123088 ],
#     [ -0.00017899541053055014,  0.00006658935105199292,  0.9999999817632504 ]
# ])

# tvec = np.array([
#     -0.2739604664189615,
#     -0.14746068715627195,
#      2.2302300866574605
# ])

####################real#######################
R = np.array([
    [  0.9989228057828026,   -0.004827419463308733,   -0.04615110083560933 ],
    [  0.005552166751138264,  0.9998630763176963,      0.015588523370739044 ],
    [  0.04606952931581641,  -0.015827970111097336,    0.9988128322317354 ]
])

tvec = np.array([
    -1.1111197064232674,
    -0.7743149563547892,
     1.0056213983535645
])



transform = np.eye(4)
transform[:3, :3] = R  # Set rotation
transform[:3, 3] = tvec  # Set translation


############rov example 1, three cones##############


# fx = 960.08227651053789
# fy = 960.08227651053789
# width = 1920
# height = 1280
# cx = width / 2
# cy = height / 2


##########################
#real data
fx = 1068.920574552045
fy = 1068.920574552045
width = 1920
height = 1080
cx = width / 2
cy = height / 2


##########################
#hope_2' data
# fx = 3576.8065146501135
# fy = 3576.8065146501135
# width = 1920
# height = 1280
# cx = width / 2
# cy = height / 2


# 🔹 Build JSON Structure
cameras_json = {
    "camera_intrinsics": {
        "1": {
            "width": width,
            "height": height,
            "focal_length": [fx, fy],
            "principal_point": [cx, cy]
        }
    },
    "frames": [
        {
            "file_path": "/home/roar3/Desktop/custom_pose.jpg",
            "transform_matrix": transform.tolist(),
            "camera_id": 1
        }
    ]
}

# 🔹 Save JSON File
with open("/home/roar3/Desktop/guessed_pose.json", "w") as f:
    json.dump(cameras_json, f, indent=4)

print("Saved /home/roar3/Desktop/guessed_pose.json")

