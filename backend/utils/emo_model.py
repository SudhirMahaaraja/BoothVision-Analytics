import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import caffe

# Define the root directory containing all emotion folders
root_dir = "backend/dataseta"  # Replace with your actual directory path

# List of emotion folders
folders = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Gather all image paths and labels
image_paths = []
labels = []

for idx, folder in enumerate(folders):
    folder_path = os.path.join(root_dir, folder)
    for file in os.listdir(folder_path):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(folder_path, file))
            labels.append(idx)  # Assign numeric labels for each emotion folder

# Optional: Create a DataFrame to mimic `labels.csv` for debugging
data = pd.DataFrame({"path": image_paths, "label": labels})

# Split the data into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# Define directories for LMDB generation
train_lmdb = 'backend/models/emotion_detection'
val_lmdb = 'backend/models/emotion_detection'

if not os.path.exists(train_lmdb):
    os.makedirs(train_lmdb)
if not os.path.exists(val_lmdb):
    os.makedirs(val_lmdb)

# Define a helper function to create LMDBs
def create_lmdb(image_paths, labels, lmdb_path, caffe=None, array_to_datum=None):
    from caffe.io import array_to_datum
    import lmdb
    import cv2

    map_size = len(image_paths) * 3 * 96 * 96  # Estimated LMDB size
    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (96, 96))
            img = img.transpose(2, 0, 1)  # Convert to CHW format

            datum = array_to_datum(img, int(label))
            txn.put(f'{i:08d}'.encode(), datum.SerializeToString())

            if i % 100 == 0:
                print(f"Processed {i}/{len(image_paths)} images for LMDB: {lmdb_path}")

# Create LMDBs
print("Creating Training LMDB...")
create_lmdb(train_paths, train_labels, train_lmdb)
print("Creating Validation LMDB...")
create_lmdb(val_paths, val_labels, val_lmdb)

# Define solver and network prototxt files
solver_prototxt = "solver.prototxt"
train_val_prototxt = "train_val.prototxt"

# Define Network architecture (Prototxt)
with open(train_val_prototxt, 'w') as f:
    f.write('''name: "EmotionDetection"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include { phase: TRAIN }
  transform_param { scale: 0.00390625 }
  data_param {
    source: "train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include { phase: TEST }
  transform_param { scale: 0.00390625 }
  data_param {
    source: "val_lmdb"
    batch_size: 32
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param { lr_mult: 1 }
  param { lr_mult: 2 }
  convolution_param {
    num_output: 32
    kernel_size: 3
    stride: 1
    weight_filler { type: "xavier" }
    bias_filler { type: "constant" }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "pool1"
  top: "fc1"
  param { lr_mult: 1 }
  param { lr_mult: 2 }
  inner_product_param {
    num_output: 256
    weight_filler { type: "xavier" }
    bias_filler { type: "constant" }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "fc1"
  top: "fc1"
}
layer {
  name: "fc2"
  type: "InnerProduct"
  bottom: "fc1"
  top: "fc2"
  param { lr_mult: 1 }
  param { lr_mult: 2 }
  inner_product_param {
    num_output: 8
    weight_filler { type: "xavier" }
    bias_filler { type: "constant" }
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "fc2"
  bottom: "label"
  top: "loss"
}
''')

# Define Solver (Prototxt)
with open(solver_prototxt, 'w') as f:
    f.write('''net: "train_val.prototxt"
base_lr: 0.01
gamma: 0.1
stepsize: 10000
lr_policy: "step"
max_iter: 50000
momentum: 0.9
weight_decay: 0.0005
solver_mode: GPU
display: 100
snapshot: 5000
snapshot_prefix: "emotion_detection"
''')

# Train the model
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver(solver_prototxt)
solver.solve()

# Save the trained model
model_path = "emotion_detection_final.caffemodel"
solver.net.save(model_path)
print(f"Model saved to {model_path}")
