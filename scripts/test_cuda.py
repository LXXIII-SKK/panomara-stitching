import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("\n--- GPU CHECK ---")
print("Num GPUs Available: ", len(gpus))

if gpus:
    for gpu in gpus:
        print("GPU Name:", gpu.name)
    print("Success! TensorFlow is using your GPU.")
else:
    print("TensorFlow cannot find a GPU. Check your NVIDIA drivers.")
print("-----------------\n")