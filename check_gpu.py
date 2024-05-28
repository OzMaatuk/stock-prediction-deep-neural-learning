import tensorflow as tf

def check_gpu_availability():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        print("GPU(s) available:")
        for gpu in gpus:
            print(f"- {gpu.name}")
    else:
        print("No GPU available. Using CPU instead.")

if __name__ == "__main__":
    check_gpu_availability()
