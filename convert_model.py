import tensorflow as tf

print("Loading old model...")

model = tf.keras.models.load_model(
    "best_lane_detection_model.h5",
    compile=False
)

print("Saving new model...")

model.save("lane_model.keras")

print("Done!")