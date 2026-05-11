import tensorflow as tf

# Load .h5 model
model = tf.keras.models.load_model(
    "best_lane_detection_model.h5"
)

# Convert model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# Save TFLite model
with open("lane_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created successfully!")