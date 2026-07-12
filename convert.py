import tensorflow as tf

# Aapka trained h5 model load kar raha hai
model = tf.keras.models.load_model('models/best_plant_model.h5')

# Model ko TFLite format mein convert kar raha hai
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Nayi halki file save kar raha hai
with open('models/model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Mubarak ho! Model successfully TFLite mein convert ho gaya hai.")