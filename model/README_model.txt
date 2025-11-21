# Model Placement for CoralEdgeTpu Detector

Please place your TensorFlow Lite (`.tflite`) model file in this directory.

The application expects a model named `model.tflite` by default.
You can specify a different model path using the `--model` command-line argument when running the detector:

Example:
  ./build/detector --model path/to/your/model.tflite

Ensure your model is compiled for the Coral Edge TPU.

If this file is missing, the application will exit with an error.
