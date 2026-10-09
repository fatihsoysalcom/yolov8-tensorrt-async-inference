# YOLOv8 TensorRT Async Inference

This Python script demonstrates asynchronous inference with YOLOv8 using NVIDIA TensorRT. It loads a pre-built TensorRT engine, preprocesses an input image, enqueues it for asynchronous processing on the GPU, and then retrieves and postprocesses the detection results.

## Language

`python`

## How to Run

1. Ensure you have a TensorRT engine file (e.g., 'yolov8.engine') generated from a YOLOv8 ONNX model.
2. Replace 'yolov8.engine' in the script with your engine path.
3. Run the script using: python yolov8_tensorrt_async.py

## Original Article

This example accompanies the Turkish article: [TensorRT ile YOLOv8: Asenkron Çıkarım Yolculuğu](https://fatihsoysal.com/blog/tensorrt-ile-yolov8-asenkron-cikarim-yolculugu/).

## License

MIT — see [LICENSE](LICENSE).
