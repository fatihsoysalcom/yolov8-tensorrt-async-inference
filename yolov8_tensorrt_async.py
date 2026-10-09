import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import cv2

# --- Configuration ---
# Replace with your actual TensorRT engine path
# You would typically generate this from a YOLOv8 ONNX model using TensorRT's tools
TRT_ENGINE_PATH = "yolov8.engine"

# Input image dimensions (should match your engine's input shape)
INPUT_H = 640
INPUT_W = 640

# Class names for YOLOv8 (example)
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- TensorRT Initialization ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(
            engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings
        bindings.append(int(device_mem))
        # Append to the appropriate list (inputs or outputs)
        if engine.binding_is_input(binding):
            inputs.append({
                "host": host_mem,
                "device": device_mem,
                "name": binding
            })
        else:
            outputs.append({
                "host": host_mem,
                "device": device_mem,
                "name": binding
            })
    return inputs, outputs, bindings, stream

# --- Inference Functions ---
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Resize and pad to match input dimensions
    ratio = min(INPUT_W / img.shape[1], INPUT_H / img.shape[0])
    new_w, new_h = int(img.shape[1] * ratio), int(img.shape[0] * ratio)
    resized_img = cv2.resize(img, (new_w, new_h))

    padded_img = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8) # BGR padding color
    pad_x = (INPUT_W - new_w) // 2
    pad_y = (INPUT_H - new_h) // 2
    padded_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img

    # Normalize to [0, 1] and transpose to NCHW format
    input_tensor = padded_img.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1)) # HWC to CHW
    input_tensor = np.expand_dims(input_tensor, axis=0) # Add batch dimension

    return input_tensor, img, (pad_x, pad_y, ratio)

def postprocess_detections(outputs, input_shape, original_image_shape, padding_info):
    # YOLOv8 outputs are typically in the shape [batch_size, num_detections, 5 + num_classes]
    # where 5 + num_classes is [x_center, y_center, width, height, confidence, class_scores...]
    # The exact output format can vary slightly based on the export process.
    # This postprocessing assumes a common format.

    # Assuming the first output tensor contains the detections
    detections = outputs[0]
    batch_size, num_boxes, _ = detections.shape

    # Filter out low-confidence detections and apply non-maximum suppression (NMS)
    # This part is simplified; a full NMS implementation would be more complex.
    # For demonstration, we'll just filter by confidence.

    # Unpack padding and ratio info
    pad_x, pad_y, ratio = padding_info
    original_w, original_h = original_image_shape

    results = []
    for i in range(batch_size):
        for j in range(num_boxes):
            box_data = detections[i, j]
            confidence = box_data[4]

            if confidence > 0.5: # Confidence threshold
                class_id = np.argmax(box_data[5:])
                class_score = box_data[5 + class_id]

                if class_score > 0.5: # Class score threshold
                    # Decode bounding box from center_x, center_y, width, height
                    center_x, center_y, width, height = box_data[:4]

                    # Convert normalized box coordinates to original image coordinates
                    # Adjust for padding and scaling
                    x_center_abs = (center_x - pad_x) / (INPUT_W / ratio)
                    y_center_abs = (center_y - pad_y) / (INPUT_H / ratio)
                    width_abs = width / (INPUT_W / ratio)
                    height_abs = height / (INPUT_H / ratio)

                    x1 = max(0, int(x_center_abs - width_abs / 2))
                    y1 = max(0, int(y_center_abs - height_abs / 2))
                    x2 = min(original_w, int(x_center_abs + width_abs / 2))
                    y2 = min(original_h, int(y_center_abs + height_abs / 2))

                    results.append({
                        "box": [x1, y1, x2, y2],
                        "score": float(class_score),
                        "class_id": int(class_id),
                        "class_name": CLASS_NAMES[class_id]
                    })
    return results

# --- Asynchronous Inference ---
class AsyncInferencer:
    def __init__(self, engine_path, input_shape=(3, INPUT_H, INPUT_W)):
        self.engine = load_engine(engine_path)
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        self.input_shape = input_shape
        self.context = self.engine.create_execution_context()
        self.request_count = 0
        self.results_queue = []

    def enqueue_inference(self, image_path):
        # Preprocess image and copy to host buffer
        input_tensor, original_img, padding_info = preprocess_image(image_path)
        np.copyto(self.inputs[0]["host"], input_tensor.ravel())

        # Enqueue asynchronous execution
        self.context.execute_async_v2(bindings=self.bindings, stream=self.stream.handle)

        # Store original image and padding info for postprocessing later
        self.results_queue.append({
            "original_img": original_img,
            "padding_info": padding_info,
            "request_id": self.request_count
        })
        self.request_count += 1

    def get_results(self):
        if not self.results_queue:
            return []

        # Wait for the stream to complete (this is where async happens)
        self.stream.synchronize()

        # Copy results from device to host
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output["host"], output["device"], self.stream)

        # Process the first completed request
        request_data = self.results_queue.pop(0)
        
        # Copy output data to numpy arrays
        output_data = [out["host"] for out in self.outputs]
        # Reshape output data to match engine's output shape
        # This reshaping needs to be dynamic based on the actual output shape
        # For YOLOv8, it's typically [batch_size, num_detections, 5 + num_classes]
        # We assume batch_size=1 here.
        num_boxes = self.engine.get_binding_shape(self.outputs[0]["name"])[1]
        num_classes = self.engine.get_binding_shape(self.outputs[0]["name"])[2] - 5
        reshaped_output = output_data[0].reshape(1, num_boxes, 5 + num_classes)

        detections = postprocess_detections(
            [reshaped_output], # Pass as a list to match expected format
            self.input_shape,
            (request_data["original_img"].shape[1], request_data["original_img"].shape[0]),
            request_data["padding_info"]
        )
        return request_data["original_img"], detections

# --- Main Execution ---
if __name__ == "__main__":
    # IMPORTANT: You need to have a yolov8.engine file generated beforehand.
    # This typically involves exporting a YOLOv8 PyTorch model to ONNX and then
    # using TensorRT's `trtexec` tool or Python API to build the engine.
    # Example command for trtexec (requires ONNX model):
    # trtexec --onnx=yolov8.onnx --saveEngine=yolov8.engine --input-shape=1x3x640x640 --fp16

    try:
        inferencer = AsyncInferencer(TRT_ENGINE_PATH)
        print(f"TensorRT engine loaded from {TRT_ENGINE_PATH}")

        # Example usage with a dummy image path. Replace with a real image.
        # Create a dummy image if it doesn't exist for testing purposes.
        dummy_image_path = "test_image.jpg"
        try:
            with open(dummy_image_path, 'rb') as f:
                pass
        except FileNotFoundError:
            print(f"Creating a dummy image: {dummy_image_path}")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_img, "Dummy Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(dummy_image_path, dummy_img)

        print(f"Enqueuing inference for {dummy_image_path}...")
        inferencer.enqueue_inference(dummy_image_path)
        print("Inference enqueued. Waiting for results...")

        # Simulate multiple requests being enqueued before processing
        # In a real app, you'd enqueue frames from a video stream or camera
        # For this example, we'll just process the single enqueued request

        start_time = time.time()
        original_img, detections = inferencer.get_results()
        end_time = time.time()

        print(f"Received {len(detections)} detections in {end_time - start_time:.4f} seconds.")

        # Draw bounding boxes on the original image
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            class_name = det["class_name"]
            label = f"{class_name}: {score:.2f}"

            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display or save the result
        output_image_path = "output_image.jpg"
        cv2.imwrite(output_image_path, original_img)
        print(f"Result saved to {output_image_path}")

        # Optional: Display the image if you have a GUI environment
        # cv2.imshow("YOLOv8 TensorRT Detections", original_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the TensorRT engine file ('yolov8.engine') exists.")
        print("You can generate it from an ONNX model using TensorRT's trtexec tool.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
