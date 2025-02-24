from ultralytics import YOLO
 
# Load a model
# model = YOLO("last.pt")  # pretrained YOLO11n model
model = YOLO("final_last.pt")  # pretrained YOLO11n model
 
# Run batched inference on a list of images
# results = model(["dataset/1.jpg", "dataset/2.jpg"])  # return a list of Results objects
results = model([r"invoicedata\YOLODataset\dataset_yolov10\test\images\6.jpg"])  # return a list of Results objects
 
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk