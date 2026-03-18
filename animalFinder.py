from ultralytics import YOLO
from IPython.display import display, Math, Latex
import cv2
import os


model = YOLO('yolov8n.pt')


image_path = 'dogs/img_1.png'


if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    print("Please place an image named 'dog_image.jpg' in your project directory or update the 'image_path' variable.")
else:
    # Run inference on the image
    # The 'results' object contains information about detected objects
    results = model(image_path)

    dogCount = 0  #
    dogLowConf = 0 # dog detections with low confidence
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class ID, confidence, and coordinates
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            # The COCO dataset class names can be accessed via the model's metadata
            class_name = model.names[class_id]

            # Filter for dogs with a certain confidence threshold
            # Confidence threshold is 50% in this scenario
            if class_name == 'dog' and confidence > 0.5:

                # IF
                dogCount = dogCount + 1

                # Draw bounding boxes and labels on the image (optional, for visualization)
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates in (x1, y1, x2, y2) format
                print(f"Detected a {class_name} with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")

                img = cv2.imread(image_path)
                color = (0, 255, 0)  # Green color for box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                            2)
                print(f"Detected a dog with high confidence confidence")

                # Display the result

            elif class_name == 'dog':
                print(f"Detected a dog with low confidence ({confidence:.2f}), skipping.")
                dogLowConf += 1
        cv2.imshow("Dog Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #tested latex but did not work.
    print(f"Found {dogCount} dogs in {image_path}")
    print(f"Found {dogLowConf} dogs with low confidence in {image_path}")

                    # Results are also automatically saved in the 'runs/detect/predict/' folder
            #print(f"Detection results saved to runs/detect/predict/ folder.")



