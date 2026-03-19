from IPython.display import display, Math, Latex
from ultralytics import YOLO
import cv2
import os

model = YOLO('yolov8n.pt')
image_path = 'dogs/img_7.png'



if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    print("Please place an image named 'img_4.png' in your project directory or update the 'image_path' variable.")
else:


    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image file.")
    else:
        results = model(image_path)

        dogCount = 0
        dogLowConf = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                if class_name == 'dog' and confidence > 0.30:
                    dogCount += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    print(f"Detected a {class_name} with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")

                    # Draw everything on the SAME image
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # counting dogs with low confidence:
                elif class_name == 'dog':
                    print(f"Detected a dog with low confidence ({confidence:.2f}), skipping.")
                    dogLowConf += 1

        #then showing the final picture after drawing all the boxes:
        cv2.imshow("Dog Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Found {dogCount} dogs in {image_path}")
        print(f"Found {dogLowConf} dogs with low confidence in {image_path}")