import cv2
from ultralytics import YOLO
import numpy as np
import random
import time
import threading

# write function for predicting 
def predict(chosen_model, img, classes=[], conf=0.8):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.8):
    results = predict(chosen_model, img, classes, conf=conf)

    class_counts = {}   # Dictionary to store counts for each class
    class_prices = {}   # Dictionary to store prices for each class
    total_bill = 0      # Variable to store the total bill

    boxes = results[0].boxes
    res_plotted = results[0].plot(labels=False, line_width=1)[:, :, ::-1]

    img = cv2.resize(img, (1200, 1600))
    res_plotted = cv2.resize(res_plotted, (1200, 1600))

    # Get image width
    img_height, img_width, _ = img.shape

    for result in results:
        for box in result.boxes:
            # Scale bounding box coordinates based on the image dimensions
            x1, y1, x2, y2 = [int(coord * img_width if idx % 2 == 0 else coord * img_height) for idx, coord in enumerate(box.xyxy[0])]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Get class name
            class_name = result.names[int(box.cls[0])]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Generate random price for the class if not already assigned
            if class_name not in class_prices:
                random.seed(class_name)
                class_prices[class_name] = random.randint(100, 5000)  # Random price between 100 and 5000
            
            # Display class name, count, and price
            text = f"{class_name}: {class_counts[class_name]} x Rs. {class_prices[class_name]}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
            
            # Ensure text fits within image boundaries
            if y1 - text_height - 10 > 0:
                # Draw background rectangle
                img = cv2.rectangle(img, (x1, y1 - text_height - 10),
                              (x1 + text_width + 10, y1 - 4),
                              (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(img, text,
                            (x1 + 5, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    
    # Generate bill for the detected classes
    bill = "----------------------------------------\n"
    for name, count in class_counts.items():
        # Calculate total price for each class
        price = class_prices[name] * count
        total_bill += price
        bill += f"{name:<10} {count:>2} x Rs. {class_prices[name]:<2} = Rs. {price:>1}\n"
    bill += "----------------------------------------\n"
    bill += f"Total bill: Rs. {total_bill:>5}\n"
    bill += "----------------------------------------\n"
    
    # Print bill on top of the image with white background
    bill_lines = bill.split('\n')
    bill_height = len(bill_lines) * 40
    cv2.rectangle(img, (0, 0), (img_width, min(bill_height, img_height)), (255, 255, 255), -1)
    for i, line in enumerate(bill_lines):
        cv2.putText(img, line, (30, min(40 + i * 40, img_height + 80)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

    return img, results, total_bill, class_counts, class_prices, res_plotted

# for image
def detect_for_image(image_path, model_path):
    # choose model
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    result_img, results = predict_and_detect(model, image, classes=[], conf=0.8)


# for video
def detect_for_video(video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Define a thread for processing frames asynchronously
    def process_frame_async(frame):
        result_img, _ = predict_and_detect(model, frame, classes=[], conf=0.8)
        cv2.imshow('Result', result_img)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Start a new thread for processing the frame asynchronously
        threading.Thread(target=process_frame_async, args=(frame,)).start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# get the names of classes in model 
def get_classes(model_path):
    model = YOLO(model_path)
    return model.names