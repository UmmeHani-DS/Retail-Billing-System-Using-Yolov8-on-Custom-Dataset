import cv2
from ultralytics import YOLO
from yolo_custom import *
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import base64
import os
import time
import streamlit_theme as stt
import torch

def detect_video(video_path, confidence, classes, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model_path).to(device)
    cap = cv2.VideoCapture(video_path)
    
    placeholder = st.empty() # Create a placeholder for displaying images
    fps_placeholder = st.empty() # Placeholder for displaying FPS

    total_time = 0
    frame_count = 0

    while cap.isOpened():
        start_time = time.time() # Record the start time
        ret, frame = cap.read()
        if not ret:
            break
        
        result_img, _, total_bill, _, _, res_plotted = predict_and_detect(model, frame, classes=[], conf=confidence)
        
        # Convert BGR image to RGB
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Display result images in columns
        with placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(result_img_rgb, use_column_width=True, caption="Bill Generated")
            with col2:
                st.image(res_plotted, use_column_width=True, caption="Detected Image")

            st.header(f'The total bill is: {total_bill}', divider='rainbow')

        end_time = time.time() # Record the end time
        frame_time = end_time - start_time # Calculate the time taken to process this frame
        total_time += frame_time
        frame_count += 1
        fps = frame_count / total_time # Calculate FPS

        # Display FPS
        fps_placeholder.text(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    stt.set_theme({'primary': '#1b3388'})
    st.title(":blue[Yolov8 on Pakistani Dresses, bottles, cosmetics, etc]")
    st.header("", divider="rainbow")
    st.sidebar.title("Choose an option")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child { width: 350px; }
        [data-testid="stSidebar"][aria-expanded="False"] > div:first-child { width: 350px; margin-left: -350px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    confidence_slider = st.sidebar.slider("     Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    st.sidebar.markdown("---")

    # Slider for selecting YOLO model version
    yolo_version = st.sidebar.select_slider(
        "    Select YOLO Model Version",
        options=["best_V1", "best_V2", "best_V3"],
        value="best_V1"
    )
    st.sidebar.markdown("---")
    
    save_image = st.sidebar.checkbox("Save Image", value=False)
    custom_classes = st.sidebar.checkbox("Custom Classes", value=False)
    assigned_class_id = []
    names = get_classes(f'best_model/{yolo_version}.pt')
    names_list = []

    for classes in names.values():
        names_list.append(classes)

    if custom_classes:
        assigned_class = st.sidebar.multiselect("Select Classes", list(names_list), default=names_list)

        for each in assigned_class:
            assigned_class_id.append(names_list.index(each))

    image_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    video_file_buffer = st.sidebar.file_uploader('Upload a video', type=['mp4'])

    demo_video = "demo.mp4"

    if image_file_buffer is not None:     
        image = np.array(Image.open(image_file_buffer))
        detected_image, total_bill, res_plotted = detect_image(image, confidence_slider, assigned_class_id, yolo_version)

        res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        # Display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(detected_image, caption="Bill Generated")
        with col2:
            st.image(res_plotted, caption="Detected Image")

        bill = "----------------------------------------\n"
        empty = ""
        for name, count in class_counts.items():
            # Calculate total price for each class
            price = class_prices[name] * count
            bill += f"{name:<10} {count:>2} x Rs. {class_prices[name]:<2} = Rs. {price:>1}\n"
            st.header(empty, divider='rainbow')
            st.write(f'{name:<10} {count:>2} x Rs. {class_prices[name]:<2} = Rs. {price:>1}', divider='rainbow')

        st.header(empty, divider='rainbow')
        # Display total bill
        st.header(f'The total bill is: {total_bill}', divider='rainbow')

        if save_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                tmpfile.write(cv2.imencode('.png', detected_image)[1])
                tmpfile.seek(0)
                b64_image = base64.b64encode(tmpfile.read()).decode("utf-8")
                href = f'<a href="data:file/png;base64,{b64_image}" download="detected_image.png">Download Detected Image</a>'
                st.markdown(href, unsafe_allow_html=True)

    if video_file_buffer is not None:
        # Save the uploaded video file locally
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(video_file_buffer.read())
            video_path = tmpfile.name

        detect_video(video_path, confidence_slider, assigned_class_id, f'best_model/{yolo_version}.pt')

        # Remove the temporary file after processing
        os.remove(video_path)

def detect_image(image, confidence, classes, model_version):
    global total_bill
    global class_counts
    global class_prices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(f'best_model/{model_version}.pt').to(device)
    result_img, results, total_bill, class_counts, class_prices, res_plotted  = predict_and_detect(model, image, classes, conf=confidence)
    print("Total Bill is: ", total_bill)
    return result_img, total_bill, res_plotted

if __name__ == '__main__':
    main()
