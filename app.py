import streamlit as st
import cv2
import numpy as np
import os

# Define the input and output directories
input_dir = "input"
output_dir = "output"

# Function to process the image and annotate it
def process_image(img):
    # Convert the image to grayscale
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through all contours and calculate the required parameters
    for i in range(len(contours)):
        # Smallest circle that just encapsulates the particle
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(img, center, radius, (0, 255, 0), 2)

        # Total surface area of the particle (in pixels)
        area = cv2.contourArea(contours[i])
        # Calculate 1.9 times the area
        enlarged_area = 0.19 * area
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Annotate 1.9 times the area on the image
        img = cv2.putText(img, "Area: {:.2f}".format(enlarged_area), (int(x) - 50, int(y) + 90), font, 0.6, (0, 255, 0),2)

        # Major axis (longest axis) in the particle that lies entirely inside the particle (in pixels)
        ellipse = cv2.fitEllipse(contours[i])
        major_axis_length = max(ellipse[1])
        img = cv2.ellipse(img, ellipse, (0, 255, 0), 2)
        img = cv2.putText(img, "length:{}".format(major_axis_length), (int(x) - 50, int(y) - 70), font, 0.6,(0, 255, 0), 2)

        # Total perimeter of the particle (in pixels)
        perimeter = cv2.arcLength(contours[i], True)
        # Calculate 1.9 times the perimeter
        enlarged_perimeter = 1.9 * perimeter

        # Annotate 1.9 times the perimeter on the image
        img = cv2.putText(img, "Perimeter: {:.2f}".format(enlarged_perimeter), (int(x) - 50, int(y) + 70), font, 0.5,(0, 255, 0), 2)

        # Centroid of the particle
        M = cv2.moments(contours[i])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        img = cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
        img = cv2.putText(img, "Centroid: ({}, {})".format(cx, cy), (cx + 10, cy + 10), font,0.4,(0, 255, 0),2)

        # Draw a line along the major axis that lies entirely inside the particle
        angle = -ellipse[2]
        x1 = int(x + major_axis_length / 4 * np.cos(np.radians(angle)))
        y1 = int(y + major_axis_length / 4 * np.sin(np.radians(angle)))
        x2 = int(x - major_axis_length / 4 * np.cos(np.radians(angle)))
        y2 = int(y - major_axis_length / 4 * np.sin(np.radians(angle)))
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    return img

# Streamlit code
st.title("Image Annotator App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Original Image", use_column_width=True)

    # Process the image and get the annotated output
    annotated_image = process_image(image)

    # Display the annotated image
    st.image(annotated_image, caption="Annotated Image", use_column_width=True)

    # Save the output image in the output directory
    output_filename = os.path.join(output_dir, "annotated_image.jpg")
    cv2.imwrite(output_filename, annotated_image)

    st.markdown(f"Download the annotated image [here]({output_filename}).")
