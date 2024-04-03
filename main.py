import streamlit as st
import face_recognition
from PIL import Image
import numpy as np

# Function to detect faces in an image using face_recognition library
def detect_faces(image):
    face_locations = face_recognition.face_locations(image)
    return face_locations

# Function to crop faces from an image
def crop_faces(image, face_locations):
    cropped_faces = []
    for top, right, bottom, left in face_locations:
        cropped_faces.append(image[top:bottom, left:right])
    return cropped_faces

# Function to calculate and display image similarity
def calculate_similarity(image1, image2):
    encoding1 = face_recognition.face_encodings(image1)[0]
    encoding2 = face_recognition.face_encodings(image2)[0]

    similarity = face_recognition.compare_faces([encoding1], encoding2)[0]
    return similarity

# Streamlit app
def main():
    st.title("Trust and Safety: Face Similarity Check")

    with st.sidebar:
        st.header("Upload Images")
        uploaded_file1 = st.file_uploader("Upload the first image")
        uploaded_file2 = st.file_uploader("Upload the second image")

    if uploaded_file1 is not None and uploaded_file2 is not None:
        # Read the images
        image1 = np.array(Image.open(uploaded_file1))
        image2 = np.array(Image.open(uploaded_file2))

        # Detect faces in both images
        face_locations1 = detect_faces(image1)
        face_locations2 = detect_faces(image2)

        # Crop faces
        cropped_faces1 = crop_faces(image1, face_locations1)
        cropped_faces2 = crop_faces(image2, face_locations2)

        # Display cropped faces
        st.subheader("Cropped Faces from Image 1")
        for i, face in enumerate(cropped_faces1):
            st.image(face, caption=f"Face {i+1}", use_column_width=True)

        st.subheader("Cropped Faces from Image 2")
        for i, face in enumerate(cropped_faces2):
            st.image(face, caption=f"Face {i+1}", use_column_width=True)

        # Calculate and display similarity
        similarity = calculate_similarity(image1, image2)
        st.subheader("Image Similarity")
        if similarity:
            st.write("# The faces are SIMILAR")
        else:
            st.write("# The faces are NOT SIMILAR")

if __name__ == "__main__":
    main()
