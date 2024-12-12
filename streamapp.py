import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import streamlit as st

# Device configuration (GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(img):
    """
    Detect faces and generate embeddings from an image.
    """
    faces = mtcnn(img)
    if faces is not None:
        embeddings = model(faces).detach().cpu().numpy()
        return faces, embeddings
    return None, None

# Load known faces from the file
def load_known_faces():
    try:
        with open("known_faces.json", "r") as f:
            known_faces = json.load(f)
        return known_faces
    except FileNotFoundError:
        return {}

known_faces = load_known_faces()

def recognize_face(detected_embedding, known_faces):
    """
    Compare the detected face embedding with known faces and return the name if matched.
    """
    detected_embedding = normalize(detected_embedding.reshape(1, -1))[0]

    for name, known_embedding in known_faces.items():
        known_embedding = np.array(known_embedding).reshape(1, -1)
        known_embedding = normalize(known_embedding)[0]
        
        similarity = cosine_similarity([detected_embedding], [known_embedding])
        normalized_similarity = similarity[0][0]
        
        if similarity[0][0] > 0.8:
            return name
    return "Unknown"

def process_image_for_recognition(image):
    """
    Detect faces, recognize them, and output similarity scores.
    """
    faces, embeddings = get_face_embedding(image)
    if faces is not None:
        recognized_names = []
        for i, face in enumerate(faces):
            detected_embedding = embeddings[i]
            recognized_name = recognize_face(detected_embedding, known_faces)
            recognized_names.append(recognized_name)
        
        return recognized_names
    return "No faces detected."

def main():
    st.title("Face Recognition App")
    st.write("Upload an image to recognize faces.")

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        recognized_names = process_image_for_recognition(img)

        if recognized_names == "No faces detected.":
            st.write("No faces detected in the image.")
        else:
            for name in recognized_names:
                st.write(f"Recognized: {name}")

if __name__ == "__main__":
    main()
