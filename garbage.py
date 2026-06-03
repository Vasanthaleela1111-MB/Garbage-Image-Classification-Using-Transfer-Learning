import pickle
import pandas
import numpy as np
import cv2
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

st.set_page_config(
    page_title="RecycleVision AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:

    st.title("♻️ RecycleVision")

    st.caption("AI-Powered Waste Classification")

    page = st.radio(
        "Navigation",
        [
            "📘 Project Introduction",
            "🗑️ Image Classification",
            "👩‍💻 Creator Info"
        ]
    )

if page == "📘 Project Introduction":

    st.title("♻️ RecycleVision AI")
    st.caption("AI-Powered Smart Waste Classification System")

    # Feature Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("⚡ Real-Time Classification")

    with col2:
        st.info("🧠 Transfer Learning Models")

    with col3:
        st.info("♻️ Smart Recycling Support")

    st.divider()

    # About Project Container
    with st.container(border=True):

        st.subheader("📖 About the Project")

        st.write("""
        RecycleVision AI is an intelligent waste classification system
        that automatically identifies garbage categories from uploaded images.

        The application leverages Transfer Learning models such as
        MobileNetV2, ResNet50, and EfficientNetB0 to classify waste
        into cardboard, glass, metal, paper, plastic, and trash.

        The goal is to promote sustainable recycling practices and
        automate waste segregation using Artificial Intelligence.
        """)

    st.write("")

    # Stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Waste Classes", "6")

    with col2:
        st.metric("Models Used", "5")

    with col3:
        st.metric("Framework", "PyTorch")

    with col4:
        st.metric("Dataset", "TrashNet")

    st.write("")

    # Features + Technologies
    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader("🚀 Key Features")

            st.markdown("""
            ✅ Real-time garbage classification

            ✅ Transfer Learning based prediction

            ✅ Image preprocessing

            ✅ High accuracy classification

            ✅ Interactive dashboard

            ✅ Smart recycling assistance
            """)

    with col2:

        with st.container(border=True):

            st.subheader("⚙️ Technologies")

            st.markdown("""
            • Python

            • PyTorch

            • Deep Learning (CNN)

            • Transfer Learning

            • Computer Vision

            • Streamlit

            • NumPy & OpenCV
            """)

    st.write("")

    # Benefits Section
    with st.container(border=True):

        st.subheader("🌍 Environmental Benefits")

        st.success("""
        ✔ Improves waste segregation accuracy

        ✔ Encourages recycling awareness

        ✔ Reduces landfill waste

        ✔ Supports smart city initiatives

        ✔ Promotes sustainable waste management
        """)

    st.write("")

    with st.expander("📊 Dataset Information"):

        st.write("""
        **Dataset:** TrashNet Garbage Classification Dataset

        Categories:
        - Cardboard
        - Glass
        - Metal
        - Paper
        - Plastic
        - Trash

        Source: Kaggle / TrashNet
        """)

elif page == '🗑️ Image Classification':

    st.title("🗑️ Garbage Image Classification")

    st.write("Upload a garbage image to predict the waste category.")

    # Device
    device = torch.device("cpu")

    # Load model
    model = model.cpu()

    with open("model1.pkl", "wb") as f:
        pickle.dump(model, f)

    # Class names
    classes = [
        'cardboard',
        'glass',
        'metal',
        'paper',
        'plastic',
        'trash'
    ]

    # Image preprocessing
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

    # Upload image
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Open image
        image = Image.open(uploaded_file).convert("RGB")

        # img_np = np.array(image)

        # img_np = cv2.cvtColor(
        #     img_np,
        #     cv2.COLOR_RGB2BGR
        # )

        # image = Image.fromarray(img_np)

        # Transform image
        img = transform(image)

        # Add batch dimension
        img = img.unsqueeze(0).to(device)

        # Move image to device
        img = img.to(device)

        # Prediction
        with torch.no_grad():

            output = model(img)
            probabilities = F.softmax(output, dim=1)
            st.write(probabilities.cpu().numpy())
            confidence,predicted_index=torch.max(probabilities,dim=1)

        predicted_class=classes[predicted_index.item()]

        confidence_score=confidence.item()*100

        st.progress(int(confidence_score))

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:
            st.subheader("Prediction Result")

            if confidence_score < 70:
                st.warning(
                    "Low confidence prediction. Image may not belong to the training dataset."
                )
            else:
                st.success(predicted_class.upper())

            st.metric(
                "Confidence Score",
                f"{confidence_score:.2f}%"
            )

            st.progress(int(confidence_score))

        prob_df = pandas.DataFrame({
            "Category": classes,
            "Probability (%)": probabilities[0].cpu().numpy() * 100
        })

        st.subheader("Prediction Probabilities")

        st.bar_chart(
            prob_df.set_index("Category")
        )

        st.progress(int(confidence_score))
elif page == '👩‍💻 Creator Info':

    st.markdown("# 👩‍💻 Creator")

    with st.container(border=True):

        st.markdown("""
        ### Vasantha Leela M

        🎓 B.E Computer Science and Engineering  
        🏫 Karpagam Academy of Higher Education  

        ### 🚀 Skills

        - Python
        - Deep Learning
        - Computer Vision
        - Transfer Learning
        - PyTorch
        - OpenCV
        - Streamlit
        - Machine Learning
        - CNN Architectures
        - Data Preprocessing & Augmentation

        ### 💡 Project Focus

        - Garbage Image Classification
        - Smart Waste Management Systems
        - Deep Learning-based Image Recognition
        - Transfer Learning Applications
        - AI-powered Recycling Solutions
        - Computer Vision Projects

        ### 🎯 Career Objective

        Passionate about building intelligent,
        scalable, and user-friendly AI-powered
        applications using Deep Learning,
        Computer Vision, and Machine Learning
        technologies for real-world environmental
        and smart automation solutions.
        """)

    st.write("")

    st.success(
        "Focused on developing AI-powered solutions using Deep Learning, Computer Vision, and Transfer Learning technologies."
    )
