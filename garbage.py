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

    # ==================================================
    # HERO SECTION
    # ==================================================

    st.title("♻️ RecycleVision AI")
    st.caption("AI-Powered Smart Waste Classification System")

    st.write("""
    Transforming waste management through Deep Learning and Computer Vision.
    Upload an image and let AI automatically identify recyclable waste categories.
    """)

    st.divider()

    # ==================================================
    # QUICK STATS
    # ==================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("♻️ Classes", "6")

    with col2:
        st.metric("🧠 Models", "5")

    with col3:
        st.metric("⚙️ Framework", "PyTorch")

    with col4:
        st.metric("📂 Dataset", "TrashNet")

    st.write("")

    # ==================================================
    # ABOUT PROJECT
    # ==================================================

    with st.container(border=True):

        st.subheader("📖 About RecycleVision AI")

        st.write("""
        RecycleVision AI is a Deep Learning-powered garbage classification
        system designed to automate waste segregation using image recognition.

        The application utilizes Transfer Learning models including
        MobileNetV2, MobileNetV3, ResNet50, EfficientNetB0, and DenseNet121
        to classify waste into six categories:

        • Cardboard
        • Glass
        • Metal
        • Paper
        • Plastic
        • Trash

        The system aims to support sustainable recycling practices
        and improve waste management efficiency through Artificial Intelligence.
        """)

    st.write("")

    # ==================================================
    # PROJECT HIGHLIGHTS
    # ==================================================

    st.subheader("🚀 Project Highlights")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success("""
        ### 🧠 AI Models

        • MobileNetV2

        • MobileNetV3

        • ResNet50

        • EfficientNetB0

        • DenseNet121
        """)

    with col2:

        st.info("""
        ### ⚙️ Technologies

        • Python

        • PyTorch

        • OpenCV

        • NumPy

        • Streamlit

        • Transfer Learning
        """)

    with col3:

        st.warning("""
        ### 🌍 Benefits

        • Smart Recycling

        • Waste Segregation

        • Faster Classification

        • Sustainable Practices

        • Environmental Awareness
        """)

    st.write("")

    # ==================================================
    # FEATURES CARDS
    # ==================================================

    st.subheader("✨ Core Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        ### 📤 Image Upload

        Upload garbage images
        directly through the
        Streamlit interface.
        """)

    with col2:
        st.info("""
        ### 🧠 AI Prediction

        Transfer Learning models
        classify waste categories
        with high accuracy.
        """)

    with col3:
        st.info("""
        ### 📊 Confidence Score

        Display prediction
        confidence for
        better decision making.
        """)

    st.write("")

    # ==================================================
    # WORKFLOW
    # ==================================================

    with st.expander("🔄 How RecycleVision Works"):

        st.markdown("""
        **Step 1:** Upload a garbage image

        ⬇️

        **Step 2:** Image preprocessing

        ⬇️

        **Step 3:** Transfer Learning model inference

        ⬇️

        **Step 4:** Waste category prediction

        ⬇️

        **Step 5:** Confidence score generation
        """)

    # ==================================================
    # DATASET INFORMATION
    # ==================================================

    with st.expander("📂 Dataset Information"):

        st.markdown("""
        **Dataset:** TrashNet Garbage Classification Dataset

        **Categories**

        - Cardboard
        - Glass
        - Metal
        - Paper
        - Plastic
        - Trash

        **Image Size:** 224 × 224

        **Training Framework:** PyTorch

        **Source:** Kaggle / TrashNet
        """)

elif page == '🗑️ Image Classification':

    st.title("🗑️ Garbage Image Classification")

    st.write("Upload a garbage image to predict the waste category.")

    # Device
    device = torch.device("cpu")

    # Load model
    try:
        model = pickle.load(open("model1_cpu.pkl", "rb"))
        model.eval()
    except Exception as e:
        st.error(f"Model Error: {e}")
        st.stop()

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
