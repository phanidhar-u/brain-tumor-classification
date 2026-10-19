import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 20px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        border-radius: 5px;
        margin: 5px 0;
    }
    .metric-card {
        background-color: white;
        padding: 0.8rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        color: #333333;
        border: 1px solid #e0e0e0;
    }
    .metric-card h3 {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        color: #666666;
    }
    .metric-card h2 {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: #1f77b4;
    }
    .metric-card p {
        font-size: 0.8rem;
        color: #888888;
        margin: 0;
    }
    .sidebar-content {
        font-size: 0.9rem;
    }
    .image-container {
        max-width: 100%;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BrainTumorCNN(num_classes=4)
    model_metadata = {
        'val_acc': 97.10,
        'train_acc': 100.0,
        'f1_score': 96.88,
        'image_size': 224
    }

    try:
        # Load the model weights
        checkpoint = torch.load('best_brain_tumor_model.pth', weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'pituitary', 'notumor'])
        model_metadata['val_acc'] = checkpoint.get('val_acc', 97.10)
        model_metadata['train_acc'] = checkpoint.get('train_acc', 100.0)
        model_metadata['f1_score'] = checkpoint.get('f1_score', 96.88)
        model_metadata['image_size'] = checkpoint.get('image_size', 224)
    except Exception:
        try:
            checkpoint = torch.load('best_brain_tumor_model.pth', weights_only=False, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'pituitary', 'notumor'])
            model_metadata['val_acc'] = checkpoint.get('val_acc', 97.10)
            model_metadata['train_acc'] = checkpoint.get('train_acc', 100.0)
            model_metadata['f1_score'] = checkpoint.get('f1_score', 96.88)
            model_metadata['image_size'] = checkpoint.get('image_size', 224)
        except Exception as e:
            st.warning(f"Model file error ({e}). Using initialized architecture.")
            class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

    model.to(device)
    model.eval()
    return model, class_names, device, model_metadata

def preprocess_image(image, image_size=224):
    """Preprocess the uploaded image"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_image(model, image_tensor, device, class_names):
    """Make prediction on the image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return {
        'predicted_class': class_names[predicted_idx.item()],
        'confidence': confidence.item(),
        'all_probabilities': probabilities[0].cpu().numpy(),
        'class_names': class_names
    }

def create_confidence_chart(prediction_result):
    """Create an interactive confidence chart"""
    df = pd.DataFrame({
        'Class': prediction_result['class_names'],
        'Confidence': prediction_result['all_probabilities']
    })

    fig = px.bar(df, x='Confidence', y='Class', orientation='h',
                 color='Confidence', color_continuous_scale='Blues',
                 title='Prediction Confidence Scores',
                 labels={'Confidence': 'Confidence Score', 'Class': 'Tumor Type'})

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=300  # Smaller chart
    )
    fig.update_traces(texttemplate='%{x:.3f}', textposition='outside')

    return fig

def create_comparison_radar(prediction_result):
    """Create a radar chart comparing all classes"""
    categories = prediction_result['class_names']
    values = prediction_result['all_probabilities'] * 100

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Confidence',
        line=dict(color='#1f77b4')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Confidence Distribution Across All Classes",
        height=300  # Smaller chart
    )

    return fig

def get_tumor_info(tumor_type):
    """Get information about each tumor type"""
    info = {
        'glioma': {
            'description': 'Gliomas are tumors that develop from glial cells in the brain and spinal cord.',
            'prevalence': 'Most common type of primary brain tumor (about 33%)',
            'treatment': 'Surgery, radiation therapy, chemotherapy',
            'survival': 'Varies by grade and type'
        },
        'meningioma': {
            'description': 'Tumors that develop from the meninges, the membranes surrounding the brain and spinal cord.',
            'prevalence': 'Most common primary brain tumor (about 36%)',
            'treatment': 'Surgery, radiation therapy',
            'survival': 'Generally good as most are benign'
        },
        'pituitary': {
            'description': 'Tumors that develop in the pituitary gland at the base of the brain.',
            'prevalence': 'About 10-15% of primary brain tumors',
            'treatment': 'Surgery, medication, radiation therapy',
            'survival': 'Generally excellent'
        },
        'notumor': {
            'description': 'No tumor detected in the MRI scan.',
            'prevalence': 'Healthy brain scan',
            'treatment': 'No treatment required',
            'survival': 'Normal life expectancy'
        }
    }
    return info.get(tumor_type, {})

def main():
    # Load model
    model, class_names, device, model_metadata = load_model()

    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Classification Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar - Moved upload section to top
    with st.sidebar:
        st.header("📤 Upload MRI Scan")
        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a brain MRI scan for classification"
        )

        # Sample images for demo
        st.header("🎯 Demo Images")
        demo_option = st.selectbox(
            "Try with sample images:",
            ["None", "Glioma Example", "Meningioma Example", "Pituitary Example", "No Tumor Example"]
        )

        # About section in expander to save space
        with st.expander("ℹ️ About This Model"):
            val_acc_val = model_metadata.get('val_acc', 97.10)
            f1_score_val = model_metadata.get('f1_score', 96.88)
            st.markdown(f"""
            <div class="sidebar-content">
            This AI model classifies brain tumors from MRI scans into 4 categories:

            - **Glioma** - Most common primary brain tumor
            - **Meningioma** - Typically benign tumors
            - **Pituitary** - Pituitary gland tumors
            - **No Tumor** - Healthy brain scans

            **Model Performance:**
            - Test Accuracy: {val_acc_val:.1f}%
            - Macro F1-Score: {f1_score_val:.1f}%
            - Trained on 5,712 full MRI scans
            - Evaluated on 1,311 test scans
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.info("⚠️ **Disclaimer**: This tool is for demonstration purposes only. Always consult healthcare professionals for medical diagnosis.")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🖼️ MRI Image Analysis")

        # Handle file upload or demo selection
        image = None
        sample_paths = {
            "Glioma Example": os.path.join("Testing", "glioma", "Te-glTr_0000.jpg"),
            "Meningioma Example": os.path.join("Testing", "meningioma", "Te-meTr_0000.jpg"),
            "Pituitary Example": os.path.join("Testing", "pituitary", "Te-piTr_0000.jpg"),
            "No Tumor Example": os.path.join("Testing", "notumor", "Te-noTr_0000.jpg")
        }

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        elif demo_option != "None":
            sample_file = sample_paths.get(demo_option)
            if sample_file and os.path.exists(sample_file):
                image = Image.open(sample_file).convert('RGB')
                st.image(image, caption=f"Demo Sample: {demo_option}", use_container_width=True)
            else:
                st.warning(f"Demo file {sample_file} not found. Please upload an image.")
                image = None
        else:
            st.info("👆 Please upload an MRI image or select a demo example to get started.")

    with col2:
        prediction_result = None
        if image is not None:
            st.header("🤖 AI Analysis Results")

            # Show prediction in progress
            with st.spinner('🔄 Analyzing MRI scan with Deep Neural Network...'):
                img_size = model_metadata.get('image_size', 224)
                image_tensor = preprocess_image(image, image_size=img_size)
                prediction_result = predict_image(model, image_tensor, device, class_names)

            # Display results
            if prediction_result:
                pred_class = prediction_result['predicted_class']
                confidence = prediction_result['confidence']

                # Color code based on prediction
                color_map = {
                    'glioma': '#FF6B6B',
                    'meningioma': '#4ECDC4',
                    'pituitary': '#45B7D1',
                    'notumor': '#96CEB4'
                }

                st.markdown(f"""
                <div class="prediction-card" style="border-left: 5px solid {color_map.get(pred_class, '#000000')}">
                    <h2 style="color: {color_map.get(pred_class, '#000000')}; margin-bottom: 1rem;">
                        🎯 Prediction: {pred_class.replace('_', ' ').title()}
                    </h2>
                    <h3 style="color: #2E86AB;">
                        Confidence: {confidence:.1%}
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                # Confidence chart
                st.plotly_chart(create_confidence_chart(prediction_result), use_container_width=True)

    # Additional information section
    if prediction_result:
        st.markdown("---")
        st.header("📊 Detailed Analysis")

        col3, col4 = st.columns([1, 1])

        with col3:
            # Radar chart
            st.plotly_chart(create_comparison_radar(prediction_result), use_container_width=True)

        with col4:
            # Tumor information
            tumor_info = get_tumor_info(prediction_result['predicted_class'])
            if tumor_info:
                st.subheader(f"About {prediction_result['predicted_class'].title()}")
                st.write(f"**Description**: {tumor_info['description']}")
                st.write(f"**Prevalence**: {tumor_info['prevalence']}")
                st.write(f"**Common Treatments**: {tumor_info['treatment']}")
                st.write(f"**Prognosis**: {tumor_info['survival']}")

        # Model performance metrics
        st.markdown("---")
        st.header("📈 Model Performance Overview")

        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

        acc_display = model_metadata.get('val_acc', 97.10)
        f1_display = model_metadata.get('f1_score', 96.88)

        with metrics_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Test Accuracy</h3>
                <h2>{acc_display:.1f}%</h2>
                <p>Unseen Test Set</p>
            </div>
            """, unsafe_allow_html=True)

        with metrics_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚡ F1-Score</h3>
                <h2>{f1_display:.1f}%</h2>
                <p>Macro Average</p>
            </div>
            """, unsafe_allow_html=True)

        with metrics_col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🧠 Classes</h3>
                <h2>4</h2>
                <p>Tumor Types</p>
            </div>
            """, unsafe_allow_html=True)

        with metrics_col4:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Training Data</h3>
                <h2>5,712</h2>
                <p>Full MRI Scans</p>
            </div>
            """, unsafe_allow_html=True)

        # Detailed performance table
        st.subheader("Per-Class Performance Metrics (Evaluated on Unseen Test Set)")
        performance_data = {
            'Class': ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'],
            'Precision': [0.9684, 0.9243, 0.9901, 0.9951],
            'Recall': [0.9200, 0.9575, 0.9967, 1.0000],
            'F1-Score': [0.9436, 0.9406, 0.9934, 0.9975]
        }
        st.dataframe(pd.DataFrame(performance_data), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Built with PyTorch & Streamlit • For demonstration purposes only</p>
        <p>Always consult medical professionals for accurate diagnosis and treatment</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()