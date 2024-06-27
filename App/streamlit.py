import streamlit as st
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import pandas as pd
import torch
import os
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models

# The energy threshold calculated on all images from the training dataset
ENERGY_THRESHOLD = -2.038491129875183

# Mapping of prediction classes to their names
class_idx_to_name_dict = {
    0: 'Black spot',
    1: 'Canker',
    2: 'Greening',
    3: 'Scab',
    4: 'Bacteria Citrus',
    5: 'Fungus Penicillium',
    6: 'Healthy',
    7: 'Pest Psyllid'
}


# Stores general recommendations in case the model decides an image is out-of-distribution
general_recommendations = """
#### **Soil Preparation**
- **Soil Type**: Choose well-draining, loamy soil. Amend clay soils with organic matter like compost or peat moss to improve drainage.
- **pH Level**: Maintain soil pH between 6.0 and 7.5. Adjust pH using lime (to increase) or sulfur (to decrease) as necessary.

#### **Watering**
- **Young Trees**: Water once or twice a week to help establish roots, ensuring the soil is moist but not waterlogged.
- **Mature Trees**: Reduce watering frequency but ensure trees are well-watered during dry periods. Opt for deep, infrequent watering sessions to foster deep root development.

#### **Fertilization**
- **Fertilizer Type**: Use nitrogen-rich or balanced N-P-K fertilizers specifically formulated for citrus or fruit trees.
- **Application Schedule**: Fertilize young trees several times a year and mature trees in early spring and again in mid-summer to support fruiting.

#### **Mulching**
- **Application**: Apply organic mulch such as wood chips or bark around the base of the tree to retain moisture, suppress weeds, and add nutrients.
- **Precautions**: Keep mulch a few inches away from the trunk to prevent rot and deter pests.

#### **Pest and Disease Monitoring**
- **Regular Checks**: Inspect trees regularly for signs of pests or diseases, noting any discoloration or abnormal growth.
- **Management Strategies**: Employ horticultural oils or insecticidal soaps to manage pests. Promptly remove and destroy any diseased plant parts.

#### **Support and Spacing**
- **Tree Support**: Stake young trees if necessary to ensure stability against winds.
- **Proper Spacing**: Space orange trees 12 to 25 feet apart, depending on variety, to ensure sufficient light and air circulation.

#### **Harvesting**
- **Optimal Timing**: Harvest oranges when they are fully mature, typically in winter or early spring. Note that oranges do not continue to ripen after being picked.
- **Harvesting Technique**: Use garden shears or a sharp knife to cut the fruit from the tree, leaving a bit of stem to extend shelf life.

#### **Additional Tips**
- Monitor soil moisture and drainage, especially following heavy rainfall.
- Consult with local agricultural extensions or professional horticulturists for advice specific to your region's climate and soil conditions.
"""

# Random orange facts
orange_facts = [
    "Oranges are the most cultivated fruit in the world, primarily grown for juice production.",
    "The orange is a hybrid between a pomelo and a mandarin, and it was first cultivated in ancient China.",
    "Oranges are rich in vitamin C, which is vital for repairing body tissues and maintaining a healthy immune system.",
    "Brazil is the largest producer of oranges in the world, followed by the United States, primarily in Florida and California.",
    "There are over 600 varieties of oranges worldwide, including navel oranges, Valencia oranges, and blood oranges.",
    "Oranges were once a rare and expensive commodity, considered a luxury item only available to the wealthy.",
    "The orange tree is an evergreen, flowering tree, with an average height of about 9 to 10 meters.",
    "Oranges are not only consumed as fresh fruit but are widely used in cooking and baking, and their zest (the outer skin) adds flavor to many recipes.",
    "The name 'orange' does not refer to the color but originally to the fragrance of the fruit (from the Sanskrit 'naranga').",
    "Oranges can be stored at room temperature or in the refrigerator and generally have a shelf life of up to two weeks."
]


def get_model(num_classes):
    # Load the pre-trained EfficientNet B7 without the classifier
    model = models.efficientnet_b7(pretrained=False)
    
    # Modify the classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, num_classes)
    )
    
    return model


# Function to load the model
def load_model(MODEL_PATH, NUM_CLASSES=8):
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_recommendations(path: str) -> pd.DataFrame:
    """
    Load recommendations from a CSV file.

    Args:
        path (str): The file path to the CSV containing recommendations.

    Returns:
        pd.DataFrame: DataFrame containing the recommendations.
    """
    recommendations = pd.read_csv(path)
    return recommendations

def transform_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Transform an image for model inference.

    Args:
        image (Image.Image): The input image to be transformed.
        device (torch.device): The device to which the image tensor will be moved.

    Returns:
        torch.Tensor: The transformed image tensor.
    """
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image = transform(image=np.array(image))['image']
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)
    return image_tensor

def calculate_probability_and_predicted_class(output: torch.Tensor) -> tuple:
    """
    Calculate the probability and predicted class from model output.

    Args:
        output (torch.Tensor): The output tensor from the model.

    Returns:
        tuple: A tuple containing the probability and predicted class.
    """
    p = torch.softmax(output, dim=1)
    probability, predicted_class = torch.max(p, 1)
    return probability, predicted_class

def calculate_energy(output: torch.Tensor) -> torch.Tensor:
    """
    Calculate the energy of the model output.

    Args:
        output (torch.Tensor): The output tensor from the model.

    Returns:
        torch.Tensor: The calculated energy.
    """
    return -torch.logsumexp(output, dim=1)

def inference(model: torch.nn.Module, image: Image.Image, device: torch.device) -> dict:
    """
    Perform inference on an input image using the specified model.

    Args:
        model (torch.nn.Module): The loaded PyTorch model.
        image (Image.Image): The input image for inference.
        device (torch.device): The device for computation.

    Returns:
        dict: A dictionary containing the inference result.
    """
    image_tensor = transform_image(image, device)

    with torch.inference_mode():
        output = model(image_tensor)
        energy = calculate_energy(output)
        energy_value = energy.item()
        probability, predicted_class = calculate_probability_and_predicted_class(output)
        probability_value = probability.item()
        predicted_class_value = predicted_class.item()

        if energy_value > ENERGY_THRESHOLD:
            return {
                'result': 'unknown',
                'energy': energy_value,
                'probability': probability_value,
                'predicted_class': predicted_class_value
            }

        return {
            'result': 'known',
            'probability': probability_value,
            'predicted_class': predicted_class_value
        }

def classify_image(image: Image.Image):
    """
    Classify an input image and display the results using Streamlit.

    Args:
        image (Image.Image): The input image for classification.
    """
    model_path = os.path.join(base_path, 'weights_efB7_aug_2.pth')
    model = load_model(MODEL_PATH=model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    result = inference(model, image, device)
    rec_path = "recommendations.csv"
    recommendations = load_recommendations(os.path.join(base_path, rec_path))

    if result['result'] == 'unknown':
        st.write("### The class is of unknown origin")
        st.write(f"**Energy**: {result['energy']:.4f}, but the energy threshold is {ENERGY_THRESHOLD:.4f}")
        st.write(f"**Potential class**: {class_idx_to_name_dict[result['predicted_class']]} with the probability {result['probability']:.4f}")
        st.write("### General Recommendations for Orange:")
        st.write(general_recommendations)
    else:
        st.write(f"### Probability: {result['probability']:.4f}")
        st.write(f"### Predicted class: {class_idx_to_name_dict[result['predicted_class']]}")
        st.write("### Recommendations")
        st.write("**Pesticide Methods:**")
        st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['pesticide'].values[0])
        st.write("**Non-pesticide Methods:**")
        st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['non-pesticide'].values[0])
        st.write("**Maintenance:**")
        st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['maintenance'].values[0])

def display_random_fact():
    """
    Display a random orange fact.
    """
    fact = random.choice(orange_facts)
    st.info(f"**Did you know?** {fact}")

def display_disease_frequency_chart(disease_counts: dict):
    """
    Display a bar chart of disease frequencies.

    Args:
        disease_counts (dict): A dictionary with disease names as keys and their counts as values.
    """
    diseases = list(disease_counts.keys())
    counts = list(disease_counts.values())

    plt.figure(figsize=(10, 6))
    plt.barh(diseases, counts, color='skyblue')
    plt.xlabel('Count')
    plt.title('Disease Frequency Detected by the Model')
    st.pyplot(plt)

# Define the base path relative to the script file location
base_path = os.path.dirname(__file__)
# Define the path to the 'examples' directory
examples_path = os.path.join(base_path, 'examples')
# Define the path to the GIF
gif_path = "https://github.com/minhnhat2001vt/Orange-Diseases-Prediction/blob/main/App/assets/orange_disease.gif?raw=true"

def main():
    """
    Main function to run the Streamlit app for orange disease detection.
    """

    st.title("Disease Detection for Orange")
    
    # Display the GIF
    st.markdown(
        f'<img src="{gif_path}" width="600" alt="Orange Disease Detection"/>',
        unsafe_allow_html=True
    )
    
    st.write("Upload your image of orange or choose one of the example images below:")
    
    # List images in the 'examples' directory
    try:
        example_images = os.listdir(examples_path)
        example_images = [os.path.join(examples_path, img) for img in example_images]
    except FileNotFoundError:
        example_images = []
        st.error("Failed to load example images.")
    selected_example = st.selectbox("Choose an example image:", ['None'] + example_images)
    if selected_example != 'None':
        image = Image.open(selected_example)
        st.image(image, caption="Selected Example Image", use_column_width=True)
        st.write("### Detecting...")
        classify_image(image)

    uploaded_file = st.file_uploader("Choose an image of orange...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("### Detecting...")
        classify_image(image)  

    # Display random cucumber fact
    display_random_fact()

    # Display disease frequency chart
    disease_counts = {
        "Black spot": 15,
        "Canker": 10,
        "Greening": 15,
        "Scab": 7,
        "bacteria _citrus": 4,
        "fungus_penicillium": 3,
        "healthy": 25,
        "pest_psyllid": 3
    }
    display_disease_frequency_chart(disease_counts)

    # Additional resources
    st.write("### Additional Resources")
    st.markdown("""
    - [Orange Growing Guide](https://www.thespruce.com/orange-tree-growing-guide-6541613)
    - [Orange Disease Management](https://www.researchgate.net/publication/354545630_A_Brief_Overview_of_Major_Citrus_Diseases_and_Pests_and_Its_Management)
    - [Orange Production](https://www.crowdfarming.com/blog/en/the-orange-journey/)
    """)

if __name__ == "__main__":
    main()
