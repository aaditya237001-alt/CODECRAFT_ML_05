import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import pickle
import sys
import os

# --------- Configuration ---------
MODEL_PATH = "food101_model.pth"             # Your model file
ENCODER_PATH = "label_encoder.pkl"           # Your label encoder file

# Example calorie lookup (you can extend this)
calorie_lookup = {
    "apple_pie": 237,
    "baby_back_ribs": 320,
    "baklava": 430,
    "beef_carpaccio": 150,
    "beef_tartare": 180,
    "beet_salad": 80,
    "beignets": 350,
    "bibimbap": 200,
    "bread_pudding": 250,
    "breakfast_burrito": 290,
    "bruschetta": 120,
    "caesar_salad": 180,
    "cannoli": 330,
    "caprese_salad": 150,
    "carrot_cake": 410,
    "ceviche": 100,
    "cheesecake": 321,
    "cheese_plate": 350,
    "chicken_curry": 240,
    "chicken_quesadilla": 310,
    "chicken_wings": 290,
    "chocolate_cake": 389,
    "chocolate_mousse": 330,
    "churros": 410,
    "clam_chowder": 160,
    "club_sandwich": 300,
    "crab_cakes": 250,
    "creme_brulee": 330,
    "croque_madame": 320,
    "cup_cakes": 395,
    "deviled_eggs": 200,
    "donuts": 452,
    "dumplings": 210,
    "edamame": 120,
    "eggs_benedict": 290,
    "escargots": 180,
    "falafel": 310,
    "filet_mignon": 275,
    "fish_and_chips": 350,
    "foie_gras": 450,
    "french_fries": 312,
    "french_onion_soup": 90,
    "french_toast": 290,
    "fried_calamari": 330,
    "fried_rice": 240,
    "frozen_yogurt": 150,
    "garlic_bread": 330,
    "gnocchi": 240,
    "greek_salad": 120,
    "grilled_cheese_sandwich": 370,
    "grilled_salmon": 208,
    "guacamole": 160,
    "gyoza": 210,
    "hamburger": 295,
    "hot_and_sour_soup": 80,
    "hot_dog": 290,
    "huevos_rancheros": 230,
    "hummus": 166,
    "ice_cream": 207,
    "lasagna": 320,
    "lobster_bisque": 180,
    "lobster_roll_sandwich": 310,
    "macaroni_and_cheese": 330,
    "macarons": 430,
    "miso_soup": 45,
    "mussels": 120,
    "nachos": 400,
    "omelette": 154,
    "onion_rings": 420,
    "oysters": 105,
    "pad_thai": 290,
    "paella": 240,
    "pancakes": 227,
    "panna_cotta": 250,
    "peking_duck": 340,
    "pho": 160,
    "pizza": 266,
    "pork_chop": 250,
    "poutine": 330,
    "prime_rib": 315,
    "pulled_pork_sandwich": 330,
    "ramen": 280,
    "ravioli": 210,
    "red_velvet_cake": 390,
    "risotto": 220,
    "samosa": 260,
    "sashimi": 130,
    "scallops": 100,
    "seaweed_salad": 90,
    "shrimp_and_grits": 310,
    "spaghetti_bolognese": 250,
    "spaghetti_carbonara": 340,
    "spring_rolls": 180,
    "steak": 271,
    "strawberry_shortcake": 290,
    "sushi": 180,
    "tacos": 230,
    "takoyaki": 180,
    "tiramisu": 454,
    "tuna_tartare": 140,
    "waffles": 291
}


# --------- Load Model ---------
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 101)  # adjust if needed
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# --------- Load Label Encoder ---------
def load_label_encoder():
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

# --------- Prediction Function ---------
def predict(image_path, model, label_encoder):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        class_index = pred.item()
        class_name = label_encoder.inverse_transform([class_index])[0]
        calories = calorie_lookup.get(class_name, "Unknown")

    print(f"\n🍽️  Predicted Food: {class_name}")
    print(f"🔥  Estimated Calories: {calories} kcal (per 100g)\n")

# --------- Main ---------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(r"Usage: python food_predictor.py <C:\Users\aadit\OneDrive\Desktop\ml task 05\TIRAMISU.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]

    model = load_model()
    label_encoder = load_label_encoder()

    predict(image_path, model, label_encoder)
