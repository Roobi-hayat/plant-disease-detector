import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
import sqlite3
import streamlit as st
import tempfile
import streamlit.components.v1 as components

# Global variables
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  
MODEL_PATH = "models/plant_disease_model.h5"
PLANT_TYPES = ["Apple", "Cherry", "Corn", "Grape", 
               "Peach", "Pepper", "Potato", 
                "Strawberry", "Tomato"]

st.set_page_config(
    page_title="AI Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar styling
st.sidebar.title("🌿 Plant Care Assistant")
st.sidebar.markdown("""
Use the AI-powered detector to diagnose plant diseases from images and get care tips.

**Steps:**
1. Select the plant type
2. Upload a plant leaf image
3. Get diagnosis and treatment advice
""")

st.sidebar.markdown("---")




st.markdown("### Step 1: Capture Image from Back Camera")

# Camera capture with base64 download
components.html("""
<style>
  #capture-container {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  video {
    max-width: 100%;
    height: auto;
  }
  #capture-btn {
    margin-top: 10px;
    padding: 10px 20px;
    font-size: 16px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
  }
  #downloadLink {
    margin-top: 10px;
    display: none;
  }
</style>

<div id="capture-container">
    <video id="video" autoplay playsinline></video>
    <canvas id="canvas" style="display:none;"></canvas>
    <button id="capture-btn" onclick="capture()">📸 Capture</button>
    <a id="downloadLink">Download Image</a>
</div>

<script>
  const video = document.getElementById('video');
  navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
    .then(stream => video.srcObject = stream)
    .catch(err => console.error("Camera error:", err));

  function capture() {
    const canvas = document.getElementById('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg');

    const link = document.getElementById('downloadLink');
    link.href = dataUrl;
    link.download = 'plant.jpg';
    link.textContent = '📥 Download Captured Image';
    link.style.display = 'block';
  }
</script>
""", height=500)

st.markdown("---")
st.markdown("### Step 2: Upload Captured Image")

uploaded_file = st.file_uploader("Upload the captured image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # 🔍 AI prediction placeholder
    st.success("✅ Image uploaded! You can now run your AI prediction.")
    # result = your_model.predict(...)
    # st.write(result)
else:
    st.info("📷 Capture and upload a plant image to continue.")



# Main header
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #2E8B57;
        text-align: center;
        margin-top: -40px;
        margin-bottom: 20px;
    }
    .sub-text {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        border: None;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #3CB371;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

class PlantDiseaseDetector:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.model = None
        self.class_names = []
        self.current_plant = None
        self.care_tips_data = [
         ('Apple___Apple_scab', 
         '1. Remove and destroy infected leaves and fruit. 2. Apply copper-based or sulfur fungicides every 7-10 days. 3. Apply neem oil as an organic alternative. 4. Prune to improve air circulation within the canopy. 5. Apply potassium bicarbonate sprays for active infections.', 
         '1. Plant scab-resistant apple varieties. 2. Clean up all fallen leaves and fruit in autumn. 3. Maintain proper tree spacing for air circulation. 4. Apply preventative fungicide in early spring before bud break. 5. Avoid overhead irrigation which wets the leaves.'),
    
         ('Apple___Black_rot', 
         '1. Prune out all dead or diseased wood, cutting at least 6 inches below visible infection. 2. Remove all mummified fruits from the tree and ground. 3. Apply fungicides containing captan or myclobutanil. 4. Treat open wounds on trees with wound dressing. 5. Remove infected fruit as soon as symptoms appear.', 
         '1. Maintain tree vigor with proper fertilization and watering. 2. Prune annually to remove dead wood and improve air circulation. 3. Apply dormant spray of copper fungicide before buds break. 4. Maintain proper spacing between trees. 5. Clean and disinfect pruning tools between trees.'),
    
         ('Apple___Cedar_apple_rust', 
         '1. Apply fungicides containing myclobutanil during active infection periods. 2. Remove galls on nearby cedar/juniper trees. 3. Apply protective fungicide sprays starting at pink bud stage. 4. Prune heavily infected branches. 5. Apply horticultural oil to reduce spore germination.', 
         '1. Plant rust-resistant apple varieties. 2. Remove nearby cedar/juniper trees if possible (alternate hosts). 3. Apply protective fungicides from early spring until early summer. 4. Maintain good air circulation with proper pruning. 5. Avoid planting apples within 1/2 mile of cedar/juniper trees.'),
    
         ('Apple___healthy', 
         '1. Water consistently providing 1-2 inches of water weekly. 2. Apply balanced fertilizer in early spring. 3. Prune annually during dormant season to maintain tree shape. 4. Thin fruit to improve size and prevent branch breakage. 5. Monitor regularly for early signs of pests or disease.', 
         '1. Apply dormant oil spray in late winter before bud break. 2. Maintain good sanitation by removing fallen fruit and leaves. 3. Apply mulch around tree base (keeping away from trunk). 4. Install proper drainage to prevent root diseases. 5. Follow regional spray schedule recommendations.'),
    
         # Cherry diseases
         ('Cherry___Powdery_mildew', 
         '1. Apply sulfur-based fungicides at first sign of infection. 2. Use potassium bicarbonate sprays for active infections. 3. Apply neem oil as an organic alternative. 4. Prune affected branches and destroy infected material. 5. Increase air circulation by thinning interior branches.', 
         '1. Plant resistant cherry varieties. 2. Space trees properly for good air circulation. 3. Apply preventative fungicides before symptoms appear. 4. Avoid excessive nitrogen fertilization. 5. Water at the base of trees to keep foliage dry.'),
    
         ('Cherry___healthy', 
         '1. Water deeply once weekly during growing season. 2. Apply balanced fertilizer in early spring. 3. Prune to an open center in late winter. 4. Apply mulch around base (keeping away from trunk). 5. Thin fruit as needed to prevent branch breakage.', 
         '1. Apply dormant oil spray in late winter. 2. Remove all fallen fruit and leaves. 3. Maintain proper air circulation with pruning. 4. Apply copper spray before bud break. 5. Install bird netting to prevent fruit damage.'),
    
         # Corn diseases
         ('Corn___Cercospora_leaf_spot Gray_leaf_spot', 
         '1. Apply foliar fungicides containing pyraclostrobin, azoxystrobin, or trifloxystrobin. 2. Time fungicide applications between tasseling and early silking stages. 3. Remove severely infected plants. 4. Maintain proper fertility to help plants withstand infection. 5. Ensure adequate drainage in fields.', 
         '1. Plant resistant hybrids. 2. Rotate crops (avoid corn following corn). 3. Till infected crop residue completely. 4. Plant in well-drained soil. 5. Avoid overcrowded planting.'),
    
         ('Corn___Common_rust', 
         '1. Apply fungicides containing azoxystrobin or pyraclostrobin. 2. Time fungicide applications at first sign of infection. 3. Maintain proper soil fertility, especially potassium. 4. Ensure adequate irrigation to reduce plant stress. 5. Remove severely infected plants.', 
         '1. Plant rust-resistant corn hybrids. 2. Schedule planting to avoid optimal rust conditions. 3. Maintain proper plant spacing for air circulation. 4. Apply preventative fungicide before disease onset. 5. Scout fields regularly for early detection.'),
    
         ('Corn___Northern_Leaf_Blight', 
         '1. Apply fungicides containing strobilurins, triazoles, or chlorothalonil. 2. Time applications at first sign of disease or before tasseling. 3. Maintain balanced soil fertility. 4. Remove severely infected plants. 5. Ensure adequate drainage in fields.', 
         '1. Plant resistant hybrids. 2. Practice crop rotation (3-year minimum). 3. Plow under crop residue after harvest. 4. Avoid overhead irrigation. 5. Control grassy weeds that may harbor disease.'),
    
         ('Corn___healthy', 
         '1. Apply nitrogen fertilizer when plants are knee-high. 2. Water consistently during silking and ear development. 3. Control weeds to reduce competition. 4. Side-dress with additional fertilizer if leaves yellow. 5. Ensure proper plant spacing (usually 8-12 inches apart).', 
         '1. Plant disease-resistant varieties. 2. Rotate crops annually. 3. Allow proper spacing between plants. 4. Control insects that can spread disease. 5. Ensure proper soil drainage.'),
    
         # Grape diseases
         ('Grape___Black_rot', 
         '1. Apply fungicides containing myclobutanil, azoxystrobin, or captan. 2. Remove infected berries and leaves. 3. Prune out infected canes during dormant season. 4. Increase air circulation by proper canopy management. 5. Time fungicide applications before rainfall events.', 
         '1. Plant resistant grape varieties. 2. Prune vines properly to enhance air circulation. 3. Remove all mummified fruits after harvest. 4. Apply dormant sprays in early spring before bud break. 5. Maintain weed-free area under vines.'),
    
         ('Grape___Esca_(Black_Measles)', 
         '1. No effective fungicidal treatment exists once infected. 2. Prune out affected canes. 3. Apply wound dressings after pruning. 4. In severe cases, remove and destroy entire vines. 5. Support vine health with balanced fertilizer and irrigation.', 
         '1. Use clean, certified planting material. 2. Disinfect pruning tools between vines. 3. Prune during dry weather to minimize infection. 4. Avoid pruning wounds near old cuts. 5. Apply protective paste to large pruning wounds.'),
    
         ('Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
         '1. Apply copper-based fungicides or those containing mancozeb. 2. Remove severely infected leaves. 3. Apply fungicide every 10-14 days during wet conditions. 4. Maintain proper canopy management for air circulation. 5. Water at soil level to avoid wetting foliage.', 
         '1. Plant resistant grape varieties. 2. Ensure proper spacing for air circulation. 3. Apply preventative fungicide before rainy periods. 4. Maintain proper pruning to reduce leaf wetness periods. 5. Remove fallen leaves from vineyard.'),
    
         ('Grape___healthy', 
         '1. Prune annually in late winter. 2. Apply balanced fertilizer in early spring. 3. Water deeply and infrequently to encourage deep root growth. 4. Thin fruit clusters to improve quality. 5. Apply compost or mulch to conserve moisture.', 
         '1. Apply dormant oil spray in late winter. 2. Use trellising systems to improve air circulation. 3. Remove weeds from base of vines. 4. Scout regularly for early signs of pest or disease. 5. Apply preventative fungicides according to regional schedules.'),

         # Peach diseases
         ('Peach___Bacterial_spot', 
         '1. Apply copper-based bactericides early in the growing season. 2. Prune out infected twigs during dry weather. 3. Maintain good air circulation through proper pruning. 4. Apply streptomycin during bloom period where approved. 5. Remove severely infected leaves and fruit.', 
         '1. Plant resistant peach varieties. 2. Avoid overhead irrigation. 3. Space trees properly for good air circulation. 4. Apply preventative copper sprays before bud break. 5. Maintain balanced tree nutrition, avoiding excess nitrogen.'),
    
         ('Peach___healthy', 
         '1. Prune annually in late winter to maintain open center. 2. Apply balanced fertilizer in early spring. 3. Thin fruit to hand-width spacing. 4. Water deeply weekly during fruit development. 5. Monitor regularly for pest and disease issues.', 
         '1. Apply dormant oil spray in late winter. 2. Remove fallen fruit and leaves. 3. Maintain proper air circulation with pruning. 4. Apply fungicides according to regional spray schedules. 5. Mulch around tree base (keeping away from trunk).'),
    
          # Pepper diseases
         ('Pepper__bell___Bacterial_spot', 
          '1. Apply copper-based bactericides every 7-10 days during wet periods. 2. Remove infected leaves and fruit. 3. Rotate with copper and mancozeb for resistance management. 4. Improve air circulation by proper plant spacing and staking. 5. Apply compost tea as a preventative measure.', 
          '1. Plant disease-resistant pepper varieties. 2. Use drip irrigation to avoid wetting foliage. 3. Practice 2-3 year crop rotation. 4. Use disease-free seeds and transplants. 5. Apply mulch to prevent soil splash onto leaves.'),
    
         ('Pepper__bell___healthy', 
         '1. Water consistently, keeping soil evenly moist. 2. Apply balanced fertilizer when plants begin producing fruit. 3. Support plants with stakes or cages. 4. Harvest peppers regularly to encourage production. 5. Apply foliar calcium spray if blossom end rot appears.', 
         '1. Plant in well-drained soil with full sun. 2. Avoid overhead watering. 3. Space plants properly for air circulation. 4. Apply mulch to maintain soil moisture and temperature. 5. Remove damaged or diseased fruit promptly.'),
    
         # Potato diseases
          ('Potato___Early_blight', 
         '1. Apply fungicides containing chlorothalonil, mancozeb, or copper. 2. Remove lower infected leaves. 3. Increase spacing between plants for better air circulation. 4. Apply fungicide every 7-10 days during wet weather. 5. Maintain adequate fertility, especially potassium.', 
         '1. Plant certified disease-free seed potatoes. 2. Practice crop rotation (3-4 years). 3. Hill soil around plants to protect tubers. 4. Apply mulch to prevent soil splash. 5. Avoid overhead irrigation.'),
    
          ('Potato___Late_blight', 
         '1. Apply fungicides containing chlorothalonil, mancozeb, or copper at first symptoms. 2. Destroy infected plants immediately. 3. Increase frequency of fungicide applications during wet weather. 4. Harvest tubers during dry weather if infection is present. 5. Allow tuber skins to cure before storage.', 
         '1. Plant certified disease-free seed potatoes. 2. Practice crop rotation (3-4 years). 3. Plant resistant varieties. 4. Destroy volunteer potatoes and nightshade weeds. 5. Avoid overhead irrigation and improve field drainage.'),
    
         ('Potato___healthy', 
         '1. Hill soil around plants as they grow. 2. Apply balanced fertilizer when plants are 6 inches tall. 3. Water deeply and consistently (1-2 inches per week). 4. Monitor for pests and diseases regularly. 5. Ensure good air circulation between plants.', 
         '1. Plant certified disease-free seed potatoes. 2. Practice crop rotation. 3. Allow proper spacing between plants. 4. Control weeds that may harbor disease. 5. Store harvested potatoes in cool, dark, dry conditions.'),
    
          # Strawberry diseases
         ('Strawberry___Leaf_scorch', 
         '1. Apply fungicides containing captan or myclobutanil. 2. Remove infected leaves. 3. Improve air circulation by proper spacing and removing runners. 4. Apply fungicide every 7-14 days during wet periods. 5. Maintain proper fertility but avoid excess nitrogen.', 
         '1. Plant resistant varieties. 2. Use drip irrigation to avoid wetting foliage. 3. Practice 3-year crop rotation. 4. Renew strawberry beds regularly. 5. Apply mulch to prevent soil splash.'),
    
         ('Strawberry___healthy', 
         '1. Remove runners unless propagating new plants. 2. Apply balanced fertilizer in early spring and after harvest. 3. Water deeply once weekly (1-1.5 inches). 4. Apply mulch to conserve moisture and reduce weeds. 5. Remove flower buds in the planting year.', 
         '1. Plant disease-free, certified plants. 2. Renew beds every 3-4 years. 3. Maintain proper spacing for air circulation. 4. Avoid overhead irrigation. 5. Remove old leaves and debris after harvest.'),
    
         # Tomato diseases
         ('Tomato___Bacterial_spot', 
         '1. Apply copper-based bactericides every 7-10 days. 2. Remove infected leaves and fruit. 3. Improve air circulation with proper spacing and staking. 4. Rotate copper with mancozeb to prevent resistance. 5. Apply compost tea as preventative measure.', 
         '1. Plant disease-resistant varieties. 2. Use drip irrigation to avoid wetting foliage. 3. Practice 2-3 year crop rotation. 4. Use disease-free seeds and transplants. 5. Apply mulch to prevent soil splash.'),
    
         ('Tomato___Early_blight', 
         '1. Apply fungicides containing chlorothalonil, mancozeb, or copper. 2. Remove lower infected leaves. 3. Stake plants for better air circulation. 4. Apply fungicide every 7-10 days during wet weather. 5. Maintain adequate fertility, especially potassium.', 
         '1. Plant disease-resistant varieties. 2. Practice crop rotation (3-4 years). 3. Space plants properly for air circulation. 4. Use mulch to prevent soil splash. 5. Water at soil level, avoiding leaf wetness.'),
    
         ('Tomato___Late_blight', 
         '1. Apply fungicides containing chlorothalonil, mancozeb, or copper at first symptoms. 2. Remove infected plants immediately. 3. Increase frequency of fungicide applications during wet weather. 4. Improve air circulation with proper spacing and staking. 5. Harvest remaining fruit if infection is present.', 
         '1. Plant resistant varieties. 2. Use disease-free transplants. 3. Improve field drainage. 4. Avoid overhead irrigation. 5. Destroy volunteer tomatoes and nightshade weeds.'),
    
         ('Tomato___Leaf_Mold', 
         '1. Apply fungicides containing chlorothalonil or copper. 2. Remove infected leaves. 3. Increase ventilation in greenhouses. 4. Space plants properly for air circulation. 5. Reduce humidity by watering in morning.', 
         '1. Plant resistant varieties. 2. Improve air circulation. 3. Keep relative humidity below 85%. 4. Use drip irrigation to avoid wetting foliage. 5. Remove crop debris after harvest.'),
    
         ('Tomato___Septoria_leaf_spot', 
         '1. Apply fungicides containing chlorothalonil, mancozeb, or copper. 2. Remove infected leaves immediately. 3. Apply fungicide every 7-10 days during wet weather. 4. Improve air circulation with proper spacing and staking. 5. Mulch around plants to prevent soil splash.', 
         '1. Practice crop rotation (3-4 years). 2. Use disease-free seeds and transplants. 3. Remove and destroy all plant debris after harvest. 4. Avoid overhead irrigation. 5. Space plants properly for good air circulation.'),
    
         ('Tomato___Spider_mites Two-spotted_spider_mite', 
         '1. Apply insecticidal soap or horticultural oil, covering leaf undersides thoroughly. 2. For severe infestations, apply miticides like bifenazate or abamectin. 3. Introduce predatory mites for biological control. 4. Increase humidity with regular misting of leaves. 5. Apply neem oil as an organic alternative.', 
         '1. Maintain proper plant hydration as water-stressed plants are more susceptible. 2. Use overhead irrigation occasionally to wash mites off plants. 3. Avoid excessive nitrogen fertilization. 4. Control weeds that may harbor mites. 5. Regularly inspect plants for early detection.'),
    
         ('Tomato___Target_Spot', 
         '1. Apply fungicides containing chlorothalonil, mancozeb, or azoxystrobin. 2. Remove infected leaves and fruit. 3. Improve air circulation with proper spacing and staking. 4. Apply fungicide every 10-14 days during wet weather. 5. Maintain proper fertility.', 
         '1. Practice crop rotation (2-3 years). 2. Use disease-free seeds and transplants. 3. Space plants properly for good air circulation. 4. Apply mulch to prevent soil splash. 5. Avoid overhead irrigation.'),
      
         ('Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
         '1. No cure exists; remove and destroy infected plants immediately. 2. Control whitefly vectors with appropriate insecticides. 3. Use reflective mulches to repel whiteflies. 4. Apply insecticidal soap or neem oil for organic control. 5. Use yellow sticky traps to monitor and reduce whitefly populations.', 
         '1. Plant resistant varieties. 2. Use virus-free transplants. 3. Control whiteflies throughout the growing season. 4. Use fine mesh row covers until flowering. 5. Maintain weed-free fields to reduce whitefly habitat.'),
    
         ('Tomato___Tomato_mosaic_virus', 
          '1. No cure exists; remove and destroy infected plants. 2. Disinfect garden tools regularly. 3. Control aphids that may spread the virus. 4. Wash hands after handling infected plants. 5. Remove weeds that may harbor the virus.', 
         '1. Plant resistant varieties with "TMV" designation. 2. Use virus-free seeds and transplants. 3. Practice crop rotation. 4. Avoid using tobacco products around tomato plants. 5. Wash hands before handling plants if you use tobacco products.'),
    
         ('Tomato___healthy', 
         '1. Stake or cage plants for support. 2. Water deeply at soil level 1-2 times weekly. 3. Apply balanced fertilizer when first fruits appear. 4. Prune suckers for indeterminate varieties. 5. Harvest regularly to encourage production.', 
         '1. Plant in well-draining soil with full sun. 2. Rotate planting locations yearly. 3. Space plants properly for air circulation. 4. Apply mulch to maintain soil moisture and temperature. 5. Remove lower leaves that touch the soil.')
        ]
        
        self._init_db()  

    def _init_db(self):
        """Initialize SQLite database for care tips."""
        conn = sqlite3.connect('plant_care.db')
        cursor = conn.cursor()
        
        # Create care_tips table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS care_tips (
                disease TEXT PRIMARY KEY,
                treatment TEXT,
                prevention TEXT
            )
        ''')
        
        # Insert sample care tips (customize these!)
        cursor.executemany('''
            INSERT OR IGNORE INTO care_tips VALUES (?, ?, ?)
        ''', self.care_tips_data)
        
        conn.commit()
        conn.close()

    def _get_care_tips(self, disease):
        """Fetch care tips from database."""
        try:
            with sqlite3.connect('plant_care.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT treatment, prevention FROM care_tips WHERE disease=?', (disease,))
                result = cursor.fetchone()
            
            if result:
                return {'treatment': result[0], 'prevention': result[1]}
            else:
                return {
                    'treatment': 'General care: Ensure proper sunlight, water, and nutrients.',
                    'prevention': 'Regularly inspect plants for early signs of disease.'
                }
        except Exception as e:
            print(f"Database error: {str(e)}")
            return {
                'treatment': 'Error retrieving treatment information.',
                'prevention': 'Error retrieving prevention information.'
            }

    def set_plant_type(self, plant_type):
        """Set the current plant type for focused prediction."""
        if plant_type in PLANT_TYPES:
            self.current_plant = plant_type
            print(f"Plant type set to: {plant_type}")
            return True
        else:
            print(f"Invalid plant type: {plant_type}")
            print(f"Available plants: {', '.join(PLANT_TYPES)}")
            return False

    def get_plant_classes(self, plant_type=None):
        """Get class names for a specific plant type or all if None."""
        if not self.class_names:
            self.load_class_names()
            
        if not plant_type:
            return self.class_names
            
        plant_classes = [cls for cls in self.class_names if cls.lower().startswith(plant_type.lower())]
        return plant_classes

    def load_dataset(self):
        """Load and prepare the dataset for training or validation."""
        print("Loading dataset...")
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        valid_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = valid_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        return train_generator, validation_generator

    def build_model(self, num_classes):
        """Use transfer learning with a pre-trained model."""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self):
        """Train the model on the dataset."""
        if not self.dataset_path:
            print("Error: Dataset path not specified")
            return
        
        train_generator, validation_generator = self.load_dataset()
        self.model = self.build_model(len(self.class_names))
        print(self.model.summary())
        
        checkpoint = ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        print(f"\nTraining model...")
        history = self.model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[checkpoint, early_stop, reduce_lr]
        )
        
        self.save_class_names()
        self.plot_training_history(history)
        self.evaluate(validation_generator)
        
        return history

    def save_class_names(self):
        """Save class names to a file."""
        os.makedirs('models', exist_ok=True)
        with open('models/class_names.txt', 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        print(f"Class names saved to models/class_names.txt")

    def load_class_names(self):
        """Load class names from file."""
        try:
            with open("class_names.txt", "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.class_names)} classes")
        except FileNotFoundError:
            print("Class names file not found")

    def load_trained_model(self):
        """Load a pre-trained model."""
        try:
            self.model = load_model("plant_disease_model.h5")
            self.load_class_names()
            print(f"Model loaded from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict(self, image_path):
        """Predict disease and return diagnosis + care tips."""
        if not self.model:
           print("Attempting to load model...")
           if not self.load_trained_model():
               print("Failed to load model")
               return None, None, None
           else:
               print("Model loaded successfully")
        
        try:
            print(f"Processing image: {image_path}")
            img = Image.open(image_path)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            print("Making prediction...")
            predictions = self.model.predict(img_array)
            print(f"Raw predictions: {predictions}")
            
            
            predictions = self.model.predict(img_array)
            
            if self.current_plant:
                plant_classes = self.get_plant_classes(self.current_plant)
                if plant_classes:
                    class_indices = {cls: i for i, cls in enumerate(self.class_names)}
                    filtered_predictions = np.zeros_like(predictions[0])
                    for cls in plant_classes:
                        if cls in class_indices:
                            idx = class_indices[cls]
                            filtered_predictions[idx] = predictions[0][idx]
                    
                    if np.sum(filtered_predictions) > 0:
                        predicted_class_index = np.argmax(filtered_predictions)
                        confidence = filtered_predictions[predicted_class_index]
                    else:
                        return f"No valid prediction for {self.current_plant}", 0.0, {"treatment": "", "prevention": ""}
                else:
                    return f"No classes found for {self.current_plant}", 0.0, {"treatment": "", "prevention": ""}
            else:
                predicted_class_index = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_index]
            
            if predicted_class_index < len(self.class_names):
                predicted_class = self.class_names[predicted_class_index]
                plant_disease = self.parse_class_name(predicted_class)
                tips = self._get_care_tips(predicted_class)
                return plant_disease, confidence, tips
            else:
                return "Unknown", 0.0, {"treatment": "", "prevention": ""}
        except Exception as e:
            print(f"Error predicting image: {str(e)}")
            return None, None, None

    def parse_class_name(self, class_name):
        """Parse the class name to extract plant type and disease condition."""
        parts = class_name.split('___') if '___' in class_name else class_name.split('_')
        plant_type = parts[0].replace('_', ' ')
        if len(parts) > 1:
            condition = parts[1].replace('_', ' ')
            return f"{plant_type} - {condition}"
        else:
            if "healthy" in class_name.lower():
                return f"{plant_type} - Healthy"
            else:
                return class_name.replace('_', ' ')

def main():
    detector = PlantDiseaseDetector()
    if not detector.load_trained_model():
        st.error("No trained model found. Please train a model first.")
        return

    #st.title("Plant Disease Detector")
    st.markdown('<div class="main-title">AI Plant Disease Detector & Care Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Upload a leaf image to detect diseases and get treatment & prevention tips</div>', unsafe_allow_html=True)

    
    # Add a decorative separator
    st.markdown("<hr style='border: 1px solid #ccc;' />", unsafe_allow_html=True) 

    
    plant_type = st.selectbox("Select Plant Type", [""] + PLANT_TYPES)
    if plant_type:
        detector.set_plant_type(plant_type)

    st.write("### Take a picture or upload an image of a leaf")
    use_camera = st.checkbox("Use Camera")
    if use_camera:
        picture = st.camera_input("Take a picture")
        if picture:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(picture.getvalue())
                image_path = tmp_file.name
                predicted_class, confidence, tips = detector.predict(image_path)
                if predicted_class:
                    st.write(f"Result: {predicted_class} ({confidence*100:.2f}% confidence)")
                    st.write(f"Treatment: {tips['treatment']}")
                    st.write(f"Prevention: {tips['prevention']}")
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
         st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name
            try:
                predicted_class, confidence, tips = detector.predict(image_path)
                if predicted_class:
                    st.success(f"Result: {predicted_class} ({confidence*100:.2f}% confidence)")
              
                    st.markdown("---")
                    st.markdown("### Treatment Tips")
                    st.info(tips['treatment'])

                    st.markdown("### Prevention Advice")
                    st.warning(tips['prevention'])

                    # Footer
                    st.markdown("""
                       <br><hr>
                       <div style='text-align: center; font-size: 14px;'>
                       Developed with ❤️ using Streamlit | 2025
                       </div>
                       """, unsafe_allow_html=True)
                else:
                    st.warning("Could not make a prediction. Please try another image.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check the console for more details.")
                print(f"Error: {traceback.format_exc()}")


if __name__ == "__main__":
    
    main()
