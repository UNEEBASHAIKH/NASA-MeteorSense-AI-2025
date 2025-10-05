# 🌟 Asteroid Hazard Prediction - Smart Model Training
print(" Welcome to Asteroid Hazard Predictor!")
print("Let's train a smart model to identify potentially hazardous asteroids...\n")

# --- Step 1: Install missing packages  ---
print("STEP 0: Checking required packages...")

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.utils import class_weight
    from sklearn.preprocessing import LabelEncoder
    import seaborn as sns
    import matplotlib.pyplot as plt
    import joblib
    print(" Basic packages loaded successfully!")
except ImportError as e:
    print(f"❌ Missing basic package: {e}")
    print("Please run: pip install pandas numpy scikit-learn seaborn matplotlib joblib")
    exit()

# Try to import XGBoOST, offer alternatives if not available
try:
    import xgboost as xgb
    USE_XGBOOST = True
    print("✅ XGBoOST loaded - we'll use the advanced model!")
except ImportError:
    print("⚠️  XGBoOST not available - we'll use Random Forest instead")
    print("   To install XGBoOST later: pip install xgboost")
    USE_XGBOOST = False
    from sklearn.ensemble import RandomForestClassifier

# --- Step 2: Loading Our Data ---
print("\n📊 STEP 1: Loading and Understanding Our Asteroid Data")

# Load our asteroid database
try:
    # Try different possible file locations
    file_paths = [
        r"C:\Users\EXTECH\Downloads\NASA Near-Earth Object (NEO) Dataset.csv",
        "NASA Near-Earth Object (NEO) Dataset.csv",
        "neo_data.csv",
        "asteroid_data.csv"
    ]
    
    asteroid_data = None
    for file_path in file_paths:
        try:
            asteroid_data = pd.read_csv(file_path)
            print(f"✅ Found data file: {file_path}")
            break
        except:
            continue
    
    if asteroid_data is None:
        print("❌ Couldn't find the asteroid data file.")
        print("   Please make sure the CSV file is in the same folder as this script.")
        exit()
        
    print(f"   We have data on {len(asteroid_data)} asteroids")
    
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# Let's see what information we have about each asteroid
print(f"\n📋 What we know about each asteroid:")
print("First few columns:", list(asteroid_data.columns)[:8])  # Show first 8 columns
if len(asteroid_data.columns) > 8:
    print(f"   ... and {len(asteroid_data.columns) - 8} more columns")

# Clean up the data - remove any duplicates and fix column names
asteroid_data.columns = asteroid_data.columns.str.strip()
original_count = len(asteroid_data)
asteroid_data.drop_duplicates(inplace=True)
print(f"✅ Data cleaned up - removed {original_count - len(asteroid_data)} duplicates")

# --- Step 3: Finding Our Target - What Makes an Asteroid Dangerous? ---
print("\n🎯 STEP 2: Identifying Hazardous Asteroids")

# Look for the column that tells us if an asteroid is dangerous
danger_columns = [col for col in asteroid_data.columns 
                 if "hazard" in col.lower() or "potentially" in col.lower() or "pha" in col.lower()]

if danger_columns:
    danger_column = danger_columns[0]
    print(f"✅ Found our target: '{danger_column}' column")
else:
    print("❌ Couldn't automatically find hazard column.")
    print("   Available columns:")
    for i, col in enumerate(asteroid_data.columns):
        print(f"   {i+1:2d}. {col}")
    
    # Try common column names
    common_names = ['is_potentially_hazardous', 'is_hazardous', 'hazardous', 'pha']
    for name in common_names:
        if name in asteroid_data.columns:
            danger_column = name
            print(f"✅ Using '{danger_column}' as our target")
            break
    else:
        print("❓ Please check your data for a column that indicates hazardous asteroids")
        exit()

# Remove asteroids where we don't know if they're dangerous or not
original_count = len(asteroid_data)
asteroid_data.dropna(subset=[danger_column], inplace=True)
removed_count = original_count - len(asteroid_data)
if removed_count > 0:
    print(f"   Removed {removed_count} asteroids with missing safety info")

# --- Step 4: Choosing What Features to Look At ---
print("\n🔍 STEP 3: Choosing What Makes Asteroids Dangerous")

# These are the characteristics that might make asteroids dangerous
possible_features = [
    "absolute_magnitude_h",           # How bright the asteroid is
    "estimated_diameter_min_km",      # Smallest possible size
    "estimated_diameter_max_km",      # Largest possible size  
    "relative_velocity_km_s",         # How fast it's moving
    "miss_distance_au",               # How close it will come to Earth
    "orbiting_body",                  # What it's orbiting
    "diameter", "size", "velocity", "distance"  # Alternative names
]

# Only use features that actually exist in our data
features_we_have = []
for feature in possible_features:
    if feature in asteroid_data.columns:
        features_we_have.append(feature)

print(f"✅ We'll analyze these asteroid characteristics:")
for feature in features_we_have:
    print(f"   • {feature}")

if not features_we_have:
    print("⚠️  No standard features found. Let's use numeric columns:")
    numeric_cols = asteroid_data.select_dtypes(include=[np.number]).columns.tolist()
    # Remove the target column if it's in numeric cols
    if danger_column in numeric_cols:
        numeric_cols.remove(danger_column)
    features_we_have = numeric_cols[:6]  # Use first 6 numeric columns
    print("   Using:", features_we_have)

# Handle categorical data (like 'orbiting_body')
encoder = None
if "orbiting_body" in asteroid_data.columns:
    encoder = LabelEncoder()
    asteroid_data["orbiting_body_encoded"] = encoder.fit_transform(asteroid_data["orbiting_body"])
    if "orbiting_body" in features_we_have:
        features_we_have.remove("orbiting_body")
    features_we_have.append("orbiting_body_encoded")
    print("   ✓ Converted 'orbiting_body' to numbers the model can understand")

# --- Step 5: Preparing Our Training Data ---
print("\n🎓 STEP 4: Preparing to Train Our Model")

# Check for missing values in features
missing_values = asteroid_data[features_we_have].isnull().sum()
if missing_values.any():
    print("⚠️  Some features have missing values. Filling with median...")
    for feature in features_we_have:
        if asteroid_data[feature].isnull().any():
            if asteroid_data[feature].dtype in ['float64', 'int64']:
                asteroid_data[feature].fillna(asteroid_data[feature].median(), inplace=True)
            else:
                asteroid_data[feature].fillna(asteroid_data[feature].mode()[0], inplace=True)

# Separate our features (what we know) from our target (what we want to predict)
X = asteroid_data[features_we_have]  # Features: asteroid characteristics
y = asteroid_data[danger_column]     # Target: is it dangerous?

# Convert target to numbers if it's not already
if y.dtype == 'object' or y.dtype == 'bool':
    y = y.astype(int)
else:
    y = y.astype(int)

print(f"📊 Safety breakdown of our asteroids:")
safe_count = sum(y == 0)
dangerous_count = sum(y == 1)
total_count = len(y)

print(f"   🟢 Safe asteroids: {safe_count} ({safe_count/total_count*100:.1f}%)")
print(f"   🔴 Dangerous asteroids: {dangerous_count} ({dangerous_count/total_count*100:.1f}%)")

if dangerous_count == 0:
    print("❌ No dangerous asteroids found in the data!")
    exit()

# Handle class imbalance (we have many more safe asteroids than dangerous ones)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
danger_weight = class_weights[1] / class_weights[0]
print(f"⚖️  Adjusting for imbalance - dangerous asteroids will get {danger_weight:.1f}x more attention")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Keep the same proportion of dangerous/safe in both sets
)

print(f"📚 Training set: {len(X_train)} asteroids")
print(f"🧪 Testing set: {len(X_test)} asteroids")

# --- Step 6: Training Our Smart Model ---
print("\n🤖 STEP 5: Training Our Model")

if USE_XGBOOST:
    print("   Using XGBoOST (Advanced Model)...")
    # Create our smart model with corrected parameters
    smart_model = xgb.XGBClassifier(
        n_estimators=200,           # Reduced for faster training
        max_depth=6,                 # How complex patterns it can learn
        learning_rate=0.1,           # How fast it learns
        subsample=0.8,               # Use 80% of data for each cycle
        colsample_bytree=0.8,        # Use 80% of features for each cycle
        scale_pos_weight=danger_weight,  # Pay more attention to rare dangerous asteroids
        random_state=42,             # For reproducible results
        eval_metric='logloss',       # How we measure learning progress
        use_label_encoder=False
    )
else:
    print("   Using Random Forest (Good Alternative)...")
    smart_model = RandomForestClassifier(
        n_estimators=100,           # Reduced for faster training
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

print("🎯 Starting training... (This might take a moment)")

# Train the model with proper parameters
try:
    if USE_XGBOOST:
        # Correct XGBoOST training with proper parameter names
        smart_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,     # Correct parameter name
            verbose=False
        )
    else:
        smart_model.fit(X_train, y_train)
        
    print("✅ Training completed! Our model has learned asteroid patterns")
    
except Exception as e:
    print(f"⚠️  Training issue: {e}")
    print("   Trying simpler training approach...")
    # Fallback: simple training without early stopping
    smart_model.fit(X_train, y_train)
    print("✅ Training completed with fallback method!")

# --- Step 7: Testing How Well Our Model Learned ---
print("\n📊 STEP 6: Testing Our Model's Performance")

# Make predictions on test data
predictions = smart_model.predict(X_test)
prediction_probabilities = smart_model.predict_proba(X_test)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f" Overall Accuracy: {accuracy:.1%}")
print(f"   (Our model correctly identifies {accuracy:.1%} of asteroids)")

# Detailed performance report
print("\n Detailed Performance Report:")
print(classification_report(y_test, predictions, 
                          target_names=['Safe Asteroids', 'Dangerous Asteroids']))

# Confusion matrix - see what kinds of mistakes we make
print("🔍 Confusion Matrix (What we got right/wrong):")
conf_matrix = confusion_matrix(y_test, predictions)
print("           Predicted")
print("           Safe  Danger")
print(f"Actual Safe  {conf_matrix[0,0]:4d}  {conf_matrix[0,1]:4d}")
print(f"       Danger {conf_matrix[1,0]:4d}  {conf_matrix[1,1]:4d}")

# Visualize the confusion matrix
try:
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Safe', 'Predicted Dangerous'],
                yticklabels=['Actually Safe', 'Actually Dangerous'])
    plt.title('Asteroid Hazard Prediction Results')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"⚠️  Could not display chart: {e}")

# --- Step 8: Understanding What Our Model Learned ---
print("\n🧠 STEP 7: What Makes Asteroids Dangerous?")
print("   (Which features are most important for predicting hazards)")

# Get feature importance
feature_importance = smart_model.feature_importances_
sorted_indices = np.argsort(feature_importance)

# Create a nice visualization
try:
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features_we_have)))
    bars = plt.barh(range(len(sorted_indices)), feature_importance[sorted_indices], color=colors)

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', ha='left', va='center')

    plt.yticks(range(len(sorted_indices)), np.array(features_we_have)[sorted_indices])
    model_type = "XGBoOST" if USE_XGBOOST else "Random Forest"
    plt.title(f'What Makes Asteroids Dangerous?\n(Feature Importance - {model_type})')
    plt.xlabel('Importance Score')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"⚠️  Could not display feature importance chart: {e}")

# Print importance in order
print("\n📋 Feature Importance Ranking:")
for i, idx in enumerate(sorted_indices[::-1]):  # Reverse for highest first
    print(f"   {i+1}. {features_we_have[idx]}: {feature_importance[idx]:.3f}")

# --- Step 9: Saving Our Trained Model ---
print("\n💾 STEP 8: Saving Our Smart Model")
# Save the trained model
model_type_name = "xgboost" if USE_XGBOOST else "random_forest"
model_filename = f'asteroid_hazard_predictor_{model_type_name}.pkl'

try:
    joblib.dump(smart_model, model_filename)
    print(f"✅ Model saved as '{model_filename}'")
    
    # Save other important information
    joblib.dump(features_we_have, 'feature_names.pkl')
    print("✅ Feature names saved")

    if encoder:
        joblib.dump(encoder, 'label_encoder.pkl')
        print("✅ Label encoder saved")
        
    print("   We can now use this model to predict new asteroids!")
    
except Exception as e:
    print(f"⚠️  Could not save model: {e}")

# --- Final Summary ---
print("\n" + "="*50)
print("🎉 TRAINING COMPLETE! SUMMARY")
print("="*50)
print(f"✅ Data: {len(asteroid_data)} asteroids analyzed")
print(f"✅ Features: {len(features_we_have)} characteristics considered")
print(f"✅ Model: {model_type_name.upper()} trained successfully")
print(f"✅ Accuracy: {accuracy:.1%} on test data")
print(f"✅ Saved: Model ready for future predictions")

if accuracy > 0.85:
    print("🎉 Excellent performance!")
elif accuracy > 0.75:
    print(" Good performance!")
else:
    print("💡 Model could use improvement - consider collecting more data")

print("\n You now have a smart asteroid hazard predictor!")
print("   Use it to analyze new asteroids and keep Earth safe! 🌍")
print("="*50)

# --- Simple Prediction Example ---
print("\n QUICK PREDICTION EXAMPLE:")
print("Here's how to check if a new asteroid is dangerous:")

# Create a simple example based on actual data statistics
example_features = {}
for feature in features_we_have:
    if feature in asteroid_data.columns:
        example_features[feature] = [asteroid_data[feature].median()]

try:
    example_df = pd.DataFrame(example_features)
    prediction = smart_model.predict(example_df)[0]
    probability = smart_model.predict_proba(example_df)[0][1]
    
    if prediction == 1:
        print(f"🚨 Example asteroid: DANGEROUS ({probability:.1%} confidence)")
    else:
        print(f" Example asteroid: SAFE ({probability:.1%} confidence)")
        
except Exception as e:
    print(f"⚠️  Could not run example prediction: {e}")

print("\n All done! Your asteroid hazard predictor is ready to use!")