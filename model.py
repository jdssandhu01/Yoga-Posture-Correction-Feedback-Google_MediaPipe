# --- 1. SETUP: IMPORT LIBRARIES ---

print("STEP 1: IMPORTING LIBRARIES...")
import os
import cv2
import warnings
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Added for saving the model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
print("Libraries imported.")


# --- 2. DEFINE PATHS & VERIFY ---

print("\nSTEP 2: VERIFYING DATASET PATH...")
project_path = '.'
dataset_path = os.path.join(project_path, 'dataset')
csv_output_path = os.path.join(project_path, 'yoga_poses_data.csv')
plots_dir = os.path.join(project_path, 'plots')
os.makedirs(plots_dir, exist_ok=True)

if not os.path.exists(dataset_path):
    print(f"ERROR: The 'dataset' folder was not found.")
    exit()
print(f"Dataset folder found at: '{os.path.abspath(dataset_path)}'")


# --- 3. PREPROCESSING (CREATE CSV FROM IMAGES) ---

print("\nSTEP 3: PREPROCESSING ALL IMAGES TO CREATE CSV...")
if os.path.exists(csv_output_path):
    print("Preprocessed CSV file found. Loading existing data.")
    df = pd.read_csv(csv_output_path)
else:
    print("Starting image processing pipeline... (This may take several minutes)")

    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)
    data_list = []
    poses_to_process = [p for p in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, p))]

    for pose_name in tqdm(poses_to_process, desc="Processing Poses"):
        pose_path = os.path.join(dataset_path, pose_name)

        for correctness_folder in os.listdir(pose_path):
            folder_name_lower = correctness_folder.lower()
            if "right steps" in folder_name_lower:
                correctness_value = 1
            elif "wrong steps" in folder_name_lower:
                correctness_value = 0
            else:
                continue

            base_correctness_path = os.path.join(pose_path, correctness_folder)
            if not os.path.isdir(base_correctness_path): continue

            for subfolder in os.listdir(base_correctness_path):
                subfolder_path = os.path.join(base_correctness_path, subfolder)
                if not os.path.isdir(subfolder_path): continue

                for image_name in os.listdir(subfolder_path):
                    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    image_path = os.path.join(subfolder_path, image_name)
                    try:
                        img = cv2.imread(image_path)
                        if img is None: continue
                        results = pose_model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            get_coords = lambda lm: [landmarks[lm.value].x, landmarks[lm.value].y]

                            shoulder_l, elbow_l, wrist_l = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER), get_coords(mp_pose.PoseLandmark.LEFT_ELBOW), get_coords(mp_pose.PoseLandmark.LEFT_WRIST)
                            hip_l, knee_l, ankle_l = get_coords(mp_pose.PoseLandmark.LEFT_HIP), get_coords(mp_pose.PoseLandmark.LEFT_KNEE), get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)
                            shoulder_r, elbow_r, wrist_r = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER), get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW), get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)
                            hip_r, knee_r, ankle_r = get_coords(mp_pose.PoseLandmark.RIGHT_HIP), get_coords(mp_pose.PoseLandmark.RIGHT_KNEE), get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)

                            data_list.append({
                                "pose": pose_name,
                                "correctness": correctness_value,
                                "angle_elbow_l": calculate_angle(shoulder_l, elbow_l, wrist_l), "angle_shoulder_l": calculate_angle(hip_l, shoulder_l, elbow_l),
                                "angle_hip_l": calculate_angle(shoulder_l, hip_l, knee_l), "angle_knee_l": calculate_angle(hip_l, knee_l, ankle_l),
                                "angle_elbow_r": calculate_angle(shoulder_r, elbow_r, wrist_r), "angle_shoulder_r": calculate_angle(hip_r, shoulder_r, elbow_r),
                                "angle_hip_r": calculate_angle(shoulder_r, hip_r, knee_r), "angle_knee_r": calculate_angle(hip_r, knee_r, ankle_r),
                            })
                    except Exception as e:
                        print(f"\nCould not process image {image_path}: {e}")
    pose_model.close()
    df = pd.DataFrame(data_list)
    df.to_csv(csv_output_path, index=False)
    print(f"\nPreprocessing complete! {len(df)} samples saved to 'yoga_poses_data.csv'.")


# --- 4. EXPERIMENT 1: MODEL COMPARISON & SAVING ---

print("\n" + "="*80)
print("EXPERIMENT 1: COMPARING SPECIALIST MODELS (RF, SVM, XGBoost)")
print("="*80)

models_to_compare = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}
comparison_results = []
all_poses_list = df['pose'].unique()

for pose_name in all_poses_list:
    print(f"\n--- Processing Pose: {pose_name} ---")
    pose_df = df[df['pose'] == pose_name].copy()
    X = pose_df.drop(['pose', 'correctness'], axis=1)
    y = pose_df['correctness']
    if len(y.unique()) < 2: continue
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    for model_name, model in models_to_compare.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        comparison_results.append({"Pose": pose_name, "Model": model_name, "Accuracy": accuracy})
        print(f"  - {model_name} Accuracy: {accuracy:.2%}")

        # ADDED: Save the best model for the live demo
        if pose_name == 'Tadasana' and model_name == 'XGBoost':
            joblib.dump(model, 'tadasana_xgboost_model.pkl')
            print(f"--- Saved {model_name} model for {pose_name} as tadasana_xgboost_model.pkl ---")

comparison_df = pd.DataFrame(comparison_results)
print("\n--- Model Comparison Summary ---")
pivot_df = comparison_df.pivot(index='Pose', columns='Model', values='Accuracy')
print(pivot_df.style.format("{:.2%}").to_string())


# --- 5. EXPERIMENT 2: FEATURE IMPORTANCE ANALYSIS ---

print("\n" + "="*80)
print("EXPERIMENT 2: ANALYZING FEATURE IMPORTANCE FOR 'Tadasana'")
print("="*80)

pose_df_feat = df[df['pose'] == 'Tadasana'].copy()
X_feat = pose_df_feat.drop(['pose', 'correctness'], axis=1)
y_feat = pose_df_feat['correctness']
model_feat = RandomForestClassifier(random_state=42)
model_feat.fit(X_feat, y_feat)
feature_df = pd.DataFrame({'Feature': X_feat.columns, 'Importance': model_feat.feature_importances_}).sort_values(by='Importance', ascending=False)
print(feature_df)

# Plot and save Tadasana feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title('Feature Importance for Tadasana')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'tadasana_feature_importance.png'), dpi=200)
plt.close()


# --- 6. EXPERIMENT 3: GLOBAL MODEL ---

print("\n" + "="*80)
print("EXPERIMENT 3: TRAINING A SINGLE 'GLOBAL' MODEL")
print("="*80)

df_global = pd.get_dummies(df, columns=['pose'], drop_first=True)
X_global = df_global.drop('correctness', axis=1)
y_global = df_global['correctness']
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_global, y_global, test_size=0.2, random_state=42, stratify=y_global)

model_global = RandomForestClassifier(random_state=42)
model_global.fit(X_train_g, y_train_g)
accuracy_global = model_global.score(X_test_g, y_test_g)
print(f"Global Model Overall Accuracy: {accuracy_global:.2%}")

print("→ Generating RandomForest accuracy plot for Global model...")
rf_cost = RandomForestClassifier(random_state=42, warm_start=True)
n_list = list(range(25, 301, 25))
train_acc_list, val_acc_list = [], []
for n_estimators in n_list:
    rf_cost.set_params(n_estimators=n_estimators)
    rf_cost.fit(X_train_g, y_train_g)
    train_pred = rf_cost.predict(X_train_g)
    val_pred = rf_cost.predict(X_test_g)
    train_acc_list.append(accuracy_score(y_train_g, train_pred))
    val_acc_list.append(accuracy_score(y_test_g, val_pred))

# Global RandomForest accuracy plot
plt.figure(figsize=(5, 3.5))
plt.plot(n_list, train_acc_list, label='Training accuracy')
plt.plot(n_list, val_acc_list, label='Validation accuracy')
plt.xlabel('Trees (n_estimators)')
plt.ylabel('Accuracy')
plt.title('Global - RandomForest Accuracy vs Trees')
plt.ylim(0.0, 1.0)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'global_rf_accuracy_vs_trees.png'), dpi=200)
plt.close()


# --- 7. EXPERIMENT 4: HYPERPARAMETER TUNING ---

print("\n" + "="*80)
print("EXPERIMENT 4: HYPERPARAMETER TUNING FOR 'Tadasana' (This may take time)")
print("="*80)

pose_df_tune = df[df['pose'] == 'Tadasana'].copy()
X_tune = pose_df_tune.drop(['pose', 'correctness'], axis=1)
y_tune = pose_df_tune['correctness']
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_tune, y_tune, test_size=0.2, random_state=42, stratify=y_tune)

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train_t, y_train_t)

print(f"\nBest parameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
accuracy_tuned = best_model.score(X_test_t, y_test_t)
print(f"Tuned Model Accuracy for 'Tadasana': {accuracy_tuned:.2%}")


# --- 8. FINAL SCRIPT COMPLETION ---

# --- 8.1 VISUALIZATION: TADASANA ACCURACY ACROSS ALGORITHMS ---

print("\n" + "="*80)
print("VISUALIZATION: Tadasana accuracies for RF, SVM, XGBoost")
print("="*80)

if 'X_train_t' in locals() and 'X_test_t' in locals():
    models_tadasana = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    algo_names = []
    algo_accuracies = []
    for algo_name, algo in models_tadasana.items():
        algo.fit(X_train_t, y_train_t)
        acc = accuracy_score(y_test_t, algo.predict(X_test_t))
        algo_names.append(algo_name)
        algo_accuracies.append(acc)
        print(f"{algo_name}: {acc:.2%}")

    plt.figure(figsize=(6, 4))
    bars = plt.bar(algo_names, algo_accuracies, color=['#4C78A8', '#F58518', '#54A24B'])
    plt.ylim(0.0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Tadasana - Algorithm Accuracy Comparison')
    for bar, val in zip(bars, algo_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2%}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tadasana_algorithms_accuracy.png'), dpi=200)
    plt.close()

print("\n" + "="*80)
print("VISUALIZATION: Vajrasana accuracies for RF, SVM, XGBoost")
print("="*80)

pose_df_vaj = df[df['pose'] == 'Vajrasana'].copy()
if not pose_df_vaj.empty and pose_df_vaj['correctness'].nunique() > 1:
    X_vaj = pose_df_vaj.drop(['pose', 'correctness'], axis=1)
    y_vaj = pose_df_vaj['correctness']
    X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
        X_vaj, y_vaj, test_size=0.2, random_state=42, stratify=y_vaj
    )

    models_vaj = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    algo_names_v = []
    algo_accuracies_v = []
    for algo_name, algo in models_vaj.items():
        algo.fit(X_train_v, y_train_v)
        acc_v = accuracy_score(y_test_v, algo.predict(X_test_v))
        algo_names_v.append(algo_name)
        algo_accuracies_v.append(acc_v)
        print(f"{algo_name}: {acc_v:.2%}")

    plt.figure(figsize=(6, 4))
    bars_v = plt.bar(algo_names_v, algo_accuracies_v, color=['#4C78A8', '#F58518', '#54A24B'])
    plt.ylim(0.0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Vajrasana - Algorithm Accuracy Comparison')
    for bar, val in zip(bars_v, algo_accuracies_v):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2%}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'vajrasana_algorithms_accuracy.png'), dpi=200)
    plt.close()
else:
    print("Skipping Vajrasana accuracy plot: insufficient class diversity or no samples.")

print("\n\n" + "="*80)
print("ALL EXPERIMENTS COMPLETE.")
print("="*80)