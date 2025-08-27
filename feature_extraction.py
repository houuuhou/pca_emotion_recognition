import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

LANDMARK_GROUPS = {
    'left_eyebrow': [70, 63, 105, 66, 107],
    'right_eyebrow': [300, 293, 334, 296, 336],
    'left_eye': [33, 160, 158, 133, 153, 144],
    'right_eye': [362, 385, 387, 263, 373, 380],
    'mouth': [61, 291, 39, 181, 17, 405, 321, 375, 393, 269],
    'nose': [1, 2, 98, 327, 358, 4, 5, 195, 197, 6, 122],
    'jawline': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397],
    'forehead': [10, 338, 297, 332, 284]
}

def preprocess_image(image):
    """Advanced image preprocessing"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def extract_geometric_features(landmarks, width, height):
    """Caractéristiques géométriques principales avec améliorations spécifiques aux expressions"""
    features = []
    feature_names = []
    
    # Caractéristiques des yeux
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    eye_closure = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
    eye_openness_diff = np.abs(eye_aspect_ratio(left_eye) - eye_aspect_ratio(right_eye))
    features.extend([eye_closure, eye_openness_diff])
    feature_names.extend(['ratio_fermeture_yeux', 'asymetrie_ouverture_yeux'])
    
    # Caractéristiques des sourcils
    left_brow_raise = np.mean(landmarks['left_eyebrow'][:,1]) - np.mean(left_eye[:,1])
    right_brow_raise = np.mean(landmarks['right_eyebrow'][:,1]) - np.mean(right_eye[:,1])
    brow_raise_asymmetry = np.abs(left_brow_raise - right_brow_raise)
    features.extend([left_brow_raise, right_brow_raise, brow_raise_asymmetry])
    feature_names.extend(['levée_sourcil_gauche', 'levée_sourcil_droit', 'asymetrie_levée_sourcils'])
    
    # Caractéristiques de la bouche
    mouth_width = np.linalg.norm(landmarks['mouth'][0] - landmarks['mouth'][4])
    mouth_openness = (landmarks['mouth'][9][1] - landmarks['mouth'][3][1]) / height
    mouth_corner_angle = np.arctan2(
        landmarks['mouth'][4][1] - landmarks['mouth'][0][1],
        landmarks['mouth'][4][0] - landmarks['mouth'][0][0]
    )
    features.extend([mouth_width, mouth_openness, mouth_corner_angle])
    feature_names.extend(['largeur_bouche', 'ouverture_bouche', 'angle_commissures'])
    
    # Autres caractéristiques
    nose_wrinkle = (landmarks['nose'][6][1] - landmarks['nose'][0][1]) / height
    forehead_tension = np.std(landmarks['forehead'][:,1])
    jaw_tightness = (landmarks['jawline'][3][0] - landmarks['jawline'][0][0]) / width
    lip_corner_puller = (landmarks['mouth'][4][0] - landmarks['mouth'][0][0]) / width
    brow_lowering = (landmarks['left_eyebrow'][2][1] - landmarks['left_eye'][1][1]) / height
    features.extend([nose_wrinkle, forehead_tension, jaw_tightness, 
                    lip_corner_puller, brow_lowering])
    feature_names.extend(['plis_nasaux', 'tension_frontale', 'tension_machoire',
                        'traction_commissures', 'abaissement_sourcils'])
    
    # Moments de Hu
    all_points = np.vstack([v for v in landmarks.values()])
    moments = cv2.moments(all_points.astype(np.float32))
    hu_moments = cv2.HuMoments(moments).flatten()
    features.extend(hu_moments[:7])
    feature_names.extend([f'moment_hu_{i+1}' for i in range(7)])
    
    return np.array(features), feature_names

def extract_single_image_features(image_path):
    """Process a single image and return features"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image = preprocess_image(image)
    height, width, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        landmark_coords = {}
        for region, indices in LANDMARK_GROUPS.items():
            landmark_coords[region] = np.array([
                [landmarks.landmark[i].x * width, landmarks.landmark[i].y * height] 
                for i in indices
            ])
        
        return extract_geometric_features(landmark_coords, width, height)
    return None, None


def eye_aspect_ratio(eye_points):
    vertical = (np.linalg.norm(eye_points[1] - eye_points[5]) + 
               np.linalg.norm(eye_points[2] - eye_points[4])) / 2
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    return vertical / horizontal

def extract_dataset_features(dataset_path):
    all_features = []
    all_labels = []
    feature_names = None  # Will be set from first successful extraction
    
    image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))])
    
    for img_name in image_files:
        img_path = os.path.join(dataset_path, img_name)
        features, names = extract_single_image_features(img_path)
        
        if features is not None:
            all_features.append(features)
            all_labels.append(img_name.split('_')[0].lower())
            if feature_names is None:  # Store feature names from first image
                feature_names = names
    
    return np.array(all_features), np.array(all_labels), feature_names

