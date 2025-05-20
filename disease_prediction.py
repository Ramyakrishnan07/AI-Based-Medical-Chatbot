import os
import numpy as np
import pandas as pd
import re
import csv
import pickle
from nltk.corpus import wordnet as wn
import nltk

# Ensure NLTK data is downloaded
nltk.download('wordnet', quiet=True)

def clean_symptom(sym):
    """Clean symptom string by removing underscores and normalizing"""
    return sym.replace('_', ' ').strip().lower()

def preprocess_symptom(symptom):
    """Process symptom input to standardized format"""
    # Convert to lowercase and remove punctuation
    symptom = re.sub(r'[^\w\s]', '', symptom.lower())
    return symptom

def jaccard_similarity(str1, str2):
    """Calculate Jaccard similarity between two strings"""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    
    return float(intersection) / union if union > 0 else 0

def find_matching_symptoms(input_symptom, all_symptoms, threshold=0.2):
    """Find symptoms similar to input using syntactic similarity"""
    matches = []
    input_symptom = preprocess_symptom(input_symptom)
    
    for symptom in all_symptoms:
        # Calculate similarity between input symptom and current symptom
        cleaned_symptom = clean_symptom(symptom)
        similarity = jaccard_similarity(input_symptom, cleaned_symptom)
        
        if similarity >= threshold:
            matches.append(symptom)
    
    return matches[:5]  # Return top 5 matches to avoid overwhelming the user

def semantic_similarity(doc1, doc2):
    """Calculate semantic similarity between two symptom descriptions"""
    # This is a simplified version using wordnet
    try:
        # Get synsets for both documents
        synsets1 = [wn.synsets(word)[0] for word in doc1.split() if wn.synsets(word)]
        synsets2 = [wn.synsets(word)[0] for word in doc2.split() if wn.synsets(word)]
        
        # Calculate the maximum similarity between each pair of synsets
        max_similarities = []
        for syn1 in synsets1:
            similarities = [syn1.path_similarity(syn2) or 0 for syn2 in synsets2]
            if similarities:
                max_similarities.append(max(similarities))
        
        # Return the average of the maximum similarities
        if max_similarities:
            return sum(max_similarities) / len(max_similarities)
    except:
        pass
    
    return 0  # Default to 0 similarity

def suggest_synonyms(symptom, all_symptoms):
    """Suggest possible symptoms based on synonyms of input symptom"""
    # Clean the input symptom
    symptom = preprocess_symptom(symptom)
    symptom_words = symptom.split()
    
    potential_matches = []
    
    # Find potential matches using synonym expansion
    for word in symptom_words:
        # Get synonyms from WordNet
        synsets = wn.synsets(word)
        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                
                # Use each synonym to find matching symptoms
                for s in all_symptoms:
                    if synonym in clean_symptom(s) and s not in potential_matches:
                        potential_matches.append(s)
    
    return potential_matches[:5]  # Limit to top 5 to avoid overwhelming the user

def one_hot_encode_symptoms(symptoms, all_symptoms):
    """Convert symptom list to one-hot encoded vector"""
    # Initialize vector with zeros
    vector = np.zeros(len(all_symptoms))
    
    # Set 1 for symptoms that are present
    for symptom in symptoms:
        if symptom in all_symptoms:
            idx = all_symptoms.index(symptom)
            vector[idx] = 1
    
    return vector

def symVONdisease(df, disease):
    """Get symptoms associated with a disease"""
    disease_symptoms = []
    for index, row in df.iterrows():
        if row['prognosis'] == disease:
            # Get all columns with value 1 (present symptoms)
            present_symptoms = [col for col in df.columns[:-1] if row[col] == 1]
            disease_symptoms.extend(present_symptoms)
    
    return list(set(disease_symptoms))  # Remove duplicates

def possible_diseases(symptoms, df):
    """Get list of possible diseases based on symptoms"""
    # Get all unique diseases in the dataset
    all_diseases = df['prognosis'].unique()
    
    # Calculate which diseases include the given symptoms
    possible_diseases = []
    for disease in all_diseases:
        disease_symptoms = symVONdisease(df, disease)
        # If at least one symptom matches, add to possible diseases
        if any(symptom in disease_symptoms for symptom in symptoms):
            possible_diseases.append(disease)
    
    return possible_diseases

def predict_disease(symptoms, all_symptoms, model):
    """Predict disease using the trained model"""
    # One-hot encode the symptoms
    input_vector = one_hot_encode_symptoms(symptoms, all_symptoms)
    
    # Reshape for sklearn
    input_data = input_vector.reshape(1, -1)
    
    # Make prediction
    return model.predict(input_data)[0]

def calculate_severity(symptoms, severity_dict, days):
    """Calculate severity of condition based on symptoms and duration"""
    severity_score = 0
    
    # Calculate symptom severity
    for symptom in symptoms:
        if symptom in severity_dict:
            severity_score += severity_dict[symptom]
    
    # Consider days experiencing symptoms
    if days > 7:  # If symptoms persisting for more than a week
        severity_score += 2
    
    # Threshold decision (can be adjusted)
    # Return 1 for high severity, 0 for low
    if severity_score > len(symptoms) * 2:  # If avg severity > 2
        return 1
    return 0

def get_description(disease, description_dict):
    """Get description of a disease"""
    return description_dict.get(disease, "No description available for this condition.")

def get_precautions(disease, precaution_dict):
    """Get precautions for a disease"""
    return precaution_dict.get(disease, ["No specific precautions listed for this condition."])

def get_severity_dict():
    """Load symptom severity data"""
    severity_dict = {}
    severity_file = os.path.join('data', 'symptom_severity.csv')
    
    try:
        with open(severity_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) >= 2:  # Ensure row has at least 2 elements
                    symptom, severity = row[0], int(row[1])
                    severity_dict[symptom] = severity
    except Exception as e:
        print(f"Error loading severity data: {e}")
    
    return severity_dict

def get_description_dict():
    """Load disease description data"""
    description_dict = {}
    description_file = os.path.join('data', 'symptom_Description.csv')
    
    try:
        with open(description_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) >= 2:  # Ensure row has at least 2 elements
                    disease, description = row[0], row[1]
                    description_dict[disease] = description
    except Exception as e:
        print(f"Error loading description data: {e}")
    
    return description_dict

def get_precaution_dict():
    """Load disease precaution data"""
    precaution_dict = {}
    precaution_file = os.path.join('data', 'symptom_precaution.csv')
    
    try:
        with open(precaution_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) >= 2:  # Ensure row has disease and at least one precaution
                    disease = row[0]
                    precautions = [p for p in row[1:] if p]  # Get non-empty precautions
                    precaution_dict[disease] = precautions
    except Exception as e:
        print(f"Error loading precaution data: {e}")
    
    return precaution_dict

def load_model():
    """Load trained model and necessary data files"""
    try:
        # Load KNN classifier
        with open(os.path.join('models', 'disease_knn.pkl'), 'rb') as f:
            knn_clf = pickle.load(f)
        
        # Load decision tree classifier as backup
        with open(os.path.join('models', 'disease_dt.pkl'), 'rb') as f:
            dt_clf = pickle.load(f)
        
        # Load training data to get symptoms list
        train_data = pd.read_csv(os.path.join('data', 'Training.csv'))
        
        # Get list of all symptoms (all columns except the last one which is 'prognosis')
        all_symptoms = list(train_data.columns[:-1])
        
        # Load severity dictionary
        severity_dict = get_severity_dict()
        
        # Load disease descriptions
        description_dict = get_description_dict()
        
        # Load precautions
        precaution_dict = get_precaution_dict()
        
        return knn_clf, dt_clf, all_symptoms, severity_dict, description_dict, precaution_dict, train_data
    
    except Exception as e:
        print(f"Error loading model and data: {e}")
        raise