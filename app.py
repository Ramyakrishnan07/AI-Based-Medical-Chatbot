import os
import logging
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Import prediction model after Flask app is created
from disease_prediction import (
    load_model, get_description, get_precautions, get_severity_dict,
    preprocess_symptom, find_matching_symptoms, suggest_synonyms,
    calculate_severity, predict_disease, possible_diseases, symVONdisease
)

# Import chatbot module
from chatbot import MedicalChatbot

# Initialize chatbot
medical_chatbot = MedicalChatbot()

# Load necessary data and models
# Initialize global variables
knn_clf = None
dt_clf = None
all_symptoms = None
symptom_severity = None
description_list = None
precaution_dict = None
df_tr = None

# Store consultation history in memory (temporary storage)
consultations = []

def load_data():
    # Load KNN classifier and data
    global knn_clf, dt_clf, all_symptoms, symptom_severity, description_list, precaution_dict, df_tr
    
    try:
        knn_clf, dt_clf, all_symptoms, symptom_severity, description_list, precaution_dict, df_tr = load_model()
        app.logger.info("Models and data loaded successfully")
    except Exception as e:
        app.logger.error(f"Error loading models and data: {e}")
        raise

# Load data at startup
with app.app_context():
    load_data()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        # Reset session data for a new conversation
        session['symptoms'] = []
        session['current_step'] = 'initial'
        session['name'] = ''
        session['possible_diseases'] = []
        return render_template('chat.html')
    
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    message = data.get('message', '')
    current_step = session.get('current_step', 'initial')
    
    if current_step == 'initial':
        # First step: get user's name
        session['name'] = message
        session['current_step'] = 'symptom1'
        return jsonify({
            'response': f"Hello {message}! What's the main symptom you're experiencing?",
            'options': None
        })
    
    elif current_step == 'symptom1':
        # Process first symptom
        processed_symptom = preprocess_symptom(message)
        matched_symptoms = find_matching_symptoms(processed_symptom, all_symptoms)
        
        if matched_symptoms:
            # If we found matching symptoms, ask user to select one
            session['temp_matched_symptoms'] = matched_symptoms
            session['current_step'] = 'symptom1_selection'
            options = [{"id": i, "text": symptom} for i, symptom in enumerate(matched_symptoms)]
            return jsonify({
                'response': "I found these symptoms that match your description. Which one describes your condition best?",
                'options': options
            })
        else:
            # No matches found, try synonyms
            synonyms = suggest_synonyms(processed_symptom, all_symptoms)
            if synonyms:
                session['temp_matched_symptoms'] = synonyms
                session['current_step'] = 'symptom1_selection'
                options = [{"id": i, "text": symptom} for i, symptom in enumerate(synonyms)]
                return jsonify({
                    'response': "I couldn't find an exact match, but are you experiencing any of these?",
                    'options': options
                })
            else:
                # No synonyms either, ask for a different symptom
                return jsonify({
                    'response': "I'm sorry, I couldn't recognize that symptom. Could you describe it differently or mention another symptom?",
                    'options': None
                })
    
    elif current_step == 'symptom1_selection':
        # User selected from symptom options
        try:
            selected_idx = int(message)
            selected_symptom = session['temp_matched_symptoms'][selected_idx]
            session['symptoms'] = [selected_symptom]
            session['current_step'] = 'symptom2'
            return jsonify({
                'response': "Thank you. What's another symptom you're experiencing?",
                'options': None
            })
        except (ValueError, IndexError):
            # Handle invalid selection
            return jsonify({
                'response': "I didn't understand that selection. Please try again.",
                'options': [{"id": i, "text": symptom} for i, symptom in enumerate(session['temp_matched_symptoms'])]
            })
    
    elif current_step == 'symptom2':
        # Process second symptom
        processed_symptom = preprocess_symptom(message)
        matched_symptoms = find_matching_symptoms(processed_symptom, all_symptoms)
        
        if matched_symptoms:
            session['temp_matched_symptoms'] = matched_symptoms
            session['current_step'] = 'symptom2_selection'
            options = [{"id": i, "text": symptom} for i, symptom in enumerate(matched_symptoms)]
            return jsonify({
                'response': "I found these symptoms that match your description. Which one describes your condition best?",
                'options': options
            })
        else:
            synonyms = suggest_synonyms(processed_symptom, all_symptoms)
            if synonyms:
                session['temp_matched_symptoms'] = synonyms
                session['current_step'] = 'symptom2_selection'
                options = [{"id": i, "text": symptom} for i, symptom in enumerate(synonyms)]
                return jsonify({
                    'response': "I couldn't find an exact match, but are you experiencing any of these?",
                    'options': options
                })
            else:
                return jsonify({
                    'response': "I'm sorry, I couldn't recognize that symptom. Could you describe it differently?",
                    'options': None
                })
    
    elif current_step == 'symptom2_selection':
        # User selected from symptom options for second symptom
        try:
            selected_idx = int(message)
            selected_symptom = session['temp_matched_symptoms'][selected_idx]
            session['symptoms'].append(selected_symptom)
            
            # Calculate possible diseases based on current symptoms
            possible_dis = possible_diseases(session['symptoms'], df_tr)
            session['possible_diseases'] = possible_dis
            
            # Move to additional symptoms questioning
            session['current_step'] = 'additional_symptoms'
            session['asked_symptoms'] = session['symptoms'].copy()
            
            if possible_dis:
                # Get the most common symptoms for these diseases to ask about
                next_symptom_to_ask = None
                for disease in possible_dis:
                    disease_symptoms = symVONdisease(df_tr, disease)
                    for symptom in disease_symptoms:
                        if symptom not in session['asked_symptoms']:
                            next_symptom_to_ask = symptom
                            session['asked_symptoms'].append(symptom)
                            break
                    if next_symptom_to_ask:
                        break
                
                if next_symptom_to_ask:
                    return jsonify({
                        'response': f"Are you also experiencing {next_symptom_to_ask}?",
                        'options': [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]
                    })
                else:
                    # If no more symptoms to ask, proceed to duration
                    session['current_step'] = 'symptom_duration'
                    return jsonify({
                        'response': "How many days have you been experiencing these symptoms?",
                        'options': None
                    })
            else:
                # No possible diseases found
                session['current_step'] = 'symptom_duration'
                return jsonify({
                    'response': "Based on the symptoms provided, I couldn't identify a specific condition. Let me ask about duration to better assess.",
                    'options': None
                })
        
        except (ValueError, IndexError):
            return jsonify({
                'response': "I didn't understand that selection. Please try again.",
                'options': [{"id": i, "text": symptom} for i, symptom in enumerate(session['temp_matched_symptoms'])]
            })
    
    elif current_step == 'additional_symptoms':
        # Track the user's response to the last asked symptom
        last_asked = session.get('last_asked_symptom', None)

        if message.lower() == 'yes' and last_asked:
            if last_asked not in session['symptoms']:
                session['symptoms'].append(last_asked)
            session['possible_diseases'] = possible_diseases(session['symptoms'], df_tr)

        # Ask the next symptom that hasn't been asked yet
        next_symptom_to_ask = None
        for disease in session['possible_diseases']:
            disease_symptoms = symVONdisease(df_tr, disease)
            for symptom in disease_symptoms:
                if symptom not in session['asked_symptoms']:
                    next_symptom_to_ask = symptom
                    session['asked_symptoms'].append(symptom)
                    session['last_asked_symptom'] = symptom  # Track it for next round
                    break
            if next_symptom_to_ask:
                break

        if next_symptom_to_ask and len(session['asked_symptoms']) < 10:
            return jsonify({
                'response': f"Are you also experiencing {next_symptom_to_ask}?",
                'options': [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]
            })
        else:
            session['current_step'] = 'symptom_duration'
            return jsonify({
                'response': "How many days have you been experiencing these symptoms?",
                'options': None
            })

    
    elif current_step == 'symptom_duration':
        try:
            days = int(message)
            session['days'] = days
            
            # Make final prediction
            predicted_disease = predict_disease(session['symptoms'], all_symptoms, knn_clf)
            severity = calculate_severity(session['symptoms'], symptom_severity, days)
            
            # Save results to session
            session['predicted_disease'] = predicted_disease
            session['severity'] = severity
            session['description'] = get_description(predicted_disease, description_list)
            session['precautions'] = get_precautions(predicted_disease, precaution_dict)
            
            # Prepare final message
            if severity == 1:
                severity_msg = "Based on your symptoms and their duration, your condition seems serious. You should consult a doctor as soon as possible."
            else:
                severity_msg = "Your condition doesn't seem severe, but you should still take precautions."
            
            session['current_step'] = 'completed'
            
            return jsonify({
                'response': f"Based on your symptoms, you may have {predicted_disease}. {severity_msg}",
                'options': None,
                'redirect': '/results'
            })
            
        except ValueError:
            return jsonify({
                'response': "Please enter a valid number of days.",
                'options': None
            })
    
    # Default response if no condition is met
    return jsonify({
        'response': "I'm sorry, I couldn't process that. Let's start over.",
        'options': None,
        'redirect': '/chat'
    })

@app.route('/results')
def results():
    if 'predicted_disease' not in session:
        return redirect(url_for('chat'))
    
    # Store the consultation in memory
    consultation_id = len(consultations) + 1
    consultation = {
        'id': consultation_id,
        'name': session.get('name', ''),
        'symptoms': session.get('symptoms', []),
        'predicted_disease': session.get('predicted_disease', ''),
        'severity': session.get('severity', 0),
        'days_experiencing': session.get('days', 1),
        'description': session.get('description', ''),
        'precautions': session.get('precautions', []),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    consultations.append(consultation)
    
    # Store consultation ID in session
    session['consultation_id'] = consultation_id
    app.logger.info(f"Saved consultation {consultation_id} to memory")
    
    return render_template(
        'results.html', 
        name=session.get('name', ''),
        disease=session.get('predicted_disease', ''),
        description=session.get('description', ''),
        precautions=session.get('precautions', []),
        severity=session.get('severity', 0),
        symptoms=session.get('symptoms', [])
    )

@app.route('/medical-assistant', methods=['GET', 'POST'])
def medical_assistant():
    """Medical assistant route that uses the chatbot model"""
    if request.method == 'POST':
        user_query = request.form.get('query', '')
        if user_query:
            response = medical_chatbot.get_response(user_query)
            return render_template('medical_assistant.html', query=user_query, response=response)
    
    return render_template('medical_assistant.html')

@app.route('/api/medical-assistant', methods=['POST'])
def medical_assistant_api():
    """API endpoint for medical assistant chatbot"""
    data = request.json
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    
    response = medical_chatbot.get_response(user_query)
    return jsonify({'response': response})

@app.route('/admin/dashboard')
def admin_dashboard():
    """Admin dashboard showing in-memory consultation data"""
    # Count statistics
    total_consultations = len(consultations)
    diseases_count = {}
    
    for consultation in consultations:
        # Count diseases
        disease = consultation['predicted_disease']
        if disease in diseases_count:
            diseases_count[disease] += 1
        else:
            diseases_count[disease] = 1
    
    # Sort diseases by frequency
    top_diseases = sorted(diseases_count.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return render_template(
        'admin_dashboard.html',
        consultations=consultations,
        total_consultations=total_consultations,
        top_diseases=top_diseases
    )

@app.route('/start_over')
def start_over():
    # Clear session data
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)