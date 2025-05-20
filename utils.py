import json
import os
from datetime import datetime

def write_json(new_data, filename='DATA.json'):
    """
    Write new data to JSON file
    """
    try:
        # Check if file exists
        if os.path.isfile(filename):
            # Load existing data
            with open(filename, 'r') as file:
                file_data = json.load(file)
            # Append new data
            file_data.append(new_data)
        else:
            # Create new file with data as list
            file_data = [new_data]
        
        # Write updated data
        with open(filename, 'w') as file:
            json.dump(file_data, file, indent=4)
            
        return True
    except Exception as e:
        print(f"Error writing to JSON: {e}")
        return False

def save_user_data(name, symptoms, diagnosis, severity, days, recommendation):
    """
    Save user consultation data
    """
    # Create data object
    user_data = {
        'name': name,
        'symptoms': symptoms,
        'diagnosis': diagnosis,
        'severity': severity,
        'days_experiencing': days,
        'recommendation': recommendation,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Write to JSON file
    return write_json(user_data, 'data/user_consultations.json')

def format_symptom_for_display(symptom):
    """
    Format symptom string for display
    """
    # Replace underscores with spaces
    formatted = symptom.replace('_', ' ')
    # Capitalize each word
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    return formatted

def format_list_for_display(items):
    """
    Format a list for display with bullet points
    """
    if not items:
        return "None provided"
    
    return '<br>• ' + '<br>• '.join(items)