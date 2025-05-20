import os
from app import app, db
from models import UserConsultation, Feedback

# Create tables
with app.app_context():
    db.create_all()
    print("Database tables created successfully!")