import os
import pandas as pd
import openai
import logging
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import nltk
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import google.generativeai as genai
import requests  # Add this import for Wikipedia API interaction

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API keys from environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY", "")
google_api_key = "AIzaSyCVhHXuHtahtlhiDrb7OQj5qZ8vBSvJ1h0" 
if google_api_key:
    genai.configure(api_key=google_api_key)

class MedicalChatbot:
    def __init__(self, dataset_path="data/healifyLLM-QA-scraped-dataset.csv"):
        self.df = None
        self.sentence_model = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.qa_pipeline = None
        self.initialized = False
        self.dataset_path = dataset_path
        self.initialize()
    
    def initialize(self):
        """Initialize the chatbot with dataset and models"""
        try:
            # Load QA dataset
            self.df = pd.read_csv(self.dataset_path)
            
            # Ensure correct column names exist
            if not all(col in self.df.columns for col in ["disease", "question", "answer"]):
                logger.error("CSV file must contain 'disease', 'question', and 'answer' columns.")
                return False
            
            # Load sentence transformer for semantic matching
            logger.info("Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load Hugging Face model for fallback QA
            try:
                logger.info("Loading Hugging Face QA model...")
                self.qa_pipeline = pipeline('question-answering', model='DistilBERT')
            except Exception as e:
                logger.error(f"Failed to load Hugging Face QA model: {e}")
                self.qa_pipeline = None
            
            self.initialized = True
            logger.info("Chatbot models and data loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            return False
    
    def get_embeddings(self, texts):
        """Get embeddings for a list of texts using the sentence transformer"""
        if not self.sentence_model:
            logger.error("Sentence transformer model not loaded")
            return None
        
        try:
            return self.sentence_model.encode(texts, convert_to_tensor=True)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None
    
    def semantic_search(self, query, threshold=0.5):
        """Perform semantic search against the QA dataset"""
        if not self.initialized or not self.sentence_model:
            return None, 0
        
        try:
            # Get embeddings
            query_embedding = self.sentence_model.encode(query, convert_to_tensor=True)
            question_embeddings = self.sentence_model.encode(self.df["question"].tolist(), convert_to_tensor=True)
            
            # Calculate similarities
            similarities = util.cos_sim(query_embedding, question_embeddings)[0]
            
            # Get best match
            best_idx = torch.argmax(similarities).item()
            best_score = similarities[best_idx].item()
            
            logger.info(f"Best semantic match score: {best_score}")
            
            if best_score >= threshold:
                return self.df.iloc[best_idx]["answer"], best_score
            
            return None, best_score
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return None, 0
    
    def query_wikipedia(self, user_query):
        url = "https://en.wikipedia.org/w/api.php"  # Endpoint for English Wikipedia
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,  # Get only the introductory section
            "explaintext": True,  # Return plain text instead of HTML
            "titles": user_query,
            }
        # """Query Wikipedia API for relevant information"""
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            pages = data["query"]["pages"]
            if pages:
                page_id = next(iter(pages))  # Get the first page ID
                if page_id != "-1":  # Check if the page exists
                    return pages[page_id]["extract"]
                else:
                    return "Page not found."
            else:
                return "No response from Wikipedia API."
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None
        except KeyError:
            return "Invalid response format from Wikipedia API."

    
    def query_huggingface(self, user_query):
        """Use Hugging Face models for question answering"""
        if not self.qa_pipeline:
            logger.warning("Hugging Face QA pipeline not available")
            return None
        
        try:
            # Create a context from top disease descriptions
            disease_context = " ".join(self.df["answer"].head(10).tolist())
            
            # Use the QA pipeline
            result = self.qa_pipeline(question=user_query, context=disease_context)
            
            # Check if the answer is meaningful
            if result['score'] > 0.1 and len(result['answer']) > 5:
                return result['answer']
            
            return None
        except Exception as e:
            logger.error(f"Hugging Face QA failed: {str(e)}")
            return None
    
    def get_response(self, user_query):
        """Get response using semantic matching with API fallbacks"""
        # Check if models are initialized
        if not self.initialized:
            success = self.initialize()
            if not success:
                # Try API fallbacks if initialization fails
                api_response = self.query_openai(user_query) or self.query_gemini(user_query)
                return api_response or "I'm sorry, I couldn't process your request at this time."
        
        # First try semantic search with sentence transformers
        answer, score = self.semantic_search(user_query)
        if answer:
            logger.info("Found answer using semantic search")
            return answer
        
        # Finally try Hugging Face model
        hf_answer = self.query_huggingface(user_query)
        if hf_answer:
            logger.info("Found answer using Hugging Face model")
            return hf_answer
        
        # Try Wikipedia if no close match found
        logger.info("No close match found in Huggingface, trying Wikipedia API")
        title = user_query
        summary = self.query_wikipedia(title)
        if summary:
            logger.info("Found answer using Wikipedia")
            return summary
        
        # Try Google Gemini API if Wikipedia fails
        logger.info("Wikipedia API failed, trying Google Gemini API")
        gemini_response = self.query_gemini(user_query)
        if gemini_response:
            logger.info("Found answer using Google Gemini API")
            return gemini_response
        
        # Try OpenAI API if Gemini fails
        logger.info("Gemini API failed, trying OpenAI API")
        openai_response = self.query_openai(user_query)
        if openai_response:
            logger.info("Found answer using OpenAI API")
            return openai_response
        

        
        # Last resort fallback
        return "I'm sorry, I don't have enough information to answer that question accurately."
