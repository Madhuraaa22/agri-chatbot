import json
import re
import time
import os
from typing import Dict, List, Optional
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TerminalAgriChatbot:
    def __init__(self, knowledge_base_path: str, training_data_path: str = None):
        self.knowledge_base_path = knowledge_base_path
        self.training_data_path = training_data_path
        self.knowledge_base = self.load_knowledge_base()
        self.training_data = self.load_training_data() if training_data_path else None
        self.lemmatizer = WordNetLemmatizer()
        self._setup_nltk()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        self.question_vectors = None
        self.indexed_data = []
        if self.training_data:
            self._build_search_indices()
        self.keywords = {
            'disease': ['disease', 'pest', 'problem', 'spots', 'leaves', 'sick', 'infection', 'fungus', 'bacteria', 'blight', 'rot', 'wilt'],
            'irrigation': ['water', 'irrigation', 'watering', 'drought', 'moisture', 'dry'],
            'fertilizer': ['fertilizer', 'nutrient', 'fertilization', 'nitrogen', 'phosphorus', 'potassium', 'compost', 'urea', 'ssp', 'mop'],
            'government': ['scheme', 'subsidy', 'government', 'yojana', 'pm-kisan', 'loan', 'kisan', 'credit'],
            'weather': ['weather', 'rain', 'monsoon', 'summer', 'winter', 'climate'],
            'pest': ['pest', 'insect', 'bug', 'caterpillar', 'aphid', 'control', 'spray', 'malathion'],
            'market': ['price', 'market', 'sell', 'selling', 'mandi', 'enam'],
            'crop': ['tomato', 'wheat', 'rice', 'crop', 'plant', 'farming', 'variety', 'seed'],
            'animal_husbandry': ['cow', 'goat', 'buffalo', 'poultry', 'cattle', 'livestock', 'milk', 'dairy'],
            'treatment': ['treatment', 'medicine', 'dose', 'bolus', 'injection', 'powder']
        }

    def _setup_nltk(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading language data... Please wait.")
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)

    def load_knowledge_base(self) -> Dict:
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Knowledge base file not found at {self.knowledge_base_path}")
            return {}

    def load_training_data(self) -> Dict:
        try:
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Loaded {data['statistics']['total_records']:,} expert answers")
                return data
        except FileNotFoundError:
            print(f"Warning: Training data not found at {self.training_data_path}")
            return None
        except Exception as e:
            print(f"Warning: Could not load training data: {e}")
            return None

    def _build_search_indices(self):
        if not self.training_data:
            return
        try:
            print("Building search engine... Please wait.")
            questions = []
            self.indexed_data = []
            for record in self.training_data['training_data']:
                questions.append(record['question'])
                self.indexed_data.append(record)
            self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
            self.question_vectors = self.tfidf_vectorizer.fit_transform(questions)
            print("Search engine ready!")
        except Exception as e:
            print(f"Warning: Could not build search indices: {e}")

    def preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        return processed_tokens

    def detect_intent(self, text: str) -> str:
        processed_tokens = self.preprocess_text(text)
        category_scores = {}
        for category, keywords in self.keywords.items():
            score = 0
            for token in processed_tokens:
                for keyword in keywords:
                    if fuzz.ratio(token, keyword) > 80:
                        score += 1
            category_scores[category] = score
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general'

    def extract_crop_name(self, text: str) -> Optional[str]:
        crops = ['tomato', 'wheat', 'rice', 'corn', 'potato', 'onion', 'garlic', 'sugarcane', 'coconut', 'brinjal', 'cucumber', 'bottle gourd', 'bitter gourd', 'ridge gourd', 'chilli', 'cabbage', 'cauliflower', 'okra', 'bhendi', 'knolkhol', 'spinach', 'french bean', 'black gram', 'rajma', 'pumpkin', 'mango', 'lemon']
        text_lower = text.lower()
        for crop in crops:
            if crop in text_lower:
                return crop
        processed_text = ' '.join(self.preprocess_text(text))
        best_match = process.extractOne(processed_text, crops)
        if best_match and best_match[1] > 70:
            return best_match[0]
        return None

    def search_training_data(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.training_data or not self.tfidf_vectorizer:
            return []
        try:
            query_vector = self.tfidf_vectorizer.transform([query.lower()])
            similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    result = self.indexed_data[idx].copy()
                    result['similarity_score'] = similarities[idx]
                    results.append(result)
            return results
        except Exception as e:
            return []

    def get_response_from_training_data(self, user_input: str) -> Optional[str]:
        if not self.training_data:
            return None
        similar_questions = self.search_training_data(user_input, top_k=3)
        if not similar_questions:
            return None
        best_match = similar_questions[0]
        if best_match['similarity_score'] > 0.3:
            answer = best_match['answer']
            answer = answer.replace('suggested to', 'I suggest you')
            answer = answer.replace('suggested', 'I recommend')
            answer = answer.replace('mr.', '')
            answer = answer.replace('advised to', 'I advise you to')
            if best_match.get('crop'):
                crop_info = f" (Related to {best_match['crop'].title()} farming)"
                answer += crop_info
            return answer
        return None

    def get_knowledge_base_response(self, user_input: str) -> str:
        intent = self.detect_intent(user_input)
        crop = self.extract_crop_name(user_input)
        faq_answer = self.search_faqs(user_input)
        if faq_answer:
            return faq_answer
        if intent == 'disease':
            return self.search_disease_info(crop, user_input)
        elif intent == 'irrigation':
            return self.get_irrigation_advice(crop)
        elif intent == 'fertilizer':
            return self.get_fertilizer_info(user_input)
        elif intent == 'government':
            return self.get_government_schemes()
        elif intent == 'weather':
            return self.get_weather_info()
        elif intent == 'pest':
            return self.get_pest_control_info()
        elif intent == 'market':
            return self.get_market_info()
        return self.get_default_response()

    def search_faqs(self, text: str) -> Optional[str]:
        if 'common_faqs' not in self.knowledge_base:
            return None
        faqs = self.knowledge_base['common_faqs']
        best_match_score = 0
        best_answer = None
        for faq in faqs:
            score = fuzz.partial_ratio(text.lower(), faq['question'].lower())
            if score > best_match_score:
                best_match_score = score
                best_answer = faq['answer']
        if best_match_score > 60:
            return best_answer
        return None

    def search_disease_info(self, crop: str, text: str) -> str:
        if crop and crop in self.knowledge_base.get('crop_diseases', {}):
            diseases = self.knowledge_base['crop_diseases'][crop]
            best_match_score = 0
            best_disease = None
            for disease_info in diseases:
                symptoms = disease_info['symptoms'].lower()
                score = fuzz.partial_ratio(text.lower(), symptoms)
                if score > best_match_score:
                    best_match_score = score
                    best_disease = disease_info
            if best_disease and best_match_score > 50:
                return f"Based on the symptoms, it might be {best_disease['disease']}.\n\nSymptoms: {best_disease['symptoms']}\nTreatment: {best_disease['treatment']}\nPrevention: {best_disease['prevention']}"
        return "I couldn't identify the specific disease. Please provide more details about the symptoms or consult a local agricultural expert."

    def get_irrigation_advice(self, crop: str = None) -> str:
        irrigation_info = self.knowledge_base.get('irrigation', {})
        response = "Irrigation Guidelines:\n\n"
        response += "General Tips:\n"
        for tip in irrigation_info.get('general_tips', []):
            response += f"• {tip}\n"
        if crop and crop in irrigation_info.get('crop_specific', {}):
            response += f"\nSpecific advice for {crop.title()}:\n"
            response += f"• {irrigation_info['crop_specific'][crop]}\n"
        return response

    def get_fertilizer_info(self, text: str) -> str:
        fertilizer_info = self.knowledge_base.get('fertilizers', {})
        response = "Fertilizer Information:\n\n"
        response += "Organic Fertilizers:\n"
        for fertilizer in fertilizer_info.get('organic', []):
            response += f"• {fertilizer['name']}: {fertilizer['benefits']}\n"
            response += f"  Application: {fertilizer['application']}\n\n"
        response += "Chemical Fertilizers:\n"
        for fertilizer in fertilizer_info.get('chemical', []):
            response += f"• {fertilizer['name']}: {fertilizer['use']}\n"
        return response

    def get_government_schemes(self) -> str:
        schemes = self.knowledge_base.get('government_schemes', {})
        response = "Government Schemes for Farmers:\n\n"
        response += "Central Government Schemes:\n"
        for scheme in schemes.get('central', []):
            response += f"• {scheme['name']}\n"
            if 'name_hindi' in scheme:
                response += f"  ({scheme['name_hindi']})\n"
            response += f"  Description: {scheme['description']}\n"
            response += f"  Eligibility: {scheme['eligibility']}\n"
            response += f"  Benefits: {scheme['benefits']}\n\n"
        return response

    def get_weather_info(self) -> str:
        weather_info = self.knowledge_base.get('weather_tips', {})
        response = "Weather-based Farming Tips:\n\n"
        for season, tips in weather_info.items():
            response += f"{season.title()} Season:\n"
            for tip in tips:
                response += f"• {tip}\n"
            response += "\n"
        return response

    def get_pest_control_info(self) -> str:
        pest_info = self.knowledge_base.get('pest_control', {})
        response = "Pest Control Methods:\n\n"
        response += "Organic Methods:\n"
        for method in pest_info.get('organic_methods', []):
            response += f"• {method}\n"
        response += "\nIntegrated Pest Management:\n"
        for method in pest_info.get('integrated_pest_management', []):
            response += f"• {method}\n"
        return response

    def get_market_info(self) -> str:
        market_info = self.knowledge_base.get('market_information', {})
        response = "Market Information:\n\n"
        response += "Price Information Sources:\n"
        for source in market_info.get('price_sources', []):
            response += f"• {source}\n"
        response += "\nSelling Tips:\n"
        for tip in market_info.get('selling_tips', []):
            response += f"• {tip}\n"
        return response

    def get_default_response(self) -> str:
        return """I understand you're asking about farming, but I need more specific information to help you better.

You can ask me about:
• Crop diseases (e.g., 'My tomato leaves have spots')
• Irrigation (e.g., 'How often should I water wheat?')
• Fertilizers (e.g., 'What fertilizer for rice?')
• Government schemes (e.g., 'Tell me about PM-KISAN')
• Pest control
• Weather tips
• Market prices
• Animal husbandry

Feel free to be more specific in your question!"""

    def get_response(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return "Please ask me a farming question!"
        greetings = ['hello', 'hi', 'namaste', 'hey', 'good morning', 'good evening']
        if any(greeting in user_input.lower() for greeting in greetings):
            return "Hello! Welcome to Agricultural Assistant. I'm here to help you with farming questions. What would you like to know?"
        help_keywords = ['help', 'what can you do', 'capabilities']
        if any(keyword in user_input.lower() for keyword in help_keywords):
            stats = ""
            if self.training_data and 'statistics' in self.training_data:
                total_records = self.training_data['statistics']['total_records']
                stats = f" I have access to {total_records:,} farming questions and answers from agricultural experts."
            return f"""I can help you with:
• Crop diseases and treatments
• Irrigation and watering advice
• Fertilizer recommendations
• Government schemes and subsidies
• Weather-based farming tips
• Pest control methods
• Market information and selling tips
• Animal husbandry and livestock care

{stats}

Just ask me questions like:
• "My tomato plants have brown spots"
• "Tell me about PM-KISAN scheme"
• "How to control aphids in brinjal?"
• "What fertilizer dose for rice?"
"""
        training_response = self.get_response_from_training_data(user_input)
        if training_response:
            return training_response
        return self.get_knowledge_base_response(user_input)

def main():
    print("Agricultural Chatbot - Terminal Version")
    print("=" * 50)
    knowledge_base_path = "data/knowledge_base.json"
    training_data_path = "data/processed_training_data.json"
    print("Loading agricultural knowledge...")
    try:
        chatbot = TerminalAgriChatbot(knowledge_base_path, training_data_path)
    except Exception as e:
        print(f"Error loading chatbot: {e}")
        return
    print("\nAgricultural Assistant Ready!")
    print("\nCommands:")
    print("• Ask any farming question")
    print("• Type 'help' for capabilities")
    print("• Type 'quit' or 'exit' to quit")
    print("• Press Ctrl+C to exit anytime")
    print()
    while True:
        try:
            user_input = input("Your question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("Thanks for using Agricultural Assistant! Happy farming!")
                break
            print("\nAgricultural Assistant:")
            start_time = time.time()
            response = chatbot.get_response(user_input)
            response_time = (time.time() - start_time) * 1000
            print(response)
            print(f"\nResponse time: {response_time:.1f}ms")
            print("-" * 50)
        except KeyboardInterrupt:
            print("\n\nThanks for using Agricultural Assistant! Happy farming!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different question.")
            continue

if __name__ == "__main__":
    main()
