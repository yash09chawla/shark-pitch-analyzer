"""
Pipeline 2: Content & Business Logic Analysis
Transcribes speech and evaluates business content quality.
"""

import whisper
import re
from typing import Dict, List
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import textstat

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class ContentBusinessAnalyzer:
    """Analyzes business content quality from transcribed speech."""
    
    def __init__(self):
        self.model = None  # Lazy load Whisper model
        self.stop_words = set(stopwords.words('english'))
        
    def _load_model(self):
        """Lazy load Whisper model to save memory."""
        if self.model is None:
            self.model = whisper.load_model("base")
    
    def analyze(self, audio_path: str) -> Dict:
        """
        Main analysis function that transcribes and evaluates business content.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing transcript and business analysis
        """
        # Transcribe audio
        self._load_model()
        result = self.model.transcribe(audio_path)
        transcript = result["text"]
        
        # Perform NLP evaluations
        nlp_evaluations = self._perform_nlp_evaluations(transcript)
        pitch_structure = self._detect_pitch_structure(transcript)
        business_score = self._calculate_business_score(nlp_evaluations, pitch_structure)
        
        return {
            'transcript': transcript,
            'nlp_evaluations': nlp_evaluations,
            'pitch_structure': pitch_structure,
            'business_viability_score': business_score
        }
    
    def _perform_nlp_evaluations(self, transcript: str) -> Dict:
        """Perform NLP evaluations on the transcript."""
        transcript_lower = transcript.lower()
        words = word_tokenize(transcript_lower)
        sentences = sent_tokenize(transcript)
        
        # Problem clarity
        problem_keywords = ['problem', 'issue', 'pain', 'challenge', 'struggle', 'difficulty', 'need', 'gap']
        problem_mentions = sum(1 for word in words if word in problem_keywords)
        problem_clarity = min(problem_mentions / 3.0, 1.0)  # Normalize to 0-1
        
        # Product differentiation
        differentiation_keywords = ['unique', 'different', 'innovative', 'revolutionary', 'first', 'only', 
                                   'better', 'improved', 'advantage', 'competitive edge']
        differentiation_mentions = sum(1 for word in words if word in differentiation_keywords)
        differentiation_score = min(differentiation_mentions / 4.0, 1.0)
        
        # Business model strength
        business_keywords = ['revenue', 'profit', 'margin', 'pricing', 'cost', 'model', 'monetize', 
                            'subscription', 'fee', 'price', 'sell', 'market']
        business_mentions = sum(1 for word in words if word in business_keywords)
        business_model_score = min(business_mentions / 5.0, 1.0)
        
        # Market opportunity
        market_keywords = ['market', 'opportunity', 'size', 'target', 'customer', 'user', 'demand', 
                          'growth', 'potential', 'billion', 'million', 'industry']
        market_mentions = sum(1 for word in words if word in market_keywords)
        market_opportunity_score = min(market_mentions / 5.0, 1.0)
        
        # Revenue logic
        revenue_keywords = ['revenue', 'sales', 'income', 'earn', 'generate', 'monetize', 'pricing', 
                           'subscription', 'recurring', 'revenue stream']
        revenue_mentions = sum(1 for word in words if word in revenue_keywords)
        revenue_logic_score = min(revenue_mentions / 3.0, 1.0)
        
        # Competition awareness
        competition_keywords = ['competitor', 'competition', 'competitive', 'vs', 'versus', 'alternative', 
                               'compare', 'differentiate', 'advantage', 'better than']
        competition_mentions = sum(1 for word in words if word in competition_keywords)
        competition_score = min(competition_mentions / 2.0, 1.0)
        
        # Additional metrics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        readability = textstat.flesch_reading_ease(transcript) / 100.0  # Normalize to 0-1
        
        return {
            'problem_clarity': float(problem_clarity),
            'product_differentiation': float(differentiation_score),
            'business_model_strength': float(business_model_score),
            'market_opportunity_articulation': float(market_opportunity_score),
            'revenue_logic': float(revenue_logic_score),
            'competition_awareness': float(competition_score),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': float(avg_sentence_length),
            'readability': float(readability)
        }
    
    def _detect_pitch_structure(self, transcript: str) -> Dict:
        """Detect if pitch follows standard structure: Hook -> Problem -> Solution -> Ask."""
        transcript_lower = transcript.lower()
        
        # Hook detection (first 20% of transcript)
        hook_portion = transcript_lower[:len(transcript_lower)//5]
        hook_keywords = ['imagine', 'what if', 'picture', 'introduce', 'present', 'excited']
        has_hook = any(keyword in hook_portion for keyword in hook_keywords)
        
        # Problem detection
        problem_keywords = ['problem', 'issue', 'pain point', 'challenge', 'struggle']
        has_problem = any(keyword in transcript_lower for keyword in problem_keywords)
        
        # Solution detection
        solution_keywords = ['solution', 'solve', 'address', 'fix', 'offer', 'provide', 'product', 'service']
        has_solution = any(keyword in transcript_lower for keyword in solution_keywords)
        
        # Ask detection (last 30% of transcript)
        ask_portion = transcript_lower[-len(transcript_lower)//3:]
        ask_keywords = ['invest', 'investment', 'funding', 'raise', 'capital', 'money', 'equity', 'ask']
        has_ask = any(keyword in ask_portion for keyword in ask_keywords)
        
        structure_score = sum([has_hook, has_problem, has_solution, has_ask]) / 4.0
        
        return {
            'has_hook': has_hook,
            'has_problem': has_problem,
            'has_solution': has_solution,
            'has_ask': has_ask,
            'structure_completeness': float(structure_score)
        }
    
    def _calculate_business_score(self, nlp_evaluations: Dict, pitch_structure: Dict) -> float:
        """Calculate overall business viability score (0-100)."""
        # Weighted average of all metrics
        scores = [
            nlp_evaluations['problem_clarity'] * 0.15,
            nlp_evaluations['product_differentiation'] * 0.15,
            nlp_evaluations['business_model_strength'] * 0.20,
            nlp_evaluations['market_opportunity_articulation'] * 0.15,
            nlp_evaluations['revenue_logic'] * 0.15,
            nlp_evaluations['competition_awareness'] * 0.10,
            pitch_structure['structure_completeness'] * 0.10
        ]
        
        business_score = sum(scores) * 100
        
        return round(max(0, min(100, business_score)), 2)
