"""
Virtual Shark Panel Feedback Engine
Generates personalized feedback from different investor personas.
"""

from typing import Dict, List
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class SharkPanel:
    """Manages a panel of virtual investor personas that provide feedback."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the shark panel.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            print("Warning: OpenAI API key not found. Using mock responses.")
        
        self.sharks = {
            'visionary': {
                'name': 'The Visionary',
                'focus': 'market potential and innovation',
                'personality': 'Enthusiastic about big ideas and market disruption. Focuses on scalability and vision.',
                'prompt_template': """You are The Visionary, an investor who gets excited about big ideas and market potential. 
You focus on innovation, scalability, and vision. You're known for asking about market size and growth potential.
Analyze this pitch and provide feedback in your unique voice. Be enthusiastic but also point out gaps in vision or market understanding."""
            },
            'finance': {
                'name': 'The Finance Shark',
                'focus': 'margins, revenue model, and financial viability',
                'personality': 'Numbers-driven, skeptical about financial projections. Cares deeply about unit economics and profitability.',
                'prompt_template': """You are The Finance Shark, a numbers-driven investor who cares about margins, revenue models, and financial viability.
You're skeptical about financial projections and always ask about unit economics, profitability, and sustainable business models.
Analyze this pitch and provide feedback in your unique voice. Be direct and focus on the financial aspects."""
            },
            'customer_advocate': {
                'name': 'The Customer Advocate',
                'focus': 'problem clarity and customer validation',
                'personality': 'Customer-obsessed, wants to see clear problem-solution fit and customer validation.',
                'prompt_template': """You are The Customer Advocate, an investor who is customer-obsessed and cares deeply about problem clarity.
You want to see clear problem-solution fit, customer validation, and evidence that people actually want this product.
Analyze this pitch and provide feedback in your unique voice. Focus on customer needs and validation."""
            },
            'skeptic': {
                'name': 'The Skeptic',
                'focus': 'challenging assumptions and identifying weaknesses',
                'personality': 'Critical thinker who challenges every assumption. Identifies weaknesses and asks tough questions.',
                'prompt_template': """You are The Skeptic, a critical investor who challenges assumptions and identifies weaknesses.
You're known for asking tough questions and pointing out flaws in the business model, market assumptions, or execution plan.
Analyze this pitch and provide feedback in your unique voice. Be critical but constructive."""
            }
        }
    
    def generate_feedback(self, voice_analysis: Dict, content_analysis: Dict) -> Dict:
        """
        Generate feedback from all sharks in the panel.
        
        Args:
            voice_analysis: Results from Pipeline 1 (voice & tone analysis)
            content_analysis: Results from Pipeline 2 (content & business analysis)
            
        Returns:
            Dictionary containing feedback from all sharks and final recommendation
        """
        # Prepare context for sharks
        context = self._prepare_context(voice_analysis, content_analysis)
        
        # Get feedback from each shark
        shark_feedbacks = {}
        for shark_id, shark_info in self.sharks.items():
            feedback = self._get_shark_feedback(shark_id, shark_info, context)
            shark_feedbacks[shark_id] = feedback
        
        # Generate final recommendation
        recommendation = self._generate_recommendation(voice_analysis, content_analysis, shark_feedbacks)
        
        return {
            'shark_feedbacks': shark_feedbacks,
            'final_recommendation': recommendation,
            'summary': self._generate_summary(voice_analysis, content_analysis, shark_feedbacks, recommendation)
        }
    
    def _prepare_context(self, voice_analysis: Dict, content_analysis: Dict) -> str:
        """Prepare context string for shark prompts."""
        delivery_score = voice_analysis['delivery_score']
        business_score = content_analysis['business_viability_score']
        transcript = content_analysis['transcript']
        
        # Voice analysis summary
        voice_summary = f"""
DELIVERY ANALYSIS:
- Delivery Score: {delivery_score}/100
- Confidence Level: {voice_analysis['emotional_tone']['confidence']:.2f}
- Nervousness Level: {voice_analysis['emotional_tone']['nervousness']:.2f}
- Enthusiasm Level: {voice_analysis['emotional_tone']['enthusiasm']:.2f}
- Hesitation Index: {voice_analysis['negative_behaviors']['hesitation_index']:.2f}
- Monotonicity: {voice_analysis['negative_behaviors']['monotonicity']:.2f}
"""
        
        # Content analysis summary
        nlp = content_analysis['nlp_evaluations']
        content_summary = f"""
BUSINESS CONTENT ANALYSIS:
- Business Viability Score: {business_score}/100
- Problem Clarity: {nlp['problem_clarity']:.2f}
- Product Differentiation: {nlp['product_differentiation']:.2f}
- Business Model Strength: {nlp['business_model_strength']:.2f}
- Market Opportunity: {nlp['market_opportunity_articulation']:.2f}
- Revenue Logic: {nlp['revenue_logic']:.2f}
- Competition Awareness: {nlp['competition_awareness']:.2f}
- Pitch Structure Completeness: {content_analysis['pitch_structure']['structure_completeness']:.2f}
"""
        
        return f"{voice_summary}\n{content_summary}\n\nPITCH TRANSCRIPT:\n{transcript}"
    
    def _get_shark_feedback(self, shark_id: str, shark_info: Dict, context: str) -> Dict:
        """Get feedback from a specific shark."""
        prompt = f"""{shark_info['prompt_template']}

{context}

Provide your feedback in 2-3 paragraphs. Be specific, constructive, and stay in character. 
End with a clear statement: "I'm IN" or "I'm OUT" or "I need more information"."""
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": shark_info['prompt_template']},
                        {"role": "user", "content": context + "\n\nProvide your feedback in 2-3 paragraphs. Be specific, constructive, and stay in character. End with a clear statement: 'I'm IN' or 'I'm OUT' or 'I need more information'."}
                    ],
                    temperature=0.8,
                    max_tokens=300
                )
                feedback_text = response.choices[0].message.content
            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                feedback_text = self._generate_mock_feedback(shark_id, context)
        else:
            feedback_text = self._generate_mock_feedback(shark_id, context)
        
        # Extract decision from feedback
        decision = self._extract_decision(feedback_text)
        
        return {
            'name': shark_info['name'],
            'focus': shark_info['focus'],
            'feedback': feedback_text,
            'decision': decision
        }
    
    def _generate_mock_feedback(self, shark_id: str, context: str) -> str:
        """Generate mock feedback when API is not available."""
        mock_feedbacks = {
            'visionary': """I love the vision here! The market potential seems massive, and I'm excited about the innovation. 
However, I need to see more clarity on how you'll scale this and capture market share. The differentiation is there, but 
I want to understand the path to becoming a market leader. I need more information before I can commit.""",
            'finance': """Let me cut to the numbers. I see you've mentioned revenue, but I need to see the unit economics. 
What's your customer acquisition cost? What's your lifetime value? Without clear financial projections and a path to profitability, 
this is too risky for me. I'm OUT.""",
            'customer_advocate': """The problem you're solving is clear, which I appreciate. However, I don't see enough 
customer validation here. Have you talked to potential customers? Do they actually want this? I need proof that there's 
real demand before I can invest. I need more information.""",
            'skeptic': """I have serious concerns. Your assumptions about the market seem optimistic, and I'm not convinced 
the problem is as big as you say. The competition is stronger than you're acknowledging, and I don't see a clear competitive 
moat. This feels like wishful thinking. I'm OUT."""
        }
        return mock_feedbacks.get(shark_id, "I need more information to make a decision.")
    
    def _extract_decision(self, feedback_text: str) -> str:
        """Extract decision from feedback text."""
        feedback_lower = feedback_text.lower()
        if "i'm in" in feedback_lower or "i am in" in feedback_lower or "invest" in feedback_lower:
            return "IN"
        elif "i'm out" in feedback_lower or "i am out" in feedback_lower or "not invest" in feedback_lower:
            return "OUT"
        else:
            return "NEED MORE INFO"
    
    def _generate_recommendation(self, voice_analysis: Dict, content_analysis: Dict, 
                                shark_feedbacks: Dict) -> Dict:
        """Generate final investment recommendation based on scores and shark decisions."""
        delivery_score = voice_analysis['delivery_score']
        business_score = content_analysis['business_viability_score']
        
        # Count shark decisions
        decisions = [shark['decision'] for shark in shark_feedbacks.values()]
        in_count = decisions.count('IN')
        out_count = decisions.count('OUT')
        need_info_count = decisions.count('NEED MORE INFO')
        
        # Calculate weighted score
        combined_score = (delivery_score * 0.3 + business_score * 0.7)
        
        # Determine recommendation
        if combined_score >= 75 and in_count >= 2:
            recommendation = "INVEST"
            confidence = "HIGH"
        elif combined_score >= 60 and in_count >= 1:
            recommendation = "INVEST"
            confidence = "MEDIUM"
        elif combined_score < 40 or out_count >= 3:
            recommendation = "NOT INVEST"
            confidence = "HIGH"
        elif need_info_count >= 2 or combined_score < 60:
            recommendation = "NEED MORE INFO"
            confidence = "MEDIUM"
        else:
            recommendation = "NEED MORE INFO"
            confidence = "MEDIUM"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'combined_score': round(combined_score, 2),
            'shark_votes': {
                'IN': in_count,
                'OUT': out_count,
                'NEED MORE INFO': need_info_count
            }
        }
    
    def _generate_summary(self, voice_analysis: Dict, content_analysis: Dict, 
                         shark_feedbacks: Dict, recommendation: Dict) -> str:
        """Generate overall summary of the pitch evaluation."""
        delivery_score = voice_analysis['delivery_score']
        business_score = content_analysis['business_viability_score']
        
        summary = f"""
=== SHARK TANK PITCH EVALUATION SUMMARY ===

DELIVERY SCORE: {delivery_score}/100
BUSINESS VIABILITY SCORE: {business_score}/100
COMBINED SCORE: {recommendation['combined_score']}/100

SHARK PANEL DECISIONS:
"""
        for shark_id, shark_feedback in shark_feedbacks.items():
            summary += f"- {shark_feedback['name']}: {shark_feedback['decision']}\n"
        
        summary += f"\nFINAL RECOMMENDATION: {recommendation['recommendation']} (Confidence: {recommendation['confidence']})"
        
        return summary

