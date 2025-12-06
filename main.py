"""
Main application for Shark Tank Pitch Analyzer
Integrates all pipelines and provides a unified interface.
"""

import os
import sys
import json
from pathlib import Path
from pipeline1_voice_tone import VoiceToneAnalyzer
from pipeline2_content_business import ContentBusinessAnalyzer
from shark_panel import SharkPanel


def analyze_pitch(audio_path: str, openai_api_key: str = None) -> dict:
    """
    Main function to analyze a pitch from audio file.
    
    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        openai_api_key: Optional OpenAI API key for shark feedback
        
    Returns:
        Complete analysis results dictionary
    """
    print("=" * 60)
    print("SHARK TANK PITCH ANALYZER")
    print("=" * 60)
    print(f"\nAnalyzing pitch from: {audio_path}\n")
    
    # Check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Initialize analyzers
    print("Initializing analyzers...")
    voice_analyzer = VoiceToneAnalyzer()
    content_analyzer = ContentBusinessAnalyzer()
    shark_panel = SharkPanel(api_key=openai_api_key)
    
    # Pipeline 1: Voice & Tone Analysis
    print("\n[Pipeline 1] Analyzing voice and tone...")
    voice_analysis = voice_analyzer.analyze(audio_path)
    print(f"✓ Delivery Score: {voice_analysis['delivery_score']}/100")
    
    # Pipeline 2: Content & Business Analysis
    print("\n[Pipeline 2] Transcribing and analyzing content...")
    content_analysis = content_analyzer.analyze(audio_path)
    print(f"✓ Business Viability Score: {content_analysis['business_viability_score']}/100")
    print(f"✓ Transcript length: {len(content_analysis['transcript'])} characters")
    
    # Pipeline 3: Shark Panel Feedback
    print("\n[Pipeline 3] Generating shark panel feedback...")
    shark_feedback = shark_panel.generate_feedback(voice_analysis, content_analysis)
    print("✓ Feedback generated from all sharks")
    
    # Combine all results
    results = {
        'voice_analysis': voice_analysis,
        'content_analysis': content_analysis,
        'shark_feedback': shark_feedback
    }
    
    return results


def print_results(results: dict):
    """Print formatted results to console."""
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    # Voice Analysis
    va = results['voice_analysis']
    print("\n--- VOICE & TONE ANALYSIS ---")
    print(f"Delivery Score: {va['delivery_score']}/100")
    print(f"Confidence: {va['emotional_tone']['confidence']:.2f}")
    print(f"Nervousness: {va['emotional_tone']['nervousness']:.2f}")
    print(f"Enthusiasm: {va['emotional_tone']['enthusiasm']:.2f}")
    print(f"Hesitation Index: {va['negative_behaviors']['hesitation_index']:.2f}")
    print(f"Monotonicity: {va['negative_behaviors']['monotonicity']:.2f}")
    
    # Content Analysis
    ca = results['content_analysis']
    print("\n--- CONTENT & BUSINESS ANALYSIS ---")
    print(f"Business Viability Score: {ca['business_viability_score']}/100")
    nlp = ca['nlp_evaluations']
    print(f"Problem Clarity: {nlp['problem_clarity']:.2f}")
    print(f"Product Differentiation: {nlp['product_differentiation']:.2f}")
    print(f"Business Model Strength: {nlp['business_model_strength']:.2f}")
    print(f"Market Opportunity: {nlp['market_opportunity_articulation']:.2f}")
    print(f"Revenue Logic: {nlp['revenue_logic']:.2f}")
    print(f"Competition Awareness: {nlp['competition_awareness']:.2f}")
    
    # Transcript
    print("\n--- TRANSCRIPT ---")
    print(ca['transcript'][:500] + "..." if len(ca['transcript']) > 500 else ca['transcript'])
    
    # Shark Feedback
    sf = results['shark_feedback']
    print("\n--- SHARK PANEL FEEDBACK ---")
    for shark_id, shark_data in sf['shark_feedbacks'].items():
        print(f"\n{shark_data['name']} ({shark_data['focus']}):")
        print(f"Decision: {shark_data['decision']}")
        print(f"Feedback: {shark_data['feedback']}")
    
    # Final Recommendation
    rec = sf['final_recommendation']
    print("\n--- FINAL RECOMMENDATION ---")
    print(sf['summary'])


def save_results(results: dict, output_path: str = "results.json"):
    """Save results to JSON file."""
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (int, float)):
            return float(obj) if isinstance(obj, (float, type(obj))) else int(obj)
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_file_path> [openai_api_key]")
        print("\nExample:")
        print("  python main.py pitch.mp3")
        print("  python main.py pitch.wav YOUR_OPENAI_API_KEY")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Analyze pitch
        results = analyze_pitch(audio_path, api_key)
        
        # Print results
        print_results(results)
        
        # Save results
        save_results(results)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
