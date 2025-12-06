"""
Example usage script for Shark Tank Pitch Analyzer
This demonstrates how to use the analyzer programmatically.
"""

from main import analyze_pitch, print_results, save_results
import os


def example_usage():
    """Example of how to use the analyzer."""
    
    # Example 1: Basic usage with audio file
    audio_file = "path/to/your/pitch.mp3"  # Replace with your audio file path
    
    # Check if file exists (for demo purposes)
    if not os.path.exists(audio_file):
        print(f"Please provide a valid audio file path.")
        print("Example: python example_usage.py")
        return
    
    # Optional: OpenAI API key for enhanced feedback
    api_key = os.getenv('OPENAI_API_KEY')  # Or set it directly
    
    try:
        # Analyze the pitch
        results = analyze_pitch(audio_file, api_key)
        
        # Print results to console
        print_results(results)
        
        # Save results to JSON
        save_results(results, "pitch_analysis_results.json")
        
        print("\n✓ Analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    example_usage()
