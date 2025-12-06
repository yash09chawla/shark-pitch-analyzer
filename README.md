# Shark Tank Pitch Analyzer

A multimodal AI pipeline that analyzes pitch delivery (voice & tone) and business content quality, then generates personalized feedback from a panel of virtual "shark" investors.

## 🎯 Features

### Pipeline 1: Voice & Tone Analysis
- Extracts vocal features (pitch, pace, volume variation, pauses)
- Detects emotional tone (confidence, nervousness, enthusiasm)
- Identifies negative behaviors (filler words, hesitation, monotonicity)
- Calculates delivery score (0-100)

### Pipeline 2: Content & Business Logic Analysis
- Transcribes speech using Whisper ASR
- Evaluates business content quality:
  - Problem clarity
  - Product differentiation
  - Business model strength
  - Market opportunity articulation
  - Revenue logic
  - Competition awareness
- Detects pitch structure (Hook → Problem → Solution → Ask)
- Assigns business viability score (0-100)

### Pipeline 3: Virtual Shark Panel
- **The Visionary**: Focuses on market potential and innovation
- **The Finance Shark**: Cares about margins, revenue model, and financial viability
- **The Customer Advocate**: Emphasizes problem clarity and customer validation
- **The Skeptic**: Challenges assumptions and identifies weaknesses
- Generates personalized feedback from each shark
- Provides final recommendation: INVEST / NOT INVEST / NEED MORE INFO

## 📋 Requirements

- Python 3.8+
- OpenAI API key (optional, for enhanced shark feedback)

## 🚀 Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Ass
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up OpenAI API key:
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

Or pass it as a command-line argument (see Usage below).

## 💻 Usage

### Basic Usage
```bash
python main.py <audio_file_path>
```

### With OpenAI API Key
```bash
python main.py <audio_file_path> <openai_api_key>
```

### Examples
```bash
# Analyze a pitch audio file
python main.py pitch.mp3

# With OpenAI API key for enhanced feedback
python main.py pitch.wav YOUR_OPENAI_API_KEY
```

### Supported Audio Formats
- MP3
- WAV
- M4A
- FLAC
- Any format supported by librosa

## 📊 Output

The analyzer generates:

1. **Console Output**: Formatted analysis results with scores and feedback
2. **JSON File** (`results.json`): Complete analysis data including:
   - Voice & tone metrics
   - Content analysis scores
   - Full transcript
   - Individual shark feedback
   - Final recommendation

### Sample Output Structure
```json
{
  "voice_analysis": {
    "delivery_score": 75.5,
    "emotional_tone": {
      "confidence": 0.72,
      "nervousness": 0.31,
      "enthusiasm": 0.68
    },
    "negative_behaviors": {
      "hesitation_index": 0.25,
      "monotonicity": 0.15
    }
  },
  "content_analysis": {
    "business_viability_score": 68.3,
    "transcript": "...",
    "nlp_evaluations": {...}
  },
  "shark_feedback": {
    "shark_feedbacks": {...},
    "final_recommendation": {...}
  }
}
```

## 🏗️ Project Structure

```
Ass/
├── main.py                      # Main application entry point
├── pipeline1_voice_tone.py     # Voice & tone analysis pipeline
├── pipeline2_content_business.py # Content & business analysis pipeline
├── shark_panel.py              # Virtual shark panel feedback engine
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .env                        # Environment variables (optional)
```

## 🔧 How It Works

1. **Audio Processing**: The system loads the audio file and extracts acoustic features
2. **Voice Analysis**: Analyzes pitch, pace, volume, pauses, and emotional indicators
3. **Transcription**: Uses Whisper to convert speech to text
4. **Content Analysis**: Performs NLP-based evaluation of business content
5. **Shark Feedback**: Each virtual shark analyzes the pitch and provides personalized feedback
6. **Recommendation**: Synthesizes all inputs to generate final investment recommendation

## 🎨 Customization

### Adding New Sharks
Edit `shark_panel.py` to add new investor personas:

```python
'new_shark': {
    'name': 'The New Shark',
    'focus': 'your focus area',
    'personality': 'personality description',
    'prompt_template': 'Your custom prompt...'
}
```

### Adjusting Scoring Weights
Modify the scoring functions in:
- `pipeline1_voice_tone.py`: `_calculate_delivery_score()`
- `pipeline2_content_business.py`: `_calculate_business_score()`
- `shark_panel.py`: `_generate_recommendation()`

## 📝 Notes

- **OpenAI API**: The system works without an API key using mock responses, but real API access provides more nuanced and contextual feedback
- **Whisper Model**: Uses the "base" model by default. You can modify this in `pipeline2_content_business.py` for better accuracy (small, medium, large)
- **Performance**: First run will download Whisper models and NLTK data, which may take a few minutes

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is created for educational purposes as part of an assignment.

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- Librosa for audio analysis
- OpenAI GPT for shark feedback generation
