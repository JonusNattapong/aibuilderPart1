import os

# --- General Configuration ---
# Determine the base path (assuming this script is in the 'DatasetAudio' folder)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_PATH, 'DataOutput')
GENERATED_MEDIA_DIR = os.path.join(OUTPUT_DIR, 'generated_media', 'audio') # Store generated audio here
NUM_SAMPLES_PER_TASK = 5 # Keep low for API testing
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds

# --- Hugging Face Inference API Configuration ---
HF_API_TOKEN = os.environ.get("HF_TOKEN") # Read from environment variable

# --- Placeholder Input Data ---
# Create a directory for placeholder input audio if needed
PLACEHOLDER_AUDIO_DIR = os.path.join(BASE_PATH, 'placeholder_audio')
os.makedirs(PLACEHOLDER_AUDIO_DIR, exist_ok=True)
# Example: Create dummy audio files if they don't exist (requires pydub)
# try:
#     from pydub import AudioSegment
#     silence = AudioSegment.silent(duration=1000) # 1 second of silence
#     for i in range(NUM_SAMPLES_PER_TASK):
#         dummy_path = os.path.join(PLACEHOLDER_AUDIO_DIR, f'dummy_audio_{i}.wav')
#         if not os.path.exists(dummy_path):
#             silence.export(dummy_path, format="wav")
# except ImportError:
#     print("Warning: pydub not installed. Cannot create dummy audio files.")
#     print("Please create placeholder audio files manually in 'placeholder_audio/' if needed.")


# --- Task-Specific Configurations ---

# Text-to-Speech (TTS)
TTS_MODEL_ID = "espnet/kan-bayashi_ljspeech_vits" # Example VITS model (check API compatibility)
# TTS_MODEL_ID = "facebook/mms-tts-eng" # Example MMS model (might work better)
TTS_FILENAME = "generated_text_to_speech_api.csv"
TTS_INPUT_TEXTS = [
    "Hello, this is a test of text to speech synthesis.",
    "The quick brown fox jumps over the lazy dog.",
    "สวัสดี นี่คือการทดสอบการสังเคราะห์เสียงพูด", # May require a Thai-specific model
    "ฝนตกหนักมากวันนี้",
    "AI Builder is creating audio datasets."
][:NUM_SAMPLES_PER_TASK]

# Automatic Speech Recognition (ASR)
ASR_MODEL_ID = "openai/whisper-large-v3" # Example Whisper model
ASR_FILENAME = "generated_asr_api.csv"
ASR_INPUT_AUDIO = [os.path.join(PLACEHOLDER_AUDIO_DIR, f'dummy_audio_{i}.wav') for i in range(NUM_SAMPLES_PER_TASK)] # Use dummy/real audio paths

# Audio Classification
AUDIO_CLASSIFICATION_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593" # Example AST model
AUDIO_CLASSIFICATION_FILENAME = "generated_audio_classification_api.csv"
AUDIO_CLASSIFICATION_INPUT_AUDIO = [os.path.join(PLACEHOLDER_AUDIO_DIR, f'dummy_audio_{i}.wav') for i in range(NUM_SAMPLES_PER_TASK)] # Use dummy/real audio paths

# Voice Activity Detection (VAD) - Often requires specific libraries or models not easily exposed via generic API
VAD_MODEL_ID = "pyannote/voice-activity-detection" # Example, might need local setup with pyannote.audio
VAD_FILENAME = "generated_vad_api.csv"
VAD_INPUT_AUDIO = [os.path.join(PLACEHOLDER_AUDIO_DIR, f'dummy_audio_{i}.wav') for i in range(NUM_SAMPLES_PER_TASK)]

# Text-to-Audio (Sound Generation) - Models might be specialized or less common on free API
TEXT_TO_AUDIO_MODEL_ID = "facebook/musicgen-small" # Example MusicGen (check API task support)
TEXT_TO_AUDIO_FILENAME = "generated_text_to_audio_api.csv"
TEXT_TO_AUDIO_PROMPTS = [
    "Sound of a dog barking",
    "Ocean waves crashing on the shore",
    "A car engine starting",
    "Synthesizer playing a simple melody",
    "Rain falling on a window"
][:NUM_SAMPLES_PER_TASK]

# Audio-to-Audio - Highly model/task specific (e.g., separation, enhancement) - Placeholder
AUDIO_TO_AUDIO_MODEL_ID = "JorisCos/DCCRNet_Libri1Mix_enhsingle_16k" # Example enhancement model
AUDIO_TO_AUDIO_FILENAME = "generated_audio_to_audio_api.csv"
AUDIO_TO_AUDIO_INPUT_AUDIO = [os.path.join(PLACEHOLDER_AUDIO_DIR, f'dummy_audio_{i}.wav') for i in range(NUM_SAMPLES_PER_TASK)]


# --- Helper to ensure directories exist ---
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GENERATED_MEDIA_DIR, exist_ok=True)
    # Create subdirs for specific tasks if needed (optional)
    # os.makedirs(os.path.join(GENERATED_MEDIA_DIR, 'tts'), exist_ok=True)
    # os.makedirs(os.path.join(GENERATED_MEDIA_DIR, 'text_to_audio'), exist_ok=True)
    # os.makedirs(os.path.join(GENERATED_MEDIA_DIR, 'audio_to_audio'), exist_ok=True)

ensure_dirs()
print(f"Audio Configuration Loaded. Output Dir: {OUTPUT_DIR}, Media Dir: {GENERATED_MEDIA_DIR}")
print(f"Placeholder audio expected in: {PLACEHOLDER_AUDIO_DIR}")
if not HF_API_TOKEN:
    print("Warning: HF_TOKEN environment variable not set. Inference API calls will likely fail.")
