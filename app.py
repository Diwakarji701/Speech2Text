import whisper
import streamlit as st

# Load the Whisper model
@st.cache_resource  # Caches the model in memory for faster reuse
def load_model():
    return whisper.load_model("base")

model = load_model()

def transcribe(audio_file):
    """
    Transcribe audio using OpenAI's Whisper model.

    Args:
        audio_file (UploadedFile): Streamlit UploadedFile object.

    Returns:
        str: Transcription text.
    """
    try:
        # Save the uploaded audio file to a temporary file
        with open("temp_audio.mp3", "wb") as f:
            f.write(audio_file.getbuffer())
        
        # Transcribe the audio file
        result = model.transcribe("temp_audio.mp3")
        return result['text']
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI
st.title("OpenAI Whisper ASR Web App")
st.write("Upload an audio file to transcribe.")

audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if audio_file:
    with st.spinner("Transcribing... Please wait!"):
        transcription = transcribe(audio_file)
    st.success("Transcription complete!")
    st.text_area("Transcribed Text", transcription, height=200)
