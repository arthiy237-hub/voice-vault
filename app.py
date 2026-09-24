import streamlit as st
import whisper
import tempfile
import os
import re
from sentence_transformers import SentenceTransformer, util
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import io

st.title("Voice-Based Concept Understanding Analyzer")
st.write("Upload your audio and get instant feedback!")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_whisper_model()
sbert_model = load_sbert_model()

FILLER_WORDS = ["um", "uh", "like", "you know", "actually", "basically", "so", "ah", "hmm"]

def count_filler_words(text):
    text_lower = text.lower()
    counts = {}
    total = 0
    for word in FILLER_WORDS:
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            counts[word] = len(matches)
            total += len(matches)
    return counts, total

def fluency_rating(total_fillers, word_count):
    if word_count == 0:
        return "N/A"
    filler_ratio = total_fillers / word_count
    if filler_ratio < 0.02:
        return "Excellent Fluency"
    elif filler_ratio < 0.05:
        return "Good Fluency"
    elif filler_ratio < 0.10:
        return "Moderate Fluency"
    else:
        return "Needs Improvement"

def generate_pdf(transcribed_text, reference_text, similarity_percent,
                  understanding_feedback, word_count, total_fillers,
                  filler_counts, fluency_feedback):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - inch

    def write_line(text, size=11, bold=False, gap=18):
        nonlocal y
        if y < inch:
            c.showPage()
            y = height - inch
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)
        c.drawString(inch, y, text)
        y -= gap

    write_line("Voice-Based Concept Understanding Analyzer - Report", size=16, bold=True, gap=30)

    write_line("Reference Concept:", bold=True)
    for line in [reference_text[i:i+90] for i in range(0, len(reference_text), 90)]:
        write_line(line)
    y -= 10

    write_line("Transcribed Text:", bold=True)
    for line in [transcribed_text[i:i+90] for i in range(0, len(transcribed_text), 90)]:
        write_line(line)
    y -= 10

    write_line(f"Semantic Similarity Score: {similarity_percent}%", bold=True)
    write_line(f"Understanding Feedback: {understanding_feedback}")
    y -= 10

    write_line(f"Total Words: {word_count}", bold=True)
    write_line(f"Filler Words Count: {total_fillers}")
    write_line(f"Fluency Rating: {fluency_feedback}")
    y -= 10

    if filler_counts:
        write_line("Filler Word Breakdown:", bold=True)
        for word, count in filler_counts.items():
            write_line(f"  - {word}: {count}")

    c.save()
    buffer.seek(0)
    return buffer

reference_text = st.text_area(
    "Enter the reference concept explanation (correct answer)",
    "Machine Learning is a field of AI where computers learn patterns from data without being explicitly programmed."
)

audio_file = st.file_uploader("Upload your audio explanation", type=["mp3", "wav", "m4a", "ogg"])

if audio_file is not None:
    st.audio(audio_file)

    if st.button("Analyze Audio"):
        with st.spinner("Transcribing... please wait"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name

            result = model.transcribe(tmp_path)
            transcribed_text = result["text"]
            os.remove(tmp_path)

        st.subheader("Transcribed Text:")
        st.write(transcribed_text)

        with st.spinner("Calculating semantic similarity..."):
            embedding1 = sbert_model.encode(transcribed_text, convert_to_tensor=True)
            embedding2 = sbert_model.encode(reference_text, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
            similarity_percent = round(similarity_score * 100, 2)

        st.subheader("Semantic Similarity Score:")
        st.metric(label="Similarity", value=f"{similarity_percent}%")

        if similarity_percent >= 75:
            understanding_feedback = "Strong Understanding"
        elif similarity_percent >= 50:
            understanding_feedback = "Moderate Understanding"
        else:
            understanding_feedback = "Poor Understanding"

        st.subheader("Understanding Feedback:")
        st.write(understanding_feedback)

        word_count = len(transcribed_text.split())
        filler_counts, total_fillers = count_filler_words(transcribed_text)
        fluency_feedback = fluency_rating(total_fillers, word_count)

        st.subheader("Fluency Analysis:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Words", word_count)
        col2.metric("Filler Words", total_fillers)
        col3.metric("Fluency Rating", fluency_feedback)

        if filler_counts:
            st.write("**Filler word breakdown:**")
            st.json(filler_counts)
        else:
            st.write("No filler words detected! 🎉")

        pdf_buffer = generate_pdf(
            transcribed_text, reference_text, similarity_percent,
            understanding_feedback, word_count, total_fillers,
            filler_counts, fluency_feedback
        )

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name="concept_analysis_report.pdf",
            mime="application/pdf"
        )
