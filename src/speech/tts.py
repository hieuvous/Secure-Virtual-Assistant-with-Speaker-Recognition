from io import BytesIO

from gtts import gTTS


def text_to_speech(text: str) -> bytes:
    """Convert Vietnamese text to MP3 bytes."""
    if not text or not text.strip():
        return b""

    buffer = BytesIO()

    tts = gTTS(
        text=text,
        lang="vi",
        slow=False,
    )

    tts.write_to_fp(buffer)

    return buffer.getvalue()