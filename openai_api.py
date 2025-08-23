import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import librosa

app = FastAPI()

stt_handler = {"handler": None}
tts_handler = {"handler": None}

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(file: UploadFile = File(...)):
    if stt_handler["handler"] is None:
        return JSONResponse(status_code=503, content={"detail": "STT handler not initialized"})

    audio_bytes = await file.read()
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

    transcription, _ = next(stt_handler["handler"].process(audio))

    return JSONResponse(content={"text": transcription})

from pydantic import BaseModel
import soundfile as sf

class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str

@app.post("/v1/audio/speech")
async def speech(request: TTSRequest):
    if tts_handler["handler"] is None:
        return JSONResponse(status_code=503, content={"detail": "TTS handler not initialized"})

    audio_chunks = []
    for chunk in tts_handler["handler"].process(request.input):
        audio_chunks.append(chunk)

    if not audio_chunks:
        return JSONResponse(status_code=500, content={"detail": "TTS failed to generate audio"})

    audio_array = np.concatenate(audio_chunks)

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, 16000, format='wav')
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="audio/wav")
