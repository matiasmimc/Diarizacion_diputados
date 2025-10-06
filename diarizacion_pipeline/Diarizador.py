from youtube_transcript_api import YouTubeTranscriptApi
from whisperx.diarize import DiarizationPipeline
from pydub import AudioSegment
from tqdm import tqdm
import numpy as np
import subprocess
import whisperx
import json
import os

"""
    Generar prediccion de turnos con WhisperX a partir de un audio y su transcripción
    Puede descargarse, con la Api de YouTube (que limita las solicitudes), o cargarse desde un archivo local
"""
class Diarizador:
    def __init__(self, device='cpu', language_code='es'):
        self.device = device
        self.language_code = language_code
        self. text_segments = {}
        self.audio_file = None
        self.chunk_files = []


    def cargar_transcricion(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.text_segments = json.load(f)
        return self.text_segments


    def descargar_transcripcion(self, video_id: str):
        yt_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['es'])

        # Dar formato
        text_segments = []
        for entry in yt_transcript:
            text_segments.append({
                "start": entry.start,
                "end": entry.start + entry.duration,
                "text": entry.text
            })
        self.text_segments = text_segments
        return self.text_segments
    

    def cargar_audio(self, path: str):
        self.audio_file = path

    
    def descargar_audio(self, video_id: str):
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

        subprocess.run([
            "yt-dlp", "--extract-audio", "--audio-format", "wav",
            "-o", "audio.%(ext)s", youtube_url
        ], check=True)

        self.audio_file = "audio.wav"


    def _chunking(self, chunk_length_ms=(10 * 60 * 1000)):
        '''
        Dividir en chunks de chunk_length segundos
        Por defecto de 10 minutos
        '''
        audio = AudioSegment.from_wav(self.audio_file)
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        for idx, chunk in enumerate(chunks):
            chunk_path = f"chunk_{idx}.wav"
            if not os.path.exists(chunk_path):
                chunk.export(chunk_path, format="wav")
            self.chunk_files.append(chunk_path)


    def diarizar(self, chunk_length_ms=(10 * 60 * 1000)):
        self._chunking(chunk_length_ms)
        print("Cargando WhisperX...")
        align_model, metadata = whisperx.load_align_model(language_code="en", device=self.device)
        diarize_model = DiarizationPipeline(use_auth_token="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", device=self.device)

        all_segments = []
        time_offset = 0.0

        # chunk_files = ["chunk_0.wav", "chunk_1.wav", ...]
        for chunk_path in tqdm(self.chunk_files, desc="Procesando chunks"):

            # Cargar audio del chunk
            audio_chunk = whisperx.load_audio(chunk_path)


            # Duración en segundos (ajustado a la frecuencia del audio)
            duracion_chunk = len(audio_chunk) / 16000.0  # 16 kHz


            # Alinear transcripción de ese rango (filtra text_segments)
            chunk_words_global = [
                w for w in self.text_segments
                if time_offset <= w["start"] < time_offset + duracion_chunk
            ]
            # alinear tiempo de la transcripcion con el del chunk
            chunk_words = []
            for w in chunk_words_global:
                w_chunk = w.copy()
                w_chunk["start"] -= time_offset
                w_chunk["end"]   -= time_offset
                chunk_words.append(w_chunk)

            aligned = whisperx.align(
                chunk_words, align_model, metadata,
                audio_chunk, self.device, return_char_alignments=False
            )

            # Diarizar ese chunk
            diar_df, embeddings = diarize_model(chunk_path, return_embeddings=True)

            # Combinar diarización + alineación
            diar_segments = whisperx.assign_word_speakers(diar_df, aligned)

            # Ajustar tiempos, agregar embeddings y guardar
            for seg in diar_segments["segments"]:
                speaker = seg.get("speaker", "Unknown")
                embedding = embeddings.get(speaker)
                if embedding is not None:
                    embedding = np.array(embedding)  # convertir a array si es necesario

                all_segments.append({
                    "start": seg.get("start", 0.0) + time_offset,
                    "end": seg.get("end", 0.0) + time_offset,
                    "speaker": speaker,
                    "embedding": embedding,
                    "text": seg.get("text", ""),
                })

            # Actualizar offset
            time_offset += duracion_chunk

        return all_segments
