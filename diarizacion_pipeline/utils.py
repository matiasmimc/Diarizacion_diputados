from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re

def extract_first_sentence(text):
    """
    Extrae la primera oración completa de un texto (terminada en . ! ?).
    Devuelve (oracion, resto).
    """
    match = re.search(r'([^.?!]*[.?!])\s*(.*)', text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    else:
        # No hay cierre → todo el texto como 'resto'
        return text.strip(), ""


def postprocesar_segmentos(all_segments):
        csv_rows = []
        for seg in all_segments:
            csv_rows.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "text": seg["text"]
            })

        df = pd.DataFrame(csv_rows)

        merged_rows = []
        current = None
        # Agrupar segmentos contiguos por el mismo speaker
        for _, row in df.iterrows():
            if current is None:
                current = row.copy()
            else:
                if current['speaker'] == row['speaker']:
                    # Mismo speaker → concatenar
                    current['end'] = row['end']
                    current['text'] += " " + row['text']
                else:
                    # Diferente speaker
                    if not re.search(r'[.!?]$', current['text'].strip()):
                        # El anterior está incompleto → añadirle primera oración del siguiente
                        primera, resto = extract_first_sentence(row['text'])
                        current['text'] += " " + primera
                        row['text'] = resto
                        if resto:  # ajustar tiempo de inicio si hay resto
                            row['start'] = row['start']
                        else:
                            # si no queda nada, saltamos este row
                            continue
                    merged_rows.append(current)
                    current = row.copy()

        # Agregar el último
        if current is not None:
            merged_rows.append(current)

        df_merged = pd.DataFrame(merged_rows)
        df_dialogo = df_merged.drop(columns=['start', 'end'])
        df_dialogo.reset_index(inplace=True, drop=True)
        segments_dialogo = df_dialogo.to_dict(orient='records')
        return segments_dialogo


def find_closest_speaker(speaker_embedding, emb_matrix, mp_uids):
    """
    Devuelve el mp_uid y la similitud del diputado más cercano al embedding dado
    """
    sims = cosine_similarity(speaker_embedding.reshape(1, -1), emb_matrix)[0]
    idx = np.argmax(sims)
    mp_uid = mp_uids[idx]
    return mp_uid, sims[idx]


def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def load_embeddings(embs_dict):
    """
    Cargar embeddings desde JSON, convertir listas a arrays
    """
    for key, value in embs_dict.items():
        value["embedding"] = np.array(value["embedding"])
        value["embeddings_matrix"] = np.array(value["embeddings_matrix"])

    return embs_dict