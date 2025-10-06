from diarizacion_pipeline import utils
from collections import defaultdict
import numpy as np

'''
    Fase online
'''
class Matcher_speakers():

    def __init__(self, all_segments, embs_dict):
        self.all_segments = all_segments
        self.embs_dict = embs_dict
        self.mp_uids = list(embs_dict.keys())
        self.emb_matrix = np.vstack([embs_dict[uid]['embedding'] for uid in self.mp_uids])
        self.speaker_map = self._unificar_embeddings()


    def _unificar_embeddings(self):
        embedding_groups = defaultdict(list)
        for seg in self.all_segments:
            emb = seg.get("embedding")
            if emb is None: 
                continue
            # Convertir el array a tupla (inmutable) para usar como clave
            emb_key = tuple(emb.tolist())
            embedding_groups[emb_key].append(seg)

        print(f"Total de embeddings únicos: {len(embedding_groups)}")

        # --- Asignar nombres consistentes a cada grupo ---
        speaker_map = {}

        for i, (emb_key, segments) in enumerate(embedding_groups.items(), start=1):
            speaker_name = f"Speaker_{i:02d}"
            for seg in segments:
                seg["speaker"] = speaker_name
            speaker_map[emb_key] = speaker_name

        print(f"{len(speaker_map)} speakers únicos renombrados.")

        return speaker_map
    

    def match(self, sim_threshold=0.75):
        """
        Matchear speakers con los guardados. Si la similitud es menor que sim_threshold, se ignora el speaker.
        """
        speaker_id_map = {}
        for emb, speaker in self.speaker_map.items():
            emb_np = np.array(emb)
            mp_uid, sim = utils.find_closest_speaker(emb_np, self.emb_matrix, self.mp_uids)
            if sim > sim_threshold:
                speaker_id_map[speaker] = mp_uid

        segments_dialogo = utils.postprocesar_segmentos(self.all_segments)

        for seg in segments_dialogo:
            mp_uid = speaker_id_map.get(seg["speaker"])
            if not mp_uid: 
                continue
            seg["speaker"] = self.embs_dict[mp_uid]["nombre_completo"]
            seg["mp_uid"] = mp_uid

        return segments_dialogo
