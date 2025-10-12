from sklearn.cluster import AgglomerativeClustering
from diarizacion_pipeline import utils
from collections import defaultdict
from random import sample
import pandas as pd
import numpy as np
import json


'''
    Fase offline:
    Crear y actualizar diccionario de embeddings de parlamentarios
'''
class Creador_embeddings:
    def __init__(self, all_segments, path_mp_uids_csv: str, path_diputados_embeddings=None):
        self.all_segments = all_segments
        self.all_valid_segments = [seg for seg in self.all_segments if seg['embedding'] is not None]
        self.cluster_dict = None
        self.segments_dialogo = None
        self.diputados_embeddings = None
        self.path_mp_uids_csv = path_mp_uids_csv
        self.path_diputados_embeddings = path_diputados_embeddings


    def definir_speakers(self, n_clusters=2):

        embedding_list = [frase['embedding'] for frase in self.all_segments]
        embedding_tuple_list = [tuple(emb) for emb in embedding_list if emb is not None] # Convert numpy arrays to tuples
        embedding_uniq_tuples = list(set(embedding_tuple_list)) # Create a set of tuples
        embedding_uniq = [np.array(t) for t in embedding_uniq_tuples] # Convert tuples back to numpy arrays
        print(f"Speakers detectados en todos los chunks: {len(embedding_uniq)}")

        # --- Clustering global para speakers consistentes ---
        embeddings_matrix = np.vstack([seg["embedding"] for seg in self.all_segments if seg['embedding'] is not None])

        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings_matrix)
        labels = clustering.labels_

        # Asignar a speaker labels consistentes
        for seg, label in zip(self.all_valid_segments, labels):
            seg["speaker"] = f"Speaker_{label}"

        print(f"Speakers finales detectados: {len(set(labels))}")

        # --- Agrupar por cluster ---
        clusters = defaultdict(list)

        for seg, label in zip(self.all_valid_segments, labels):
            clusters[label].append(seg["embedding"])

        # --- Calcular embedding representativo ---
        cluster_dict = {}

        for label, emb_list in clusters.items():
            if len(emb_list) > 1000:
                emb_list = sample(emb_list, 1000)
            emb_matrix = np.vstack(emb_list)

            # mediana como representante
            centroid = np.median(emb_matrix, axis=0)

            cluster_dict[label] = {
                "embedding": centroid,
                "emb_matrix": emb_matrix,
                "count": len(emb_list),
            }
        self.cluster_dict = cluster_dict
    
    
    def get_speakers_list(self):
        '''
        Obtener speakers detectados en la transcripción
        '''
        speakers_list = [frase['speaker'] for frase in self.all_segments]
        speakers_uniq = list(set(speakers_list))
        speakers_uniq.remove("Unknown")
        return sorted(speakers_uniq, key=lambda s: int(s.split("_")[1]))
    

    def get_segments_dialogo(self):
        """
        Obtener dialogos para reconocer a los speakers
        """
        return utils.postprocesar_segmentos(self.all_valid_segments)  

    def _load_df_ids(self, path='../data/parlamentarios_con_partido.csv'):
        df_ids = pd.read_csv(path)
        df_ids = df_ids[~df_ids['nombre_completo'].isna()]
        return df_ids


    def _load_diputados_embeddings(self, path='../data/diputados_embeddings.json'):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key, value in data.items():
            value["embedding"] = np.array(value["embedding"])
            value["embeddings_matrix"] = np.array(value["embeddings_matrix"])
        return data
    

    def update_dictionary(self, speakers_dict: dict):
        '''
        Agregar uids correspondientes y matchear con los embeddings
        '''
        if self.path_diputados_embeddings is None:
            diputados_embeddings = {}
        else:
            diputados_embeddings = self._load_diputados_embeddings(self.path_diputados_embeddings)

        df_ids = self._load_df_ids(self.path_mp_uids_csv)
        parlamentarios = []
        for speaker, nombre in speakers_dict.items():
            parlamentario = df_ids[df_ids['nombre_completo'].str.startswith(nombre)]
            if len(parlamentario) == 1 :
                parlamentarios.append(parlamentario.iloc[0])
            elif len(parlamentario) == 0:
                parlamentarios.append(pd.Series({'nombre_completo':nombre, 
                                                 'mp_uid':None, 
                                                 'partido_militante_actual_nombre':None}))
            else:
                raise Exception(f"Nombre '{nombre}' coincide con {len(parlamentario)} parlamentarios en df_ids")

        for speaker, parlamentario in zip(speakers_dict.keys(), parlamentarios):
            label = int(speaker.split("_")[1])    
            mp_uid = parlamentario['mp_uid']
            if mp_uid is None:
                continue
            diputado = diputados_embeddings.get(mp_uid, None)
            if diputado is None:
                diputado = dict(
                    nombre_completo = parlamentario['nombre_completo'],
                    partido = parlamentario['partido_militante_actual_nombre'],
                    embedding = self.cluster_dict[label]["embedding"],
                    embeddings_matrix = self.cluster_dict[label]["emb_matrix"],
                    embeddings_count = self.cluster_dict[label]["count"],
                )
            else:
                emb_matrix = np.vstack([diputado['embeddings_matrix'], self.cluster_dict[label]["emb_matrix"]])
                diputado['embeddings_matrix'] = emb_matrix
                diputado['embeddings_count'] += self.cluster_dict[label]["count"]
                diputado['embedding'] = np.median(emb_matrix, axis=0)
            diputados_embeddings[int(mp_uid)] = diputado

        self.diputados_embeddings = diputados_embeddings
        return diputados_embeddings
    

    def exportar_diccionario_embeddings(self, ouput_path):
        with open(ouput_path, "w", encoding="utf-8") as f:
            json.dump(self.diputados_embeddings, f, ensure_ascii=False, indent=2, default=utils.numpy_to_python)