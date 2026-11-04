# dspy_utils.py (version CORRIGÉE)
import dspy
from utils.config import CHAT_MODEL, MISTRAL_API_KEY, text_splitter, embeddings
from utils.vector_store import VectorStoreManager
from langchain.schema import Document
import logging
import numpy as np
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


lm = dspy.LM(
    model=CHAT_MODEL,
    api_key=MISTRAL_API_KEY,
    api_base="https://api.mistral.ai/v1"  # URL de l'API Mistral
)

dspy.settings.configure(lm=lm)

# Definition d'une signature
class ToumolaInfoExtractor(dspy.Signature):
    """Extrait les informations complètes sur les plantes médicinales et les tradipraticiens à partir du contexte."""

    question: str = dspy.InputField(
        desc="Question de l'utilisateur sur la médecine traditionnelle africaine"
    )
    context: str = dspy.InputField(
        desc="Contexte contenant des informations sur les plantes médicinales, les remèdes traditionnels et les tradipraticiens spécialisés."
    )
    plant_names: list[str] = dspy.OutputField(
        desc="Liste des noms de plantes mentionnées (noms scientifiques et locaux)."
    )
    diseases_treated: list[str] = dspy.OutputField(
        desc="Liste des maladies ou symptômes traités."
    )
    preparation_methods: list[str] = dspy.OutputField(
        desc="Méthodes de préparation des remèdes (infusion, décoction, cataplasme, poudre, macération, etc.)."
    )
    posologie: list[str] = dspy.OutputField(
        desc="Posologie et dosages recommandés pour chaque plante mentionnée (quantité, fréquence, durée du traitement). Format: 'Nom de plante: dosage et fréquence'."
    )
    contre_indications: list[str] = dspy.OutputField(
        desc="Contre-indications, précautions d'emploi et effets secondaires possibles pour chaque plante. Inclure les populations à risque (femmes enceintes, enfants, personnes avec certaines conditions)."
    )
    zones_culture: list[str] = dspy.OutputField(
        desc="Zones géographiques de culture ou de récolte des plantes mentionnées (régions, pays, types de climat/terrain). Format: 'Nom de plante: zones géographiques'."
    )
    tradipraticiens: str = dspy.OutputField(
        desc="""Tradipraticiens pertinents au format JSON. Extraire du contexte les tradipraticiens dont la spécialité correspond aux maladies/plantes mentionnées(le context). Format:
        [
            {
                "nom": "Nom du tradipraticien",
                "specialite": "Spécialité principale",
                "maladies_traitees": ["maladie1", "maladie2"],
                "contact": "Téléphone ou email si disponible",
                "localisation": "Ville/région",
                "experience": "Années d'expérience si mentionné"
            }
        ]
        Si aucun tradipraticien pertinent n'est trouvé, retourner une liste vide []."""
    )
    answer: str = dspy.OutputField(
        desc="Réponse complète et structurée incluant: plantes recommandées, préparation, posologie détaillée, contre-indications importantes, zones de culture, et tradipraticiens qualifiés avec leurs coordonnées"
    )
    images: list[str] = dspy.OutputField(
        desc="Les liens URL directs vers des images réelles et accessibles des plantes médicinales mentionnées. Chaque URL doit être un lien complet (https://...) pointant vers un fichier image valide (.jpg, .png, etc.) (PAS de liens Wikimedia)"
    )
    confidence: float = dspy.OutputField(
    desc="Le score de confiance de la réponse entre 0.0 et 1.0, basé sur la pertinence des documents trouvés et la qualité des informations médicales fournies"
        )
    source: str = dspy.OutputField(
        desc="Extraire et retourner le lien URL dans les métadonnées. Si plusieurs sources, retourner celle du document le plus pertinent."
    )
    page: str = dspy.OutputField(
        desc= "Indique le numero de la page"
    )


toumola_qa = dspy.Predict(ToumolaInfoExtractor)

# question = " Je me suis reveiller ce matin mais je ne me sens pas du tout a l'aise j'ai la nose, quoi faire ? "


def build_context(retrieved_results):
    """Transformer les résultats de recherche en contexte formaté avec métadonnées."""
    if not retrieved_results:
        return "Aucune information trouvée."
    
    context_parts = []
    
    for i, result in enumerate(retrieved_results, 1):
        # En-tête du document
        header = f"Document {i} (Pertinence: {result['score']:.1f}%)"
        
        # Métadonnées du document
        metadata = result.get('metadata', {})
        
        # Informations sur la source (IMPORTANT pour DSPy)
        source = metadata.get('source') or metadata.get('original_source', 'Source non disponible')
        page = metadata.get('page', 'N/A')
        page_label = metadata.get('page_label', page)
        
        header += f"\nSource: {source}"
        header += f"\nPage: {page_label}"
        
        # Métadonnées spécifiques aux plantes (si disponibles)
        if 'plant_name' in metadata:
            header += f"\nPlante: {metadata['plant_name']}"
        if 'scientific_name' in metadata:
            header += f"\nNom scientifique: {metadata['scientific_name']}"
        if 'local_names' in metadata:
            header += f"\nNoms locaux: {metadata['local_names']}"
        if 'title' in metadata and metadata['title']:
            header += f"\nTitre: {metadata['title']}"
        
        # Contenu du document
        text = result['text']
        
        # Assembler
        context_parts.append(f"{header}\n\n{text}")
    
    # Joindre tous les documents
    context = "\n\n" + "="*60 + "\n\n".join(context_parts)
    
    return context

# Initialisation du vector store
def get_vector_store():
    logging.info("Initialisation du gestionnaire de Vector Store...")
    return VectorStoreManager()


vector_store = get_vector_store()

# Recherche des 5 vecteurs les plus proches
# La méthode search() génère elle-même l'embedding de la question
# retrieved = vector_store.search(
#     query_text=question,  # ✅ Passer la question directement
#     k=5,                  # Nombre de résultats
#     min_score=0.6         # Score minimum (optionnel, entre 0 et 1)
# )

# context = build_context(retrieved)

# response = toumola_qa(
#     question = question,
#     context = context
# )

# data = {
#     "answer": response.answer,
#     "plant_names": response.plant_names,
#     "diseases_treated": response.diseases_treated,
#     "preparation_methods":response.preparation_methods,
#     "therapeutes":response.therapeutes,
#     "zones_culture":response.zones_culture,
#     "contre_indications":response.contre_indications,
#     "images":response.images
# }

# json_response = json.dumps(data, indent=2, ensure_ascii=False)
# print(json_response)
