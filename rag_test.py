import logging
from utils.vector_store import VectorStoreManager
from utils.query_classifier import QueryClassifier
from utils.config import MEDICAL_RAG_SYSTEM_PROMPT, NO_CONTEXT_SYSTEM_PROMPT, DIRECT_MODE_SYSTEM_PROMPT, CHAT_MODEL
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from utils.dspy_utils import compiled_classifier, compiled_rag_responder, compiled_direct_responder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_vector_store():
    logging.info("Initialisation du gestionnaire de Vector Store...")
    return VectorStoreManager()

def get_query_classifier():
    logging.info("Initialisation du classification de requete...")
    return QueryClassifier()

vector_store = get_vector_store()
query_classifier = get_query_classifier()


prompt = "Bonjour j'ai la diarhee depuis 3 jours que dois je faire?"
result = query_classifier.needs_rag(prompt)
original = result["original"]
needs_rag = result["needs_rag"]
confidence = result["confidence"]
reason = result["reason"]

# dspy_result = compiled_classifier(query_classifier=prompt)
# needs_rag = dspy_result.category == "medical"
# confidence = dspy_result.confidence
# reason = dspy_result.reason


mode_str = "RAG" if needs_rag else "Direct"
logging.info("Classification de la requête: {mode_str} (confiance: {confidence:.2f}) - Raison: {reason}")

if needs_rag:
    retrieved_docs = vector_store.search(prompt, k=4, min_score=0.8)
    # print(retrieved_docs)

else:
    retrieved_docs = []

# Mode rag et documents récupérés
if needs_rag and retrieved_docs:
    logging.info(f"{len(retrieved_docs)} documents récupérés.")
    context_str = "\n\n---\n\n".join([
                    f"Source: {doc['metadata'].get('source', 'Inconnue')} (Score: {doc['score']:.4f})\nContenu: {doc['text']}"
                    for doc in retrieved_docs
                ])
    system_prompt = f"""
    {MEDICAL_RAG_SYSTEM_PROMPT}
    **CONTEXTE FOURNI :**
    {context_str}
    Utilise uniquement ce contexte pour répondre à la requête de l'utilisateur.
    """
    # response = compiled_rag_responder(context_str= context_str, confidence=confidence, reason= reason, prompt= prompt).answer

# Mode rag sans documents récupérés
elif needs_rag and not retrieved_docs:
    logging.warning("Aucun document pertinent trouvé.")
    context_str = "Aucune information pertinente trouvée dans les documents."
    system_prompt = f"""
        {NO_CONTEXT_SYSTEM_PROMPT}
        **CONTEXTE FOURNI :**
        {context_str}
    """
    # response = "Désolé, je n'ai pas trouvé d'informations pertinentes dans mes documents."


# Mode direct
else:
    logging.info("Réponse en mode direct sans contexte.")
    context_str = "Mode direct: réponse basée sur les connaissances générales du modèle."
    system_prompt = f"""
        {DIRECT_MODE_SYSTEM_PROMPT}
        **CONTEXTE FOURNI :**
        {context_str}
    """
    # response = compiled_direct_responder(query_classifier= prompt).response

print(f"\n Context utilisé :{context_str}\n")
model = init_chat_model(CHAT_MODEL)
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=original)
]
response  = model.invoke(messages)
print("Réponse du modèle:")
print(response.content)