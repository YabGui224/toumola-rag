from fastapi import APIRouter, HTTPException
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from utils.config import CHAT_MODEL, DIRECT_MODE_SYSTEM_PROMPT, MEDICAL_RAG_SYSTEM_PROMPT, NO_CONTEXT_SYSTEM_PROMPT
from utils.query_classifier import QueryClassifier
from utils.vector_store import VectorStoreManager
import json
from utils.dspy_utils import vector_store, build_context, toumola_qa

router = APIRouter(prefix="/api")

# Initialisation des services (comme dans rag_test.py)
# vector_store = VectorStoreManager()
# query_classifier = QueryClassifier()

class QuestionRequest(BaseModel):
    prompt: str

@router.post("/prompt")
async def ask_question(request: QuestionRequest):
    prompt = request.prompt
    """
    Endpoint pour poser une question au système RAG.
    Retourne la réponse générée par le modèle.
    """
    try:

        retrieved = vector_store.search(
            query_text=prompt,  
            k=5,         
            min_score=0.6
        )

        context = build_context(retrieved)

        response = toumola_qa(
            question = prompt,
            context = context
        )

        data = {
            "answer": response.answer.replace('\n', ' '),
            "confidence": response.confidence,
            "source": response.source,
            "page": response.page,
            "plant_names": response.plant_names,
            "diseases_treated": response.diseases_treated,
            "preparation_methods":response.preparation_methods,
            "tradipraticiens":response.tradipraticiens,
            "zones_culture":response.zones_culture,
            "contre_indications":response.contre_indications,
            "images":response.images
        }

        json_response = json.dumps(data, indent=2, ensure_ascii=False)
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
