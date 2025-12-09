from fastapi import FastAPI
from app.api.endpoints import router
import logging

# Configuration du logging (comme dans tes fichiers existants)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="API RAG Médicale",
    description="API pour répondre aux questions médicales avec RAG ou en mode direct.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Bienvenue à l'API RAG Médicale. Utilisez le endpoint /api/prompt pour poser des questions."}


app.include_router(router)
