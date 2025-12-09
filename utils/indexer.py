# indexer.py
import argparse
import logging
from typing import Optional

from config import INPUT_DIR, PDF_INPUT_DIR
from data_loader import load_and_parse_files, load_and_parse_pdf_in_directory
from vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_indexing(input_directory: str, pdf_input_directory: str):
    """Exécute le processus complet d'indexation."""
    logging.info("--- Démarrage du processus d'indexation ---")
    

    # --- Étape 2: Chargement et Parsing des Fichiers ---
    logging.info(f"Chargement et parsing des fichiers depuis: {input_directory}")
    # documents = load_and_parse_files(input_directory)
    documents = load_and_parse_pdf_in_directory(pdf_input_directory)
    # documents.extend(pdf_docs)

    if not documents:
        logging.warning("Aucun document n'a été chargé ou parsé. Vérifiez le contenu du dossier d'entrée.")
        logging.info("--- Processus d'indexation terminé (aucun document traité) ---")
        return

    # --- Étape 3: Création/Mise à jour de l'index Vectoriel ---
    logging.info("Initialisation du gestionnaire de Vector Store...")
    vector_store = VectorStoreManager() # Le constructeur ne fait que charger s'il existe

    logging.info("Construction de l'index Faiss (cela peut prendre du temps)...")
    # Cette méthode va splitter, générer les embeddings, créer l'index et sauvegarder
    vector_store.build_index(documents)

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info(f"Nombre de documents traités: {len(documents)}")
    if vector_store.index:
        logging.info(f"Nombre de chunks indexés: {vector_store.index.ntotal}")
    else:
        logging.warning("L'index final n'a pas pu être créé ou est vide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Répertoire contenant les fichiers sources (par défaut: {INPUT_DIR})"
    )

    parser.add_argument(
        "--pdf-input-dir",
        type=str,
        default=PDF_INPUT_DIR,
        help=f"Répertoire contenant les fichiers pdf (par défaut: {PDF_INPUT_DIR})"
    )

    args = parser.parse_args()

    run_indexing(input_directory=args.input_dir, pdf_input_directory=args.pdf_input_dir)