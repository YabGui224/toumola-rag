import logging
from typing import Optional, List
from langchain_community.document_loaders import PyPDFDirectoryLoader, CSVLoader
from langchain_core.documents import Document
from typing import Dict
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_parse_csv(file_path: str) -> Optional[List[Document]]:
    """ Charge et analyse un fichier CSV donné."""
    try:
        loader = CSVLoader(
            file_path=file_path,
            encoding="utf-8"
        )
        return loader.load()

    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier CSV {file_path}: {e}")
        return None
    


def load_and_parse_pdf_in_directory(directory_path: str) -> List[Document] :
    """ Charge les fichiers PDF  """ 
    #verifier si c'est un dossier 
    input_dir = Path(directory_path)
    if not input_dir.is_dir():
        logging.error(f"Le répertoire d'entrée '{input_dir}' n'existe pas.")

    logging.info(f"Traitements des fichiers du dossier: '{input_dir}' en cours...")
    loader = PyPDFDirectoryLoader(input_dir)
    return loader.load()


def load_and_parse_files(input_dir: str) -> List[Document]:
    """
    Charge et parse récursivement les fichiers d'un répertoire.
    Retourne une liste de dictionnaires, chacun représentant un document.
    """
    documents: List[Document] = []
    input_path = Path(input_dir)

    if not input_path.is_dir():
        logging.error(f"Le répertoire d'entrée '{input_dir}' n'existe pas.")
        return []
        
    logging.info(f"Parcours du répertoire source: {input_dir}")
    for file_path in input_path.rglob("*.*"): # Parcourt tous les fichiers récursivement
        if file_path.is_file():
            relative_path = file_path.relative_to(input_path)
            # Le nom du dossier source est le premier composant du chemin relatif
            source_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
            ext = file_path.suffix.lower()
            doc_list = None

            logging.debug(f"Traitement du fichier: {relative_path} (Dossier source: {source_folder})")

            if ext == ".csv":
                doc_list = load_and_parse_csv(str(file_path))
            
            else:
                logging.warning(f"Type de fichier non supporté ignoré: {relative_path}")
                continue

            if doc_list:
                documents.append(doc_list)
            else:
                 logging.warning(f"Aucun texte n'a pu être extrait de {relative_path}")

    logging.info(f"{len(documents)} documents chargés et parsés.")
    return documents


# print(load_and_parse_pdf_in_directory('D:/ai_projects/toumola/data/pdf')[50].page_content)