"""
Module de classification des requêtes pour déterminer si une question nécessite RAG
"""

import re
import logging
from typing import Literal, Tuple, Dict
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
import dspy
from utils.config import CHAT_MODEL, model, CLASSIFICATION_SYSTEM_PROMPT

class QueryClassifier:
    """
    Classe pour classifier les requêtes et déterminer si elles nécessitent RAG
    """
    
    def __init__(self):
        """
        Initialise le classificateur de requêtes
        """
        self.llm = model
        
        # Mots-clés liés à l'usage des plantes medicinale pour se faire soigner pour determiner si la requete neccessite l'utilisation de RAG
        # Mots-clés liés à l'usage des plantes médicinales
        self.plantes_keywords = [
            # Termes généraux de traitement
            "traiter", "soigner", "guérir", "remède", "traitement", "soin", "médicament",
            "plante", "herbe", "feuille", "racine", "écorce", "fleur", "fruit",
            "traditionnel", "naturel", "médecine traditionnelle", "phytothérapie",
            "tradipraticien", "guérisseur", "herboriste",
            
            # Types de préparations
            "décoction", "infusion", "tisane", "thé", "cataplasme", "pommade", "baume",
            "poudre", "jus", "huile", "macération", "lotion", "massage", "bain",
            "inhalation", "vapeur", "compresse", "pâte", "sirop",
            
            # Maladies courantes
            "paludisme", "malaria", "palu", "fièvre", "fièvre élevée",
            "diabète", "sucre", "glycémie", "hyperglycémie",
            "hypertension", "tension", "HTA", "tension élevée", "pression artérielle",
            "diarrhée", "dysenterie", "selles liquides", "gastro",
            "toux", "rhume", "grippe", "bronchite", "asthme", "respiratoire",
            "maux de ventre", "ballonnements", "digestion", "indigestion", "constipation",
            "plaie", "blessure", "coupure", "infection", "cicatrisation", "abcès",
            "douleur", "rhumatisme", "arthrose", "arthrite", "articulaire",
            "anémie", "fatigue", "faiblesse", "épuisement",
            "mycose", "champignon", "dermatose",
            
            # Symptômes
            "symptôme", "signe", "douleur", "mal", "souffrir",
            "fièvre", "frissons", "sueurs", "température",
            "maux de tête", "céphalée", "migraine",
            "nausée", "vomissement", "vomi",
            "vertige", "étourdissement",
            "essoufflement", "difficultés respiratoires",
            "pâleur", "faible", "fatigué",
            "gonflement", "inflammation", "rougeur",
            "démangeaison", "grattage", "prurit",
            
            # Plantes médicinales courantes (noms publics)
            "neem", "nim", "papaye", "papayer", "vernonie", "foroforo",
            "moringa", "néré", "kinkéliba", "kinkeliba", "kankaliba",
            "bissap", "hibiscus", "oseille de guinée",
            "gingembre", "menthe", "citronnelle",
            "goyave", "goyavier", "manguier",
            "baobab", "aloès", "aloe vera",
            "guiera", "basilic", "poivre de guinée",
            "ricin", "karité", "caïlcédrat",
            
            # Noms locaux (Malinké, Soussou, Poular)
            "kini", "wèrè", "papayo", "papay", "boloko", "ewuro",
            "foré", "womtimi", "folere", "nebedayi", "kankalibandyi",
            "gnanamé", "tangamanè", "adja", "nanané", "nanai", "naanae",
            "soro kèlè", "sere toro", "naïdi", "gwayabi", "goyaba",
            "sira", "kira", "bokki", "aloé", "alodé", "aluwera",
            "jala", "kayo", "kahi", "netetu", "nété",
            
            # Questions typiques
            "comment", "quoi", "quel", "quelle", "pourquoi", "où", "qui",
            "utiliser", "prendre", "consommer", "préparer", "faire",
            "dose", "posologie", "quantité", "combien",
            "efficace", "marche", "fonctionne", "résultat",
            "précaution", "danger", "risque", "effet secondaire", "contre-indication",
            "grossesse", "enceinte", "allaitement", "enfant", "bébé",
            
            # Parties de plantes
            "feuille", "feuilles", "racine", "racines", "écorce", "écorces",
            "fleur", "fleurs", "fruit", "fruits", "graine", "graines",
            "tige", "tiges", "rhizome", "bulbe", "tubercule",
            "gel", "latex", "sève", "résine", "beurre", "huile",
            
            # Actions thérapeutiques
            "soulager", "calmer", "apaiser", "diminuer", "réduire",
            "arrêter", "stopper", "prévenir", "éviter",
            "renforcer", "fortifier", "booster", "stimuler",
            "purifier", "détoxifier", "nettoyer", "drainer",
            "cicatriser", "régénérer", "réparer",
            
            # Localisation guinéenne
            "conakry", "kindia", "labé", "mamou", "kankan", "boké",
            "guinée", "guinéen", "guinéenne",
            "tradipraticien", "contacte", "téléphone", "numéro",
            
            # Mots connexes
            "conseil", "recommandation", "avis", "suggestion",
            "prévention", "hygiène", "alimentation", "régime",
            "santé", "maladie", "malade", "patient", "souffrant",
            "naturel", "bio", "organique", "local", "africain",
        ]
        
        # Questions générales qui ne nécessitent pas l'utilisation de RAG mais les capacites de discusion du model LLM
        # Questions générales qui ne nécessitent pas l'utilisation de RAG
        self.general_patterns = [
            # Salutations
            r"\b(bonjour|salut|bonsoir|coucou|hey|hello|hi)\b",
            r"\b(comment (ça va|vas-tu|allez-vous|tu vas))\b",
            r"\b(ça va|ca va|sa va)\b",
            r"\b(quoi de neuf|quoi de 9)\b",
            
            # Présentations
            r"\b(qui es-tu|qui êtes-vous|c'est quoi ton nom|tu es qui)\b",
            r"\b(comment tu t'appelles|quel est ton nom)\b",
            r"\b(tu es (un|une) (bot|robot|ia|intelligence))\b",
            r"\b(présente-toi|présentation)\b",
            
            # Remerciements
            r"\b(merci|remercie|thanks|thx)\b",
            r"\b(je te remercie|je vous remercie)\b",
            r"\b(c'est gentil|sympa)\b",
            
            # Politesse
            r"\b(s'il (te|vous) plaît|stp|svp)\b",
            r"\b(excuse(z)?-moi|pardon|désolé)\b",
            r"\b(au revoir|bye|à bientôt|à plus|salut|ciao)\b",
            r"\b(bonne (journée|soirée|nuit))\b",
            
            # Questions sur le bot
            r"\b(que (peux-tu|pouvez-vous|tu peux|vous pouvez) faire)\b",
            r"\b(comment (tu fonctionnes|ça marche|utiliser))\b",
            r"\b(à quoi tu sers|quelle est ta fonction)\b",
            r"\b(tu (es|fais) quoi)\b",
            r"\b(aide(-moi)?|help)\b",
            
            # Questions générales (non médicales)
            r"\b(quelle heure|il est quelle heure)\b",
            r"\b(quel jour|quelle date)\b",
            r"\b(météo|temps qu'il fait|il fait (beau|chaud|froid))\b",
            r"\b(raconte(-moi)? (une blague|une histoire))\b",
            r"\b(tu aimes|tu préfères)\b",
            
            # Expressions courantes
            r"\b(ok|okay|d'accord|ouais|oui|non)\b",
            r"\b(je comprends|compris|pigé)\b",
            r"\b(c'est (bon|bien|cool|super))\b",
            r"\b(pas mal|bien joué|bravo)\b",
            
            # Questions existentielles (non médicales)
            r"\b(pourquoi (tu|vous) existes?)\b",
            r"\b(sens de la vie|but de l'existence)\b",
            r"\b(dieu existe|religion)\b",
            
            # Small talk
            r"\b(tu (habites|vis) où)\b",
            r"\b(tu as quel âge|âge as-tu)\b",
            r"\b(tu manges quoi|tu dors)\b",
            r"\b(tu as (des amis|une famille))\b",
            
            # Feedback
            r"\b(tu es (nul|bête|idiot|mauvais))\b",
            r"\b(tu es (bien|bon|intelligent|génial))\b",
            r"\b(j'aime (bien)? (ça|cela|ton aide))\b",
            
            # Questions vagues
            r"^(hein|quoi|hum|euh|ben)\??$",
            r"^\?+$",
            r"^test$",
        ]   
    

    def needs_rag(self, query: str) -> Dict[str, any]:
        """
        Détermine si une requête nécessite RAG
        
        Args:
            query: Requête de l'utilisateur
            
        
        """
        # Convertir la requête en minuscules pour la comparaison
        query_lower = query.lower()
        
        # 1. Vérifier les patterns de questions générales (salutations, remerciements, etc.)
        for pattern in self.general_patterns:
            if re.match(pattern, query_lower):
                return {
                "original":query,
                "needs_rag":False,
                "confidence":0.95,
                "reason": "Question générale ou salutation"
                }
        
        # 2. Vérifier la présence de mots-clés liés aux plantes médicinales
        plantes_keywords_found = [kw for kw in self.plantes_keywords if kw in query_lower]
        if plantes_keywords_found:
            keywords_str = ", ".join(plantes_keywords_found)
            return {
                "original":query,
                "needs_rag":True,
                "confidence":0.9,
                "reason": f"Contient des mots-clés liés à la sante: {keywords_str}"
            }
        # 3. Utiliser le LLM pour les cas ambigus
        if self.llm:
            return self._classify_with_llm(query)
        
        # Par défaut, utiliser RAG pour les questions longues (plus de 5 mots)
        words = query.split()
        if len(words) > 5:
            return True, 0.6, "Question complexe (plus de 5 mots)"
        
        # Par défaut, ne pas utiliser RAG
        return False, 0.5, "Aucun critère spécifique détecté"
    
    def _classify_with_llm(self, query: str) -> Dict[str, any]:
            """
            Utilise le LLM pour classifier la requête
            
            Args:
                query: Requête de l'utilisateur
                
            Returns:
                Tuple (besoin_rag, confiance, raison)
            """
            try:
                
                model = init_chat_model(CHAT_MODEL)
                messages = [
                    SystemMessage(content=CLASSIFICATION_SYSTEM_PROMPT),
                ]
                response  = model.invoke(messages)
                print(response.content)

                result = response.content
                logging.info(f"Classification LLM pour '{query}': {result}")
                
                # Analyser la réponse
                if result["category"] == "medical":
                    confidence = 0.85  # Confiance élevée dans la décision du LLM
                    # reason = result.replace("RAG - ", "").replace("RAG-", "").replace("RAG:", "").strip()
                    return {
                        "original":query,
                        "needs_rag":True,
                        "confidence":confidence,
                        "reason": result["main_intent"]
                    }
                
                elif result["category"] == "general":
                    confidence = 0.85
                    return {
                        "original":query,
                        "needs_rag":False,
                        "confidence":confidence,
                        "reason": result["main_intent"]
                    }
                else:
                    # Réponse ambiguë, utiliser RAG par défaut
                    confidence = 0.50
                    return {
                        "original":query,
                        "needs_rag":True,
                        "confidence":confidence,
                        "reason": result["main_intent"]
                    }  # "Classification ambiguë, utilisation de RAG par précaution"
                    
            except Exception as e:
                logging.error(f"Erreur lors de la classification avec LLM: {e}")
                # En cas d'erreur, utiliser RAG par défaut
                # return True, 0.5, f"Erreur de classification: {str(e)}"
                return {
                        "original":query,
                        "needs_rag":True,
                        "confidence":0.5,
                        "reason": f"Erreur de classification: {str(e)}"
                    }
    


# lm = dspy.LM(model=CHAT_MODEL, api_key=MISTRAL_API_KEY, temperature=0.9, api_base="https://api.mistral.ai")
# dspy.configure(lm=lm)

# class ClassifyWithDSPy(dspy.Signature):
#     """Classify sentiment of a given sentence using DSPy."""

#     setence: str = dspy.InputField()
#     sentiment: Literal["rag", "direct"] = dspy.OutputField()
#     confidence: float = dspy.OutputField()
#     reason: str = dspy.OutputField()

# classify = dspy.Predict(ClassifyWithDSPy)
# classify(setence="Bonjour, comment traiter le paludisme?")