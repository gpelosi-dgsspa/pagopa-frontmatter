import os
import yaml
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import google.generativeai as genai
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from redis_vector_store import RedisVectorStore

from abc import ABC, abstractmethod
from openai import OpenAI


# --- Funzioni di Caricamento Risorse ---

def load_prompt_and_knowledge_base() -> tuple[str, str]:
    """
    Carica il prompt master e la knowledge base testuale (escludendo schema.org).
    """
    try:
        script_dir = Path(__file__).resolve().parent
        prompt_path = script_dir / "config" / "master_prompt.txt"

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        knowledge_base_content = ""
        kb_path = script_dir / "knowledge_base"
        if kb_path.is_dir():
            for file_path in kb_path.glob('*.*'):
                if 'schemaorg' not in file_path.name:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        knowledge_base_content += f.read() + "\n\n"

        return prompt_template, knowledge_base_content.strip()
    except FileNotFoundError as e:
        raise SystemExit(
            f"Errore: File di configurazione non trovato. Controlla che il percorso sia corretto.\nDettagli: {e}")


# --- Funzioni di Configurazione AI e Vector Store ---

@dataclass
class LLMConfig:
    provider: str
    client: Any
    model: str
    embedding_provider: str



def resolve_embedding_provider() -> str:
    """Determina il provider degli embeddings basandosi sulla configurazione."""
    override = os.getenv("EMBEDDING_PROVIDER")
    if override:
        return override.strip().lower()

    llm_provider = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    if llm_provider == "openai":
        return "openai"
    if llm_provider == "openrouter":
        return "sentence-transformers"
    return "google"


def configure_embedding_function(provider: str | None = None):
    provider_name = (provider or resolve_embedding_provider()).strip().lower()

    if provider_name in {"google", "gemini"}:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Errore: GEMINI_API_KEY è obbligatoria.")
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")

        def google_embed(texts: list[str]) -> list[list[float]]:
            model = genai.embedder(model_name)
            return [model.embed_content(content=t)["embedding"] for t in texts]

        return google_embed

    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Errore: OPENAI_API_KEY è obbligatoria.")
        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

        def openai_embed(texts: list[str]) -> list[list[float]]:
            resp = client.embeddings.create(model=model_name, input=texts)
            return [d.embedding for d in resp.data]

        return openai_embed

    if provider_name in {"sentence-transformers", "local"}:
        model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)

        def st_embed(texts: list[str]) -> list[list[float]]:
            return model.encode(texts).tolist()

        return st_embed

    raise SystemExit(f"Provider embedding '{provider_name}' non supportato.")


def configure_ai_models() -> tuple[LLMConfig, RedisVectorStore]:
    provider = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    embedding_override = os.getenv("EMBEDDING_PROVIDER")
    model_name: str

    # --- Config LLM (come prima) ---
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Errore: La chiave API 'GEMINI_API_KEY' non è stata trovata.")
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        llm_client = genai.GenerativeModel(model_name)
        default_embedding = "google"

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Errore: La chiave API 'OPENAI_API_KEY' non è stata trovata.")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm_client = OpenAI(api_key=api_key)
        default_embedding = "openai"

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("Errore: La chiave API 'OPENROUTER_API_KEY' non è stata trovata.")
        model_name = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
        llm_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        default_embedding = "sentence-transformers"

    elif provider == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("Errore: La chiave API 'ANTHROPIC_API_KEY' non è stata trovata.")
        model_name = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")
        llm_client = Anthropic(api_key=api_key)
        default_embedding = "google"

    else:
        raise SystemExit(f"Provider LLM '{provider}' non supportato.")

    embedding_provider = (embedding_override or default_embedding).strip().lower()
    embedding_function = configure_embedding_function(embedding_provider)

    # --- Usa RedisVectorStore ---
    redis_store = RedisVectorStore(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD"),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        index_name=os.getenv("REDIS_INDEX_NAME", "schema_embeddings"),
        key_prefix=os.getenv("REDIS_KEY_PREFIX", "doc:"),
        dims=int(os.getenv("EMBEDDING_DIM", 384)),  # es. MiniLM = 384
        distance_metric=os.getenv("REDIS_DISTANCE", "COSINE"),
    )

    # Crea indice se non esiste
    redis_store.create_index(recreate=False)

    llm_config = LLMConfig(
        provider=provider,
        client=llm_client,
        model=model_name,
        embedding_provider=embedding_provider,
    )

    return llm_config, redis_store


# --- Funzione di Ricerca Vettoriale ---
def retrieve_relevant_schemas(collection: RedisVectorStore, query_text: str) -> str:
    """Esegue una ricerca vettoriale su Redis Vector per ottenere gli schemi più pertinenti."""
    if not query_text:
        return "Nessun contenuto da analizzare."

    try:
        results = collection.query(query_texts=[query_text], n_results=3)
        documents = results.get("documents", [])
        if documents and documents[0]:
            return "\n".join(documents[0])
        return "Nessuno schema pertinente trovato."
    except Exception as e:
        print(f"  -> Errore durante la ricerca su ChromaDB: {e}")
        return "Errore durante il recupero degli schemi."

def open_vector_store_redis(
    host: str = os.getenv("REDIS_HOST", "localhost"),
    port: int = int(os.getenv("REDIS_PORT", "6379")),
    password: str | None = os.getenv("REDIS_PASSWORD"),
    ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true",
    index_name: str = os.getenv("REDIS_INDEX", "idx:docs"),
    key_prefix: str = os.getenv("REDIS_PREFIX", "doc:"),
    dims: int = int(os.getenv("EMBEDDING_DIMS", "1536")),
    distance_metric: str = os.getenv("REDIS_DISTANCE", "COSINE"),  # COSINE|L2|IP
):
    """
    Apre (o crea) l'indice vettoriale su Redis.
    NB: la creazione dell'indice è idempotente; chiamala all'avvio app.
    """
    store = RedisVectorStore(
        host=host, port=port, password=password, ssl=ssl,
        index_name=index_name, key_prefix=key_prefix, dims=dims,
        distance_metric=distance_metric,
    )
    store.create_index(recreate=False)
    return store


def retrieve_relevant_schemas_redis(
    store: RedisVectorStore,
    query_text: str,
    k: int = 3,
    sbert_model: str = "all-MiniLM-L6-v2",
) -> str:
    if not query_text:
        return "Nessun contenuto da analizzare."
    try:
        embedding_function = configure_embedding_function("sentence-transformers")
        q_emb = embedding_function([query_text])[0]
        hits = store.query(q_emb, k=k, return_fields=["id", "text", "meta", "score"])
        if not hits:
            return "Nessuno schema pertinente trovato."
        # Concatena i campi testo restituiti
        pieces = [(h.get("text") or "").decode("utf-8", "ignore") if isinstance(h.get("text"), (bytes, bytearray)) else (h.get("text") or "") for h in hits]
        return "".join(pieces) if pieces else "Nessuno schema pertinente trovato."
    except Exception as e:
        print(f"  -> Errore durante la ricerca su Redis: {e}")
        return "Errore durante il recupero degli schemi."


# --- Funzione di Generazione ---
def generate_frontmatter(
        llm_config: LLMConfig,
        prompt_template: str,
        schema_context: str,
        kb_content: str,
        content: str,
) -> str | None:
    """Genera il frontmatter usando il provider LLM selezionato."""
    final_prompt = prompt_template.replace("{{KNOWLEDGE_BASE_CONTENT}}", kb_content)
    final_prompt = final_prompt.replace("{{SCHEMA_DEFINITIONS}}", schema_context)
    final_prompt = final_prompt.replace("{{MARKDOWN_CONTENT}}", content)

    try:
        if llm_config.provider == "gemini":
            generation_config = genai.types.GenerationConfig(response_mime_type="text/plain")
            response = llm_config.client.generate_content(final_prompt, generation_config=generation_config)
            raw_output = response.text

        elif llm_config.provider in {"openai", "openrouter"}:
            response = llm_config.client.chat.completions.create(
                model=llm_config.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "Sei un assistente che produce frontmatter YAML valido."},
                    {"role": "user", "content": final_prompt},
                ],
            )
            raw_output = response.choices[0].message.content

        elif llm_config.provider == "claude":
            response = llm_config.client.messages.create(
                model=llm_config.model,
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": final_prompt}],
            )
            raw_output = "".join(block.text for block in response.content if getattr(block, "type", "text") == "text")

        else:
            raise ValueError(f"Provider LLM non gestito: {llm_config.provider}")

        if not raw_output:
            return None

        cleaned_response = raw_output.strip().removeprefix("```yaml").removeprefix("```").removesuffix("```").strip()
        return cleaned_response
    except Exception as e:
        print(f"  -> Errore durante la chiamata all'API AI ({llm_config.provider}): {e}")
        return None


# --- Funzione di Validazione ---
def validate_and_parse_yaml(yaml_string: str) -> dict | None:
    """Tenta di fare il parsing di una stringa YAML e la restituisce come dizionario."""
    try:
        data = yaml.safe_load(yaml_string)
        if isinstance(data, dict):
            return data
        else:
            # print("--- INIZIO OUTPUT AI NON VALIDO (Tipo non dizionario) ---")
            # print(yaml_string)
            # print("--- FINE OUTPUT AI NON VALIDO ---")
            return None
    except yaml.YAMLError as e:
        # print("--- INIZIO OUTPUT AI NON VALIDO (Errore di parsing) ---")
        # print(yaml_string)
        # print(f"Dettagli errore parser YAML: {e}")
        # print("--- FINE OUTPUT AI NON VALIDO ---")
        return None

