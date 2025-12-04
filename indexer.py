import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import rdflib
from rdflib.namespace import RDFS, RDF
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
from redis_vector_store import RedisVectorStore
import torch
import logging
from logging.handlers import RotatingFileHandler
import time
from datetime import timedelta


def setup_logging(level: str = "INFO", log_file: str | None = None):
    logger = logging.getLogger()
    logger.setLevel(level.upper())
    # reset handlers se rilanci in REPL
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        file_h = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
        file_h.setFormatter(fmt)
        logger.addHandler(file_h)

    # rumore da lib esterne -> WARNING
    logging.getLogger("rdflib").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)

    return logger

def human_td(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))



def build_embedder(model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    def _emb(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    return _emb


def parse_schema_org_rdf(file_path: Path) -> dict:
    print(f"Lettura e parsing del file RDF: {file_path}...")
    file_extension = file_path.suffix.lower()
    if file_extension == '.jsonld':
        rdf_format = 'json-ld'
    elif file_extension == '.rdf':
        rdf_format = 'xml'
    else:
        raise SystemExit(f"Errore: estensione file non supportata '{file_extension}'. Usare .jsonld o .rdf.")
    print(f"Formato RDF rilevato: '{rdf_format}'")

    g = rdflib.Graph()
    g.parse(str(file_path), format=rdf_format)
    print("Parsing completato. Estrazione degli schemi...")

    schemas = {}

    query_classes = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <https://schema.org/>
    SELECT ?class ?comment
    WHERE {
      ?class a rdfs:Class .
      ?class rdfs:comment ?comment .
      FILTER(STRSTARTS(STR(?class), "https://schema.org/"))
    }
    """
    for row in g.query(query_classes):
        class_uri = str(row["class"])
        class_name = class_uri.replace("https://schema.org/", "")
        if not class_name or class_name[0].islower():
            continue
        schemas[class_name] = {"description": str(row["comment"]), "properties": {}}

    query_properties = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <https://schema.org/>
    SELECT ?prop ?comment ?domain
    WHERE {
      ?prop a rdf:Property .
      ?prop rdfs:comment ?comment .
      ?prop schema:domainIncludes ?domain .
      FILTER(STRSTARTS(STR(?prop), "https://schema.org/"))
    }
    """
    for row in g.query(query_properties):
        prop_name = str(row["prop"]).replace("https://schema.org/", "")
        domain_name = str(row["domain"]).replace("https://schema.org/", "")
        if domain_name in schemas:
            schemas[domain_name]["properties"][prop_name] = str(row["comment"])

    print(f"Estratti {len(schemas)} schemi validi dal file RDF.")
    return schemas


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Indicizzatore Schema.org → Redis (vector)")
    parser.add_argument("--kb-dir", default="knowledge_base", help="Cartella contenente il file *schemaorg*.jsonld/.rdf")
    parser.add_argument("--sbert-model", default=os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2"), help="Modello SBERT per embeddings")
    # --- Parametri Redis ---
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--redis-password", default=os.getenv("REDIS_PASSWORD"))
    parser.add_argument("--redis-ssl",dest="redis_ssl", action="store_true",default=False, help="Abilita SSL/TLS per Redis (default: False)")
    parser.add_argument("--redis-index", default=os.getenv("REDIS_INDEX", "idx:docs"))
    parser.add_argument("--redis-prefix", default=os.getenv("REDIS_PREFIX", "doc:"))
    parser.add_argument("--distance", default=os.getenv("REDIS_DISTANCE","COSINE"), choices=["COSINE","L2","IP"])
    parser.add_argument("--recreate-index", action="store_true", help="Drop & create dell'indice prima dell'indicizzazione")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"),choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Livello logging")
    parser.add_argument("--log-file", default=os.getenv("LOG_FILE"), help="File di log (opzionale)")
    args = parser.parse_args()

    # Logging
    log = setup_logging(args.log_level, args.log_file)
    log.info("Avvio indicizzazione Schema.org → Redis")
    log.debug(f"Args: {vars(args)}")

    # 1) Modello SBERT e dimensione embeddings
    t0 = time.perf_counter()
    sbert_device = "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"
    sbert = SentenceTransformer("models/sbert-mini", device=sbert_device)
    dims = int(sbert.get_sentence_embedding_dimension())
    log.info(f"SBERT model: {args.sbert_model} • device: {sbert_device} • dims: {dims}")
    log.debug(f"Caricamento SBERT in {human_td(time.perf_counter() - t0)}")

    # 2) Redis Vector Store
    log.info(f"Connessione Redis → host={args.redis_host} port={args.redis_port} ssl={args.redis_ssl}")
    t1 = time.perf_counter()
    store = RedisVectorStore(
        host=args.redis_host, port=args.redis_port, password=args.redis_password, ssl=args.redis_ssl,
        index_name=args.redis_index, key_prefix=args.redis_prefix, dims=dims, distance_metric=args.distance
    )
    store.create_index(recreate=args.recreate_index)
    log.info(f"Redis index: {args.redis_index} • prefix: {args.redis_prefix} • recreate: {args.recreate_index}")
    log.debug(f"Connessione/indice Redis pronta in {human_td(time.perf_counter() - t1)}")

    # 3) Localizza il file schema.org
    kb_path = Path(args.kb_dir)
    schema_file = None
    for ext in ('.jsonld', '.rdf'):
        f = next(kb_path.glob(f'*schemaorg*{ext}'), None)
        if f:
            schema_file = f
            break
    if not schema_file:
        raise SystemExit(f"Errore: Nessun file schema.org (*.jsonld|*.rdf) trovato in {kb_path}/")
    log.info(f"File RDF rilevato: {schema_file.name}")

    # 4) Parsing + indicizzazione
    t_parse = time.perf_counter()
    schema_data = parse_schema_org_rdf(schema_file)
    parse_secs = time.perf_counter() - t_parse
    if not schema_data:
        raise SystemExit("Errore: Nessuno schema estratto dal file RDF.")
    N = len(schema_data)
    log.info(f"Parsing completato in {human_td(parse_secs)} • Schemi totali: {N}")

    log.info("Inizio indicizzazione su Redis...")
    BATCH = int(os.getenv("BATCH", "1000"))
    batch_ids, batch_docs, batch_metas = [], [], []
    inserted = 0

    batch_no = 0
    start_all = time.perf_counter()

    def flush_batch():
        nonlocal batch_ids, batch_docs, batch_metas, inserted, batch_no
        if not batch_ids:
            return
        batch_no += 1
        t_emb = time.perf_counter()
        embs = sbert.encode(batch_docs, normalize_embeddings=True, show_progress_bar=False).astype("float32").tolist()
        emb_secs = time.perf_counter() - t_emb

        t_up = time.perf_counter()
        store.upsert(ids=batch_ids, embeddings=embs, texts=batch_docs, metadatas=batch_metas)
        up_secs = time.perf_counter() - t_up

        inserted += len(batch_ids)
        elapsed = time.perf_counter() - start_all
        remaining = (elapsed / inserted) * (N - inserted) if inserted else 0
        rate = inserted / elapsed if elapsed > 0 else 0.0

        log.info(
            f"[Batch {batch_no}] upsert={len(batch_ids)} • emb={emb_secs:.2f}s • redis={up_secs:.2f}s • tot_ins={inserted}/{N} • {rate:.1f} doc/s • ETA={human_td(remaining)}")
        batch_ids, batch_docs, batch_metas = [], [], []

    # loop
    for schema_name, schema_info in schema_data.items():
        description = schema_info.get("description", "")
        props = ", ".join(schema_info.get("properties", {}).keys())
        content = f"Schema: {schema_name}. Descrizione: {description}. Proprietà: {props}."
        doc_id = schema_name
        meta = {"schema_name": schema_name, "num_properties": len(schema_info.get("properties", {}))}
        batch_ids.append(doc_id)
        batch_docs.append(content)
        batch_metas.append(meta)
        if len(batch_ids) >= BATCH:
            flush_batch()

    # ultimo batch
    flush_batch()

    # 5) Manifest
    try:
        res = store.r.execute_command("FT.SEARCH", args.redis_index, "*", "LIMIT", 0, 0)
        total = res[0] if isinstance(res, (list, tuple)) else inserted
    except Exception:
        total = inserted

    elapsed_all = time.perf_counter() - start_all
    log.info(
        f"Indicizzazione COMPLETATA • total={total} • tempo={human_td(elapsed_all)} • avg_rate={total / elapsed_all:.1f} doc/s")

    manifest = {
        "redis_host": args.redis_host, "redis_port": args.redis_port, "redis_index": args.redis_index,
        "redis_prefix": args.redis_prefix, "distance": args.distance, "dims": dims,
        "sbert_model": args.sbert_model, "count": total, "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    man_path = Path(args.kb_dir) / "manifest.schemaorg.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Manifest scritto: {man_path}")

if __name__ == "__main__":
    main()