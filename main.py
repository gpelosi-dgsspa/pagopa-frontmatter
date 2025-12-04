import argparse
from pathlib import Path
from dotenv import load_dotenv
import ai_core
import file_handler
import sys
import os


def main():
    """Funzione principale per orchestrare il processo di generazione del frontmatter."""
    sys.stdout.reconfigure(encoding='utf-8')
    load_dotenv()

    parser = argparse.ArgumentParser(description="Aggiunge frontmatter generato da AI ai file Markdown.")
    parser.add_argument("--path", type=str, required=True, help="Il percorso della cartella contenente i file .md")
    parser.add_argument("--dry-run", action="store_true", help="Esegue lo script senza modificare i file.")
    parser.add_argument("--force", action="store_true", help="Sovrascrive il frontmatter esistente.")

    # --- Redis Vector Store params ---
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--redis-password", default=os.getenv("REDIS_PASSWORD"))
    parser.add_argument(
        "--redis-ssl", dest="redis_ssl", action="store_true", default=False,
        help="Abilita SSL/TLS per Redis (default: False)"
    )
    parser.add_argument("--redis-index", default=os.getenv("REDIS_INDEX", "idx:docs"))
    parser.add_argument("--redis-prefix", default=os.getenv("REDIS_PREFIX", "doc:"))
    parser.add_argument("--distance", default=os.getenv("REDIS_DISTANCE", "COSINE"), choices=["COSINE", "L2", "IP"])

    # --- Sbert Model params ---
    parser.add_argument(
        "--sbert-model",
        default=os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2"),
        help="Modello SBERT per embeddings di query.",
    )

    args = parser.parse_args()

    print("--- Avvio del processo ---")
    if args.dry_run:
        print("Modalità DRY-RUN: Nessun file verrà modificato.")

    total_files = files_processed = files_updated = files_skipped = files_failed = 0
    markdown_files = []

    try:
        print("[+] Caricamento risorse e configurazione AI...")
        prompt_template, kb_content = ai_core.load_prompt_and_knowledge_base()
        llm_config, schema_collection = ai_core.configure_ai_models()
        print(f"[+] Modello LLM selezionato: {llm_config.provider} ({llm_config.model})")
        print(f"[+] Provider embeddings: {llm_config.embedding_provider}")
        #print(f"[+] Archivio ChromaDB: {ai_core.get_chroma_persist_directory()}")

        # Redis vector store per retrieval (crea l'indice se non esiste)
        store = ai_core.open_vector_store_redis(
            host=args.redis_host, port=args.redis_port, password=args.redis_password, ssl=args.redis_ssl,
            index_name=args.redis_index, key_prefix=args.redis_prefix, dims=args.dims, distance_metric=args.distance
        )

        print("[+] Risorse caricate con successo.")

        markdown_files = file_handler.scan_markdown_files(Path(args.path))
        total_files = len(markdown_files)
        print(f"[+] Trovati {total_files} file Markdown da elaborare.")

        files_processed = 0
        files_updated = 0
        files_skipped = 0
        files_failed = 0

        for i, file_path in enumerate(markdown_files):
            relative_path = os.path.relpath(file_path, args.path)
            print(f"\n--- Elaborazione di: {relative_path} ({i + 1}/{total_files}) ---")
            files_processed += 1

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content.strip():
                    print("  -> File vuoto. Saltato.")
                    files_skipped += 1
                    continue

                print("  -> Ricerca schemi pertinenti su Redis...")
                #schema_context = ai_core.retrieve_relevant_schemas(schema_collection, content)
                schema_context = ai_core.retrieve_relevant_schemas_redis(
                    store, content, k=args.k, sbert_model=args.sbert_model
                )
                print("  -> Contesto recuperato. Generazione frontmatter in corso...")

                generated_yaml_str = ai_core.generate_frontmatter(
                    llm_config, prompt_template, schema_context, kb_content, content
                )

                if not generated_yaml_str:
                    print("  -> Errore: L'AI non ha restituito un output.")
                    files_failed += 1
                    continue

                validated_frontmatter = ai_core.validate_and_parse_yaml(generated_yaml_str)

                if validated_frontmatter:
                    if not args.dry_run:
                        was_updated = file_handler.update_file_with_frontmatter(
                            file_path, validated_frontmatter, args.force
                        )
                        if was_updated:
                            print("  -> File aggiornato con successo.")
                            files_updated += 1
                        else:
                            # La funzione ora ritorna False sia per errore che per file saltato,
                            # la logica di stampa è già dentro la funzione.
                            files_skipped += 1
                    else:
                        print("  -> DRY-RUN: Frontmatter generato e valido.")
                        # print("--- INIZIO DRY-RUN OUTPUT ---")
                        # print(generated_yaml_str)
                        # print("--- FINE DRY-RUN OUTPUT ---")
                else:
                    print("  -> Errore: L'output dell'AI non è un YAML valido.")
                    files_failed += 1

            except Exception as e:
                print(f"  -> Errore imprevisto durante l'elaborazione del file: {e}")
                files_failed += 1

    except SystemExit as e:
        print(f"\nERRORE CRITICO: {e}")
    except Exception as e:
        print(f"\nERRORE IMPREVISTO: {e}")
    finally:
        print("\n--- Processo completato ---")
        print(f"File totali: {total_files}")
        print(f"File elaborati: {files_processed}")
        print(f"File aggiornati: {files_updated}")
        print(f"File saltati (o già con frontmatter): {files_skipped}")
        print(f"File falliti: {files_failed}")
        print("------------------------")


if __name__ == "__main__":
    main()

