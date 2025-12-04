import argparse
import os
from dotenv import load_dotenv
from github import GithubException
import ai_core
import git_handler
import file_handler
import sys
from pathlib import Path
import datetime


# --- FUNZIONE AGGIORNATA per tracciare i file modificati ---
def process_folder(root_path, llm_config, schema_collection, force):
    """
    Scansiona una cartella, elabora ogni file Markdown e restituisce un riepilogo
    e la lista dei percorsi dei file effettivamente aggiornati.
    """
    if isinstance(root_path, str):
        root_path = Path(root_path)

    markdown_files = file_handler.scan_markdown_files(root_path)
    summary = {"processed": 0, "updated": 0, "skipped": 0, "errors": 0}
    updated_files_paths = []  # Lista per tracciare i file modificati

    if not markdown_files:
        print("Nessun file Markdown trovato nel percorso specificato.")
        return summary, updated_files_paths

    prompt_template, kb_content = ai_core.load_prompt_and_knowledge_base()

    for file_path in markdown_files:
        relative_path = os.path.relpath(file_path, root_path)
        print(f"[+] Elaborazione di: {relative_path}")
        summary["processed"] += 1

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                print("  -> File vuoto. Saltato.")
                summary["skipped"] += 1
                continue

            print("  -> Ricerca schemi pertinenti su Redis...")
            schema_context = ai_core.retrieve_relevant_schemas_redis(schema_collection, content)
            print("  -> Contesto recuperato.")

            print("  -> Generazione frontmatter con AI...")
            generated_yaml = ai_core.generate_frontmatter(
                llm_config=llm_config,
                prompt_template=prompt_template,
                schema_context=schema_context,
                kb_content=kb_content,
                content=content,
            )

            if not generated_yaml:
                summary["errors"] += 1
                continue

            parsed_data = ai_core.validate_and_parse_yaml(generated_yaml)

            if parsed_data:
                if file_handler.update_file_with_frontmatter(file_path, parsed_data, force):
                    summary["updated"] += 1
                    updated_files_paths.append(str(file_path))  # Aggiunge il file alla lista
                else:
                    summary["skipped"] += 1
            else:
                summary["errors"] += 1

        except Exception as e:
            print(f"  -> Errore imprevisto durante l'elaborazione del file: {e}")
            summary["errors"] += 1

    return summary, updated_files_paths


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    load_dotenv()

    parser = argparse.ArgumentParser(description="Aggiunge frontmatter AI a file Markdown in un repository GitHub.")
    parser.add_argument("--repo", type=str, required=True, help="Nome del repository GitHub (es. 'owner/repo').")
    parser.add_argument("--branch", type=str, default=None,
                        help="Il branch specifico su cui lavorare (default: branch principale del repo).")
    parser.add_argument("--folder", type=str, default=".",
                        help="La cartella specifica all'interno del repo su cui lavorare (default: root).")
    parser.add_argument("--force", action="store_true", help="Sovrascrive il frontmatter esistente.")
    args = parser.parse_args()

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise SystemExit("Errore: La variabile d'ambiente GITHUB_TOKEN non è impostata.")

    handler = git_handler.GitHandler(github_token)
    temp_dir = git_handler.setup_temp_dir()

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    branch_name = f"feat/add-ai-frontmatter-{timestamp}"
    commit_message = "feat: Aggiunge frontmatter generato da AI"
    pr_title = "Aggiunta Frontmatter AI"
    pr_body = "Questa PR è stata generata automaticamente per aggiungere metadati strutturati (frontmatter) ai file di documentazione."

    try:
        print(f"--- Avvio processo per il repository: {args.repo} ---")
        upstream_repo = handler.get_repo(args.repo)
        source_branch = args.branch if args.branch else upstream_repo.default_branch
        print(f"[+] Branch target: {source_branch}")

        try:
            upstream_repo.get_branch(source_branch)
            print(f"  -> Branch '{source_branch}' trovato.")
        except GithubException as e:
            if e.status == 404:
                raise SystemExit(
                    f"Errore: Il branch '{source_branch}' non è stato trovato nel repository '{args.repo}'.")
            elif e.status == 403:
                raise SystemExit(
                    f"Errore: Accesso negato al repository '{args.repo}'. Verifica il token GitHub e i permessi.")
            elif e.status == 401:
                raise SystemExit(f"Errore: Token GitHub non valido o scaduto.")
            else:
                raise SystemExit(f"Errore GitHub ({e.status}): {e.data.get('message', str(e))}")
        except Exception as e:
            raise SystemExit(f"Errore imprevisto durante la verifica del branch: {e}")

        is_fork = not handler.has_push_access(upstream_repo)
        fork_url = None

        if is_fork:
            print("[!] L'utente non ha permessi di scrittura. Procedura di Fork & PR.")
            forked_repo = handler.fork_repo(upstream_repo)
            fork_url = forked_repo.clone_url
        else:
            print("[+] L'utente ha permessi di scrittura. Procedura diretta.")

        handler.clone_repo(upstream_repo.clone_url, temp_dir, source_branch)
        handler.setup_and_sync_repo(temp_dir, source_branch, fork_url=fork_url)
        handler.create_branch(temp_dir, branch_name)

        processing_path = os.path.join(temp_dir, args.folder) if args.folder != "." else temp_dir

        print("\n[+] Caricamento risorse e avvio elaborazione file AI...")
        llm_config, schema_collection = ai_core.configure_ai_models()
        print(f"[+] Modello LLM selezionato: {llm_config.provider} ({llm_config.model})")
        print(f"[+] Provider embeddings: {llm_config.embedding_provider}")

        # --- CHIAMATA AGGIORNATA ---
        summary, updated_files = process_folder(
            root_path=processing_path,
            llm_config=llm_config,
            schema_collection=schema_collection,
            force=args.force
        )

        if summary['updated'] == 0:
            print("\n[!] Nessun file è stato aggiornato. Il processo termina qui.")
            return

        print("\n[+] Finalizzazione delle modifiche su Git...")
        # --- CHIAMATA AGGIORNATA ---
        commit_success = handler.commit_and_push(temp_dir, branch_name, commit_message, updated_files,
                                                 fork_url=fork_url)

        if commit_success:
            handler.create_pull_request(
                upstream_repo=upstream_repo, head_branch=branch_name,
                base_branch=source_branch, title=pr_title, body=pr_body, is_fork=is_fork
            )
        else:
            print("\n[!] ERRORE: Il commit e push sono falliti. Impossibile creare la Pull Request.")
            print(f"[!] Le modifiche sono state applicate localmente in: {temp_dir}")
            print("[!] Puoi tentare di risolvere manualmente o rieseguire il processo.")
            raise SystemExit("Processo terminato con errori durante il commit/push.")

    except SystemExit as e:
        print(f"\nERRORE CRITICO: {e}")
    except Exception as e:
        print(f"\nERRORE IMPREVISTO: {e}")
    finally:
        git_handler.cleanup_temp_dir(temp_dir)
        print("\n--- Processo GitHub completato ---")


if __name__ == "__main__":
    main()

