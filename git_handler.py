import os
import re
import shutil
import tempfile
from github import Github
import subprocess
import stat


def validate_branch_name(branch_name: str) -> str:
    """Valida il nome del branch per prevenire command injection."""
    if not branch_name:
        raise ValueError("Il nome del branch non può essere vuoto")

    # Permette solo caratteri alfanumerici, -, _, /
    if not re.match(r'^[a-zA-Z0-9/_-]+$', branch_name):
        raise ValueError(f"Nome del branch non valido: '{branch_name}'. Usa solo lettere, numeri, -, _, /")

    # Limita la lunghezza
    if len(branch_name) > 255:
        raise ValueError(f"Nome del branch troppo lungo: {len(branch_name)} caratteri (max 255)")

    return branch_name


def validate_git_url(url: str) -> str:
    """Valida l'URL del repository Git per prevenire command injection."""
    if not url:
        raise ValueError("L'URL del repository non può essere vuoto")

    # Permette solo URL HTTPS o git@
    if not (url.startswith('https://') or url.startswith('git@')):
        raise ValueError(f"URL del repository non valido: '{url}'. Usa HTTPS o SSH")

    return url


class GitHandler:
    def __init__(self, token):
        if not token:
            raise ValueError("È richiesto un token GitHub.")
        self.g = Github(token)
        self.user = self.g.get_user()

    def get_repo(self, repo_name):
        try:
            return self.g.get_repo(repo_name)
        except Exception:
            raise SystemExit(f"Repository '{repo_name}' non trovato o accessibile.")

    def has_push_access(self, repo):
        return repo.permissions.push

    def fork_repo(self, upstream_repo):
        from github import GithubException

        print(f"  -> Creazione del fork di '{upstream_repo.full_name}'...")
        try:
            return self.user.create_fork(upstream_repo)
        except GithubException as e:
            # 422 Unprocessable Entity = fork già esiste
            if e.status == 422:
                print("  -> Fork già esistente. Utilizzo quello.")
                return self.g.get_repo(f"{self.user.login}/{upstream_repo.name}")
            # Altri errori GitHub
            print(f"  -> Errore GitHub durante la creazione del fork: {e.status} - {e.data.get('message', str(e))}")
            raise e
        except Exception as e:
            print(f"  -> Errore imprevisto durante la creazione del fork: {e}")
            raise e

    def clone_repo(self, repo_url, path, branch):
        print(f"  -> Clonazione del branch '{branch}' da {repo_url}...")
        # Validazione input
        validate_git_url(repo_url)
        validate_branch_name(branch)

        try:
            subprocess.run(
                ["git", "clone", "--branch", branch, repo_url, path],
                check=True, capture_output=True, timeout=300  # 5 minuti timeout
            )
        except subprocess.TimeoutExpired:
            raise SystemExit(f"Timeout durante la clonazione del repository (superati 5 minuti).")
        except subprocess.CalledProcessError as e:
            print(f"Errore standard:\n{e.stderr.decode('utf-8', errors='ignore')}")
            raise SystemExit("Impossibile clonare il repository.")

    def setup_and_sync_repo(self, repo_path, base_branch, fork_url=None):
        print(f"  -> Sincronizzazione forzata del branch di base '{base_branch}' con 'origin'...")
        # Validazione input
        validate_branch_name(base_branch)
        if fork_url:
            validate_git_url(fork_url)

        try:
            if fork_url:
                print(f"  -> Configurazione del remote 'fork' per il push: {fork_url}")
                subprocess.run(
                    ["git", "remote", "add", "fork", fork_url],
                    cwd=repo_path, check=True, capture_output=True, timeout=30
                )

            print("  -> Fetch da 'origin'...")
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=repo_path, check=True, capture_output=True, timeout=180
            )

            print(f"  -> Checkout del branch di base '{base_branch}'...")
            subprocess.run(
                ["git", "checkout", base_branch],
                cwd=repo_path, check=True, capture_output=True, timeout=30
            )

            print(f"  -> Reset forzato di '{base_branch}' a 'origin/{base_branch}'...")
            subprocess.run(
                ["git", "reset", "--hard", f"origin/{base_branch}"],
                cwd=repo_path, check=True, capture_output=True, timeout=30
            )
            print("  -> Sincronizzazione completata. La base è ora pulita.")

        except subprocess.TimeoutExpired as e:
            print("\n--- TIMEOUT DURANTE LA SINCRONIZZAZIONE ---")
            print(f"Comando: {' '.join(e.cmd)}")
            print("---------------------------------------")
            raise SystemExit("Timeout durante la sincronizzazione del repository locale.")
        except subprocess.CalledProcessError as e:
            print("\n--- ERRORE DURANTE LA SINCRONIZZAZIONE ---")
            print(f"Comando fallito: {' '.join(e.cmd)}")
            print(f"Errore standard:\n{e.stderr.decode('utf-8', errors='ignore')}")
            print("---------------------------------------")
            raise SystemExit("Impossibile sincronizzare il repository locale.")

    def create_branch(self, repo_path, branch_name):
        print(f"  -> Creazione del branch di lavoro: '{branch_name}'")
        # Validazione input
        validate_branch_name(branch_name)

        try:
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=repo_path, check=True, capture_output=True, timeout=30
            )
        except subprocess.TimeoutExpired:
            print("\n--- TIMEOUT DURANTE LA CREAZIONE DEL BRANCH ---")
            raise SystemExit("Timeout durante la creazione del branch di lavoro.")
        except subprocess.CalledProcessError as e:
            print("\n--- ERRORE DURANTE LA CREAZIONE DEL BRANCH ---")
            print(f"Errore standard:\n{e.stderr.decode('utf-8', errors='ignore')}")
            print("-----------------------------------------")
            raise SystemExit("Impossibile creare il branch di lavoro.")

    # --- FUNZIONE AGGIORNATA per un commit selettivo ---
    def commit_and_push(self, repo_path: str, branch_name: str, message: str, updated_files: list,
                        fork_url: str = None) -> bool:
        from pathlib import Path

        # Validazione input
        validate_branch_name(branch_name)
        if fork_url:
            validate_git_url(fork_url)

        try:
            if not updated_files:
                print("  -> Nessun file è stato modificato, nessun commit da creare.")
                return False

            repo_path_obj = Path(repo_path).resolve()

            print("  -> Aggiunta selettiva dei file modificati...")
            for file_path in updated_files:
                # Validazione del percorso per prevenire path traversal
                file_path_obj = Path(file_path).resolve()

                # Verifica che il file sia sotto repo_path
                try:
                    file_path_obj.relative_to(repo_path_obj)
                except ValueError:
                    print(f"  -> ATTENZIONE: File {file_path} non è sotto {repo_path}, saltato")
                    continue

                # Usa percorsi relativi per git add
                relative_path = file_path_obj.relative_to(repo_path_obj)
                subprocess.run(
                    ["git", "add", str(relative_path)],
                    cwd=repo_path, check=True, capture_output=True, timeout=30
                )

            print("  -> Esecuzione del commit...")
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_path, check=True, capture_output=True, timeout=30
            )

            remote_to_push = 'fork' if fork_url else 'origin'
            print(f"  -> Push delle modifiche sul remote '{remote_to_push}' (branch: '{branch_name}')...")

            subprocess.run(
                ["git", "push", "-u", remote_to_push, branch_name],
                cwd=repo_path, check=True, capture_output=True, timeout=180
            )

            print("  -> Push completato con successo.")
            return True
        except subprocess.TimeoutExpired as e:
            print("\n--- TIMEOUT DURANTE L'ESECUZIONE DI GIT ---")
            print(f"Comando: {' '.join(e.cmd)}")
            print("------------------------------------")
            return False
        except subprocess.CalledProcessError as e:
            print("\n--- ERRORE DURANTE L'ESECUZIONE DI GIT ---")
            print(f"Comando fallito: {' '.join(e.cmd)}")
            print(f"Errore standard:\n{e.stderr.decode('utf-8', errors='ignore')}")
            print("------------------------------------")
            return False
        except Exception as e:
            print(f"  -> Errore imprevisto durante il commit/push: {e}")
            return False

    def create_pull_request(self, upstream_repo, head_branch, base_branch, title, body, is_fork: bool):
        # Validazione input
        validate_branch_name(head_branch)
        validate_branch_name(base_branch)

        head_ref = f"{self.user.login}:{head_branch}" if is_fork else head_branch

        pulls = upstream_repo.get_pulls(state='open', head=head_ref, base=base_branch)
        if pulls.totalCount > 0:
            print(f"\n[!] Una Pull Request da '{head_ref}' a '{base_branch}' esiste già.")
            print(f"  -> URL: {pulls[0].html_url}")
            return

        print(f"\n[+] Creazione della Pull Request da '{head_ref}' a '{base_branch}'...")
        try:
            pr = upstream_repo.create_pull(
                title=title,
                body=body,
                head=head_ref,
                base=base_branch
            )
            print(f"  -> Pull Request creata con successo!")
            print(f"  -> URL: {pr.html_url}")
        except Exception as e:
            print(f"  -> Errore durante la creazione della Pull Request: {e}")


def setup_temp_dir():
    return tempfile.mkdtemp()


def handle_remove_readonly(func, path, exc_info):
    # Rimuove il flag di sola lettura e ritenta
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup_temp_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path, onerror=handle_remove_readonly)


