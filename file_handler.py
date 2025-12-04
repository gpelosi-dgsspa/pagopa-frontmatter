import frontmatter
from pathlib import Path
import os


def scan_markdown_files(root_path: Path | str) -> list[Path]:
    """Scansiona ricorsivamente una directory e restituisce una lista di file .md."""
    # --- CORREZIONE: Converte il percorso da stringa a oggetto Path se necessario ---
    # Questo assicura che possiamo sempre usare metodi come .is_dir()
    root_path = Path(root_path)

    if not root_path.is_dir():
        raise SystemExit(f"Errore: Il percorso '{root_path}' non è una directory valida.")
    return list(root_path.glob('**/*.md'))


def update_file_with_frontmatter(file_path: Path, new_frontmatter_data: dict, force: bool = False):
    """
    Legge un file markdown, aggiorna il suo frontmatter e lo salva.
    """
    try:
        # Legge il file ignorando possibili errori di encoding
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            post = frontmatter.load(f)

        # Controlla se il file ha già metadati e se non si sta forzando la sovrascrittura
        # Nota: post.metadata è sempre un dict, ma potrebbe essere vuoto {}
        if post.metadata is not None and len(post.metadata) > 0 and not force:
            print(f"  -> File già con frontmatter. Saltato (usa --force per sovrascrivere).")
            return False

        # Unisce i vecchi metadati (se presenti) con i nuovi
        # NOTA: update() sovrascrive i valori esistenti con quelli nuovi.
        # Se si desidera preservare alcuni campi specifici, decommentare il codice seguente:
        #
        # preserved_fields = ['author', 'date', 'custom_field']  # Campi da preservare
        # preserved_data = {k: post.metadata[k] for k in preserved_fields if k in post.metadata}
        # post.metadata.update(new_frontmatter_data)
        # post.metadata.update(preserved_data)  # Ripristina campi preservati

        post.metadata.update(new_frontmatter_data)

        # Convertiamo esplicitamente il post in una stringa prima di scrivere
        # Questo ci dà pieno controllo sull'encoding e previene l'errore str/bytes.
        new_file_content = frontmatter.dumps(post)

        # Scrive la stringa risultante nel file, assicurando la codifica UTF-8
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_file_content)

        return True

    except Exception as e:
        print(f"  -> Errore durante la scrittura del file: {e}")
        return False

