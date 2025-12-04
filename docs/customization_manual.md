# Customizing the Knowledge Base and Master Prompt

This manual explains how to adapt the AI Frontmatter Injector to your own metadata model. It covers how the runtime assembles the knowledge base, how to extend or replace the master prompt, and how to validate your changes end to end.

## 1. Architectural overview

The frontmatter workflow is orchestrated by `ai_core.load_prompt_and_knowledge_base()`, which reads the prompt template from `config/master_prompt.txt` and concatenates every non-RDF file found in `knowledge_base/`. The combined text replaces the `{{KNOWLEDGE_BASE_CONTENT}}` placeholder in the prompt before the Markdown document and Schema.org snippets are injected.【F:ai_core.py†L16-L36】【F:config/master_prompt.txt†L1-L75】 During processing, `processing_core.process_folder()` feeds the assembled prompt to the LLM and writes the generated YAML frontmatter back to disk.【F:processing_core.py†L8-L66】 Understanding this pipeline will help you predict how configuration edits affect the AI’s behaviour.

## 2. Designing knowledge-base bundles

The knowledge base acts as a portable configuration bundle. Each file contributes literal text to the prompt, so structure your content for readability:

- **File types** – Markdown (`.md`), YAML (`.yml`/`.yaml`), and plain text are all safe. Files whose names contain `schemaorg` are ignored to prevent duplicating the RDF dump.【F:ai_core.py†L16-L36】
- **Segmentation** – Organize large playbooks into focused sections (for example `frontmatter_blueprint.md`, `taxonomy.md`, `schema_hints.md`). The loader preserves file order based on glob results, so prefix filenames with numbers (`01_`, `02_`) if ordering matters.
- **YAML blocks** – Describe blueprints and policy tables inside fenced YAML blocks. The runtime does not parse these blocks but passes them verbatim to the LLM, which then follows the instructions (see the stock `metadata_playbook.md` for an example).【F:knowledge_base/metadata_playbook.md†L1-L48】

### 2.1 Blueprint customization checklist

1. Draft the desired YAML structure inside a code block under a `frontmatter_blueprint` key.
2. Mark each field as `required`, `optional`, or `optional_list` so the LLM knows when to emit them.
3. Include plain-language rules beneath the code block to explain how to infer each value.
4. Keep lists of allowed values up to date; prefer short, action-oriented labels that appear in your documentation corpus.

### 2.2 Schema guidance

- Provide `schema_hints` blocks describing acceptable Schema.org `@type` values, common properties, or JSON-LD templates.【F:knowledge_base/metadata_playbook.md†L50-L77】
- When you introduce a new `@type`, ensure the corresponding Schema.org definitions exist in your ChromaDB index (see §4).
- Use placeholder markers like `<role inferred from audience.primaryRole>` inside templates to show the LLM how to map frontmatter fields into JSON-LD.

### 2.3 Localization and tone

The prompt inherits the language and tone of your knowledge-base text. Write instructions in the language you expect the AI to use, and call out localisation rules explicitly (for example, “Always output ISO 8601 dates” or “Prefer British English spelling”).

## 3. Updating the master prompt

The master prompt in `config/master_prompt.txt` defines the workflow and must always expose the following placeholders: `{{KNOWLEDGE_BASE_CONTENT}}`, `{{SCHEMA_DEFINITIONS}}`, and `{{MARKDOWN_CONTENT}}`.【F:config/master_prompt.txt†L9-L25】 Keep these tokens intact or the runtime will fail to substitute the relevant sections.

### 3.1 Safe editing practices

- **Preserve objective and process headers** – The current prompt separates sections (`OBJECTIVE`, `CONTEXT`, `PROCESS`, etc.), which helps keep instructions scannable. You can rewrite the prose inside each section but maintain a clear hierarchy.
- **Reference knowledge-base rules** – Reinforce how the LLM should use the blueprint (for example “Follow the localisation policy in the knowledge base”).
- **Document YAML output rules** – Add or adjust constraints in the `OUTPUT RULES` section when your schema changes (e.g., new indentation requirements, quoting rules, or list semantics).【F:config/master_prompt.txt†L27-L75】
- **Validate example output** – Update the `EXAMPLE OUTPUT` block whenever the blueprint changes so that reviewers and future maintainers have a canonical reference.【F:config/master_prompt.txt†L76-L117】

### 3.2 Adding new placeholders

If you need the runtime to inject additional context (for example, repository metadata), you must:

1. Reserve a unique placeholder token (e.g., `{{REPO_CONTEXT}}`) in the prompt template.
2. Update `ai_core.load_prompt_and_knowledge_base()` to fetch the new resource and perform the substitution before sending the prompt to the LLM.
3. Pass the new content when calling `ai_core.generate_frontmatter()` so the placeholder is replaced at runtime.

## 4. Maintaining the Schema.org index

Your schema hints are only effective when the underlying ChromaDB collection contains the referenced definitions. After editing `knowledge_base/` to introduce new Schema.org types or properties:

1. Ensure the correct embedding provider is configured via environment variables (`LLM_PROVIDER`, `EMBEDDING_PROVIDER`, and the corresponding API key).【F:ai_core.py†L41-L129】
2. Run the indexer to ingest the latest Schema.org data:
   ```bash
   python indexer.py
   ```
3. Confirm that `schemaorg-current-https.rdf` (or your custom RDF file) is present in `knowledge_base/`; `indexer.py` reads this file to populate the vector store.
4. Re-run your frontmatter workflow on a sample document to verify that the retrieved definitions match the new guidance.

## 5. Testing and validation workflow

1. **Dry runs** – Execute your pipeline in `dry_run` mode (if you add such a flag) or run against disposable branches to inspect the generated YAML before committing changes. The CLI will print progress for each Markdown file processed.【F:processing_core.py†L8-L66】
2. **Manual review** – Compare the generated frontmatter against your blueprint to ensure required fields are present and optional fields are omitted when data is missing.
3. **Schema validation** – Use external JSON-LD validators to confirm the `schema` block is valid and includes all required properties for your chosen `@type`.
4. **Regression tests** – Maintain a set of representative Markdown files and capture expected frontmatter outputs. After updating the knowledge base or prompt, rerun the tool and diff the results to catch regressions in taxonomy or formatting.

## 6. Deployment tips

- Version the entire `knowledge_base/` directory so that prompt changes can be reviewed alongside code updates.
- For per-client customisations, ship separate knowledge-base bundles and swap them by copying the desired bundle into `knowledge_base/` before running the CLI.
- When automating through `github_main.py`, ensure your new knowledge base is committed to the branch so remote runners pick up the latest configuration.

By treating the knowledge base and master prompt as first-class configuration artifacts, you can tailor the AI Frontmatter Injector to a wide variety of documentation ecosystems without touching the core processing logic.
