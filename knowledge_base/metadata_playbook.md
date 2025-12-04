# Metadata playbook

This knowledge base defines a neutral blueprint for generating YAML frontmatter and Schema.org JSON-LD.

## Frontmatter blueprint
```yaml
frontmatter_blueprint:
  document:
    title: required
    summary: required
    type: required
    topics: optional_list
  audience:
    primaryRole: required
    additionalRoles: optional_list
    experienceLevel: optional
  lifecycle:
    status: optional
    lastUpdated: optional
  metadata:
    keywords: optional_list
    relatedResources: optional_list
```

- Treat `required` fields as mandatory. If a value cannot be inferred, provide the best concise description based on the Markdown source.
- `optional_list` indicates that the field should appear only when information is available; otherwise omit the key entirely.
- Use domain-specific vocabulary for `document.type` and `audience.primaryRole` derived from the document itself.
- Keep `summary` under 30 words when possible.

## Taxonomy guidance
- Prefer verbs or task-oriented nouns for `document.type` when the content describes procedures (e.g., `How-To`, `Tutorial`).
- Use technology names, platform components, or conceptual topics for `document.topics`.
- Choose `audience.experienceLevel` from `beginner`, `intermediate`, or `advanced` when the document implies a clear skill expectation.

## Schema hints
```yaml
schema_hints:
  preferred_types:
    - TechArticle
    - Article
    - HowTo
  common_properties:
    - name
    - description
    - inLanguage
    - keywords
    - learningResourceType
    - audience
  audience_template:
    "@type": "Audience"
    "audienceType": <role inferred from audience.primaryRole>
```

- Select the Schema.org `@type` that best matches the Markdown content while staying within the `preferred_types` list when applicable.
- Map `audience.primaryRole` into the `audience` property using the `audience_template`.
- Include additional properties only when explicitly supported by the retrieved Schema.org definitions.
