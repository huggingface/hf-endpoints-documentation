# CLAUDE.md

Project context and conventions for the Hugging Face Inference Endpoints documentation.

## Tech Stack

- **Builder**: [`hf-doc-builder`](https://github.com/huggingface/doc-builder) (`pip install hf-doc-builder==0.4.0 watchdog --upgrade`)
- **Content**: Markdown (`.md`) under `docs/source/`, with a few custom MDX-like components (`<Tip>`, iframes)
- **Deployment**: auto-built and pushed by `huggingface/doc-builder`'s reusable workflow on merge to `main`; PRs get their own preview build

## Development Commands

```bash
doc-builder preview endpoints docs/source/ --not_python_module   # local live preview
doc-builder build endpoints docs/source/ --build_dir build/ --not_python_module  # build
node scripts/generate-illustration.mjs   # regenerate homepage hero SVGs after editing illustration.svelte
```

## Project Structure

```
docs/source/
  _toctree.yml        # left-nav structure — REQUIRED for any new/removed page, see below
  _redirects.yml       # old-path -> new-path redirects for renamed/removed pages
  index.md, quick_start.md, about.md, api_reference.md   # Overview section
  tutorials/           # end-to-end "build this" walkthroughs (chat bot, embedding pipeline, transcription)
  support/             # pricing, FAQ
  guides/              # how-to / conceptual pages on one feature area (autoscaling, logs, security, ...)
  engines/             # one page per supported inference engine (vLLM, TGI, SGLang, TEI, llama.cpp, custom containers)
assets/                 # images referenced by docs (see "Adding Images" below)
scripts/generate-illustration.mjs  # regenerates the homepage hero illustration SVGs from illustration.svelte
.github/workflows/      # doc-build.yaml (push to main), doc-pr-build.yaml / doc-pr-upload.yaml (PR previews)
```

**Where new content belongs:**
- A new feature/setting on an existing concept (e.g. a new autoscaling option) → add a section to the relevant `guides/*.md` page, not a new file.
- A brand-new feature area with no existing home → new file under `guides/`.
- A full worked "build an app using this" walkthrough → `tutorials/`.
- Engine-specific behavior/config → the matching `engines/*.md` page.
- `api_reference.md` is mostly a live Swagger iframe pulling the deployed OpenAPI spec directly — don't hand-document individual API fields there; that's what the `guides/` pages and the spec itself are for.

## Adding or Removing a Page

1. Create/remove the `.md` file under the right subdirectory above.
2. **Register it in `docs/source/_toctree.yml`** (`local: <path-without-.md>`, `title: <nav label>`) — a page not listed here won't appear in the left nav even if it builds.
3. If renaming or removing a previously-published page, add an entry to `docs/source/_redirects.yml` (`old/path: new/path`) so existing links don't 404.

## Doc Conventions

- Use `<Tip>...</Tip>` blocks for callouts/caveats (warnings, defaults, non-obvious behavior) — see any file under `guides/` for examples.
- Heading levels: `#` page title (one per file), `##` major sections, `###` subsections.
- Cross-reference other pages with relative links, optionally with an anchor built from the heading text lowercased with spaces removed, e.g. `[here](./analytics#hardwareutilisation)`.
- Prose is second-person ("you can configure..."), concise, and prefers short paragraphs over long ones.
- When documenting an API field/config shape, use a fenced ```json block showing a realistic request snippet rather than a prose-only description.

## Adding Assets/Images

Images can only be embedded via `https://` links — reference them with the
`https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/` prefix, not a relative path.

The homepage hero illustration is generated from `illustration.svelte` (source of truth for geometry/animation); after
editing it, regenerate the committed SVGs with `node scripts/generate-illustration.mjs`. Engine logos are vendored under
`assets/logos/` (copied from the Inference Endpoints site) so the illustration doesn't depend on external URLs; refresh
them with `curl -sL -o assets/logos/FILE.svg https://endpoints.huggingface.co/logos/FILE.svg` if they change upstream.

## Related Repo

The product code this documentation describes lives in `hf-endpoints` (sibling directory). When a PR there ships a
user-facing feature, check whether the corresponding `guides/*.md` page here needs updating.
