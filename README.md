# Roy Xing's website

A small academic website made from Markdown, plain HTML, and CSS. No browser
JavaScript, external fonts, frontend framework, or database.

## Edit the website

Open a file in GitHub, click the pencil, make your changes, and commit them to
`master`. GitHub Actions builds and publishes the website automatically.

| What you want to change | File |
| --- | --- |
| Introductory paragraph | `home/index.md` |
| BMO image and its caption | `home/figure.md` and `home/images/` |
| News and updates | `home/updates.md` |
| Email, GitHub, and Substack links | `home/contact.md` |
| Research page introduction | `research/index.md` |
| Publications | Automatically imported into `research/publications.md` |
| CV | Replace **`cv/cv.pdf`**, keeping that filename |
| Projects page heading | `projects/index.md` |
| Project titles and captions | One Markdown file per project in `projects/items/` |
| Project pictures | `projects/images/` |
| Colors, spacing, and fonts | `site/style.css` |
| Your name and navigation | `site/template.html` |

Text files use ordinary Markdown: `[link text](https://example.com)`,
`**bold text**`, and `![image description](images/photo.webp)`.
The homepage introduction is one paragraph; a blank line starts a new paragraph.

### Add an update

Add a line at the top of the list in `home/updates.md`:

```markdown
- **Oct 2026:** Your news, with an optional [link](https://example.com).
```

Use a month and year in the bold label. Updates appear in file order, so you
control their order and can include upcoming events.

### Add or change a project

1. Put the image in `projects/images/`. WebP, JPEG, and PNG work.
2. Copy a file in `projects/items/` and edit it:

```markdown
## Project title

![A useful description of the picture](images/my-robot.webp)

A short caption about the project.
```

The filename controls the order (`01-…`, `02-…`, etc.). Images use paths relative
to the **projects page**, as above. Delete a project's Markdown file to remove it.
The gallery shows images, titles, and captions; it has no popups or detail pages.
The two existing animated images have `-still.webp` companions, shown to visitors
who prefer reduced motion. Existing images have been optimized for the web.

### Automatic research updates

The workflow checks [Google Scholar](https://scholar.google.com/citations?user=dq-VzqkAAAAJ&hl=en)
every Monday, on pushes, and when run manually. It imports the full publication
list, groups it by year, and links arXiv entries directly to the papers. Other
entries link to their Scholar record. No API keys are needed.

`research/publications.md` is the saved publication list and is committed when it
changes. It will be overwritten by the next successful refresh; put your own
research introduction in `research/index.md` instead. Google sometimes blocks
automated requests. If that happens, the workflow logs a warning and publishes
the last saved list instead of breaking the website or erasing publications.
You can run **Actions → Build and deploy website → Run workflow** to retry.

GitHub may disable scheduled workflows in public repositories after 60 days
without activity. Re-enable the workflow in the Actions tab if that happens.

### Run locally

Python 3.12 or newer is recommended. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build.py
python -m http.server 8000 --directory _site
```

Open `http://localhost:8000`. After editing Markdown or CSS, rerun
`python scripts/build.py` and refresh the browser. The local build works offline
after dependencies are installed. `_site/` is generated; do not edit it.

Optional checks:

```sh
python scripts/sync_scholar.py
python scripts/build.py
python -m unittest discover -s tests -v
```

### Deployment

In the repository's **Settings → Pages**, the source should be **GitHub Actions**.
The workflow deploys `_site/` on `master`; pull requests only build and test.
The site is configured for the root domain `https://rx-00.github.io/`.
There is no last-updated year or copyright date displayed on the site.

The previous site and original full-size images remain available in Git history
at commit `7489da5`.
