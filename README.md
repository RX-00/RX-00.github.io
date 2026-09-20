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
| Animated ASCII banner | `home/banner.txt` |
| News and updates | `home/updates.md` |
| Email, GitHub, and Substack links | `home/contact.md` |
| Research page introduction | `research/index.md` |
| Publications | `research/publications.md` |
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

### Add or update a publication

Edit `research/publications.md` directly. Publications appear in file order,
grouped under year headings. Copy an existing entry or use this format:

```markdown
## 2026

### [Paper title](https://arxiv.org/abs/your-paper)

Author names

Conference, journal, or preprint information
```

Add papers under the appropriate year, with newer years first. Your research
introduction lives in `research/index.md`. The Google Scholar link remains on the
research page, but there is no automatic import to overwrite your edits.

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
