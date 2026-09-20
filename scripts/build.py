"""Build the Markdown pages into a plain HTML website. Run: python scripts/build.py"""
from datetime import datetime
from html import escape
from pathlib import Path
from string import Template
from urllib.parse import urljoin, urlparse
import shutil

from bs4 import BeautifulSoup
import markdown
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "_site"

ICONS = {
    "Email": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 6 9 7 9-7"/>',
    "GitHub": '<path fill="currentColor" stroke="none" d="M12 .9a11.1 11.1 0 0 0-3.5 21.6c.6.1.8-.2.8-.5v-2.1c-3.4.7-4.1-1.4-4.1-1.4-.5-1.3-1.2-1.6-1.2-1.6-1.1-.8.1-.8.1-.8 1.2.1 1.8 1.2 1.8 1.2 1.1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.8-1.6-2.7-.3-5.6-1.4-5.6-6.1 0-1.3.5-2.4 1.2-3.3-.1-.3-.5-1.6.1-3.3 0 0 1-.3 3.4 1.3a11.8 11.8 0 0 1 6.2 0C17.9 3.7 19 4 19 4c.6 1.7.2 3 .1 3.3.8.9 1.2 2 1.2 3.3 0 4.7-2.9 5.8-5.6 6.1.4.4.8 1.1.8 2.2V22c0 .3.2.6.8.5A11.1 11.1 0 0 0 12 .9Z"/>',
    "Substack": '<path d="M4 3h16M4 7h16"/><path fill="currentColor" stroke="none" d="M4 11h16v12l-8-5-8 5z"/>',
}


def render(path, page_dir):
    """Resolve asset paths relative to the page folder, not the output location."""
    soup = BeautifulSoup(markdown.markdown(path.read_text(encoding="utf-8")), "html.parser")
    for tag in soup.select("[href], [src]"):
        attr = "src" if tag.has_attr("src") else "href"
        value = tag[attr]
        if not urlparse(value).scheme and not value.startswith(("/", "#")):
            tag[attr] = urljoin(f"/{page_dir}/", value)
    for img in soup.select("img"):
        if not urlparse(img["src"]).scheme:
            source = ROOT / img["src"].lstrip("/")
            with Image.open(source) as image:
                img["width"], img["height"] = image.size
        img["loading"] = "lazy"
        img["decoding"] = "async"
    return soup


def contacts():
    soup = render(ROOT / "home/contact.md", "home")
    links = []
    for link in soup.select("a"):
        label = link.get_text()
        if label not in ICONS:
            raise ValueError(f"Unknown contact icon: {label}")
        icon = f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">{ICONS[label]}</svg>'
        links.append(f'<a href="{escape(link["href"], quote=True)}" aria-label="{escape(label)}" title="{escape(label)}">{icon}</a>')
    return '<footer class="contact" aria-label="Contact links">' + "\n".join(links) + "</footer>"


def home():
    figure = render(ROOT / "home/figure.md", "home")
    img = figure.find("img").extract()
    img["loading"] = "eager"
    img["fetchpriority"] = "high"
    caption = figure.find_all("p")[-1].decode_contents()
    intro = render(ROOT / "home/index.md", "home")
    updates = render(ROOT / "home/updates.md", "home")
    banner_text = escape((ROOT / "home/banner.txt").read_text(encoding="utf-8").strip())
    banner = f'''<div class="ascii-banner" aria-hidden="true">
  <div class="ascii-banner-track">
    <pre>{banner_text}</pre>
    <pre>{banner_text}</pre>
  </div>
</div>'''
    updates.h2["id"] = "updates-heading"
    for item in updates.select("li"):
        date = item.find("strong")
        if date is None:
            raise ValueError("Every update needs a bold date, e.g. **Oct 2026:**")
        label = date.get_text().rstrip(":")
        # Accept both short and full English month names, including 'Sept'.
        normalized = label.replace("Sept ", "Sep ")
        parsed = None
        for fmt in ("%b %Y", "%B %Y"):
            try:
                parsed = datetime.strptime(normalized, fmt)
                break
            except ValueError:
                pass
        if parsed is None:
            raise ValueError(f"Unrecognized update date: {label}")
        date.extract()
        contents = item.decode_contents().strip()
        item.clear()
        time = updates.new_tag("time", datetime=parsed.strftime("%Y-%m"))
        time.string = label
        text = updates.new_tag("span")
        text.append(BeautifulSoup(contents, "html.parser"))
        item.extend([time, text])
    return f'''<h1 class="sr-only">Roy Xing</h1>
<section class="intro" aria-label="About me">
  <figure id="figure-1">{img}<figcaption>{caption}</figcaption></figure>
  <div class="intro-copy">{intro}</div>
</section>
{banner}
<section class="updates" aria-labelledby="updates-heading">{updates}</section>'''


def research():
    heading = render(ROOT / "research/index.md", "research")
    papers = render(ROOT / "research/publications.md", "research")
    output = []
    current = None
    for element in list(papers.children):
        if element.name == "h2":
            current = None
            output.append(str(element))
        elif element.name == "h3":
            current = papers.new_tag("article", attrs={"class": "publication"})
            current.append(element.extract())
            output.append(current)
        elif current is not None and element.name:
            current.append(element.extract())
    return str(heading) + '<div class="publications">' + "\n".join(map(str, output)) + "</div>"


def projects():
    output = []
    for path in sorted((ROOT / "projects/items").glob("*.md")):
        content = render(path, "projects")
        heading, img = content.find("h2"), content.find("img")
        if heading is None or img is None:
            raise ValueError(f"{path.name} needs a ## title and a Markdown image.")
        image_paragraph = img.parent
        img.extract()
        image_paragraph.decompose()
        heading.extract()
        source = ROOT / img["src"].lstrip("/")
        still = source.with_name(source.stem + "-still.webp")
        visual = f"<picture>{img}</picture>"
        if still.exists():
            still_url = "/" + still.relative_to(ROOT).as_posix()
            visual = f'<picture><source media="(prefers-reduced-motion: reduce)" srcset="{still_url}">{img}</picture>'
        output.append(f'<article class="project">{visual}{heading}{content}</article>')
    return str(render(ROOT / "projects/index.md", "projects")) + '<div class="project-grid">' + "\n".join(output) + "</div>"


def page(route, title, content, page_class, footer=""):
    template = Template((ROOT / "site/template.html").read_text())
    html = template.substitute(
        title=escape(title),
        description="Roy Xing — PhD student in computer science at Dartmouth College. Robotics, motion, and embodied intelligence.",
        content=content, page_class=page_class, footer=footer,
        home_current=' aria-current="page"' if route == "index.html" else "",
        research_current=' aria-current="page"' if route.startswith("research/") else "",
        projects_current=' aria-current="page"' if route.startswith("projects/") else "",
    )
    target = OUTPUT / route
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


def build():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir()
    for folder in ["home/images", "projects/images"]:
        shutil.copytree(ROOT / folder, OUTPUT / folder)
    for asset in ["site/style.css", "site/favicon.svg", "cv/cv.pdf"]:
        target = OUTPUT / asset
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / asset, target)
    page("index.html", "Roy Xing", home(), "home-page", contacts())
    page("research/index.html", "Research · Roy Xing", research(), "research-page")
    page("projects/index.html", "Projects · Roy Xing", projects(), "projects-page")
    page("404.html", "Page not found · Roy Xing", '<h1>Page not found</h1><p><a href="/">Back to the homepage</a>.</p>', "not-found")
    (OUTPUT / ".nojekyll").touch()
    print("Built homepage, research, projects, CV, and 404 into _site/.")


if __name__ == "__main__":
    build()
