"""Check the public artifact, including navigation and page-local assets."""
from pathlib import Path
import unittest
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup

from scripts.build import OUTPUT, build


class WebsiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build()

    def test_internal_links_images_and_fragments_resolve(self):
        for path in OUTPUT.rglob("*.html"):
            soup = BeautifulSoup(path.read_text(), "html.parser")
            route = "/" + path.relative_to(OUTPUT).as_posix()
            for tag in soup.select("[href], [src], [srcset]"):
                for attr in ("href", "src", "srcset"):
                    if attr not in tag.attrs:
                        continue
                    url = urlparse(urljoin(route, tag[attr]))
                    if url.scheme or url.netloc:
                        continue
                    with self.subTest(page=route, target=tag[attr]):
                        target = OUTPUT / unquote(url.path).lstrip("/")
                        if target.is_dir():
                            target /= "index.html"
                        self.assertTrue(target.is_file(), str(target))
                        if url.fragment:
                            linked = BeautifulSoup(target.read_text(), "html.parser")
                            self.assertIsNotNone(linked.find(id=url.fragment))

    def test_pages_work_without_scripts_or_third_party_assets(self):
        for path in OUTPUT.rglob("*.html"):
            soup = BeautifulSoup(path.read_text(), "html.parser")
            self.assertFalse(soup.select("script, iframe"))
            self.assertEqual(len(soup.select("main")), 1)
            self.assertEqual(len(soup.select("h1")), 1)
            for img in soup.select("img"):
                self.assertTrue(img.get("alt"))
                self.assertTrue(img.get("width"))
                self.assertTrue(img.get("height"))
                self.assertTrue(img["src"].startswith("/"))

    def test_accessible_home_structure(self):
        soup = BeautifulSoup((OUTPUT / "index.html").read_text(), "html.parser")
        self.assertEqual(len(soup.select(".intro-copy > p")), 3)
        self.assertEqual(len(soup.select(".ascii-banner .ascii-banner-track > pre")), 2)
        self.assertEqual(soup.select_one(".ascii-banner").get("aria-hidden"), "true")
        self.assertGreater(len(soup.select(".updates li time[datetime]")), 0)
        self.assertEqual(len(soup.select("figure#figure-1 figcaption")), 1)
        self.assertEqual([a.get("aria-label") for a in soup.select(".contact a")], ["Email", "GitHub", "Substack"])
        self.assertTrue(soup.select_one('.contact a[aria-label="Email"]')["href"].startswith("mailto:"))

    def test_old_site_is_not_published(self):
        for old in ("blog", "assets", "categories", "feed.xml", "README.md", "scripts", "tests"):
            self.assertFalse((OUTPUT / old).exists(), old)
        soup = BeautifulSoup((OUTPUT / "projects/index.html").read_text(), "html.parser")
        self.assertGreater(len(soup.select(".project h2")), 0)
        self.assertFalse(soup.select(".modal, button"))

    def test_cv_and_research_are_present(self):
        self.assertTrue((OUTPUT / "cv/cv.pdf").read_bytes().startswith(b"%PDF"))
        soup = BeautifulSoup((OUTPUT / "research/index.html").read_text(), "html.parser")
        papers = soup.select(".publication h3 a")
        self.assertGreater(len(papers), 0)


if __name__ == "__main__":
    unittest.main()
