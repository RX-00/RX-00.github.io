"""Exercise real Scholar markup, pagination, and safe failure behavior offline."""
from pathlib import Path
import tempfile
import unittest

from scripts.sync_scholar import fetch_publications, parse_profile, render_publications, sync

FIXTURE = (Path(__file__).parent / "fixtures/scholar.html").read_text()


class ScholarTests(unittest.TestCase):
    def test_real_profile_has_all_three_publications(self):
        papers, more = parse_profile(FIXTURE)
        self.assertEqual(len(papers), 3)
        self.assertFalse(more)
        self.assertEqual(papers[0]["url"], "https://arxiv.org/abs/2606.26392")
        self.assertEqual(papers[0]["authors"], "R Xing, S Ree, B Plancher")
        self.assertEqual(papers[-1]["year"], "2022")

    def test_blocked_empty_or_incomplete_pages_are_rejected(self):
        for html in ("<h1>Sorry, unusual traffic</h1>", '<div id="gsc_prf_in">Roy Xing</div>', FIXTURE.replace('class="gsc_a_at"', 'class="changed"'), FIXTURE.replace('id="gsc_bpf_more"', 'id="changed"')):
            with self.subTest(html=html[:60]), self.assertRaises(ValueError):
                parse_profile(html)

    def test_pagination_collects_later_pages(self):
        first = FIXTURE.replace('disabled=""', '').replace('disabled', '')
        second = FIXTURE.replace('MPC-Injection:', 'New paper:').replace('2606.26392', '2607.12345')
        urls = []
        def fetch(url):
            urls.append(url)
            return first if len(urls) == 1 else second
        papers = fetch_publications(fetch)
        self.assertEqual(len(papers), 4)
        self.assertIn("cstart=100", urls[1])

    def test_partial_fetch_does_not_replace_saved_markdown(self):
        first = FIXTURE.replace('disabled=""', '').replace('disabled', '')
        calls = []
        def fetch(url):
            calls.append(url)
            if len(calls) > 1:
                raise TimeoutError("Scholar unavailable")
            return first
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "publications.md"
            saved.write_text("Previous successful publication list")
            with self.assertRaises(TimeoutError):
                sync(saved, fetch)
            self.assertEqual(saved.read_text(), "Previous successful publication list")

    def test_repeated_page_is_rejected(self):
        first = FIXTURE.replace('disabled=""', '').replace('disabled', '')
        with self.assertRaises(ValueError):
            fetch_publications(lambda _: first)

    def test_sync_is_repeatable_and_groups_by_year(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "publications.md"
            sync(saved, lambda _: FIXTURE)
            original = saved.read_bytes()
            original_time = saved.stat().st_mtime_ns
            sync(saved, lambda _: FIXTURE)
            self.assertEqual(saved.read_bytes(), original)
            self.assertEqual(saved.stat().st_mtime_ns, original_time)
            self.assertEqual(saved.read_text().count("## 2026\n"), 1)

    def test_non_arxiv_papers_keep_scholar_link_and_undated_group(self):
        changed = FIXTURE.replace('arXiv preprint arXiv:2606.26392', 'Conference on Robot Learning')
        papers, _ = parse_profile(changed)
        self.assertTrue(papers[0]["url"].startswith("https://scholar.google.com/citations?"))
        papers[0]["year"] = ""
        self.assertIn("## Undated", render_publications([papers[0]]))


if __name__ == "__main__":
    unittest.main()
