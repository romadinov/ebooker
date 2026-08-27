"""EPUB -> ordered, narratable chapters.

Deliberately defensive: the library is largely FB2-converted EPUBs whose markup
is inconsistent between books. Anything structurally surprising is recorded on
the chapter as a warning rather than raised, so one odd book cannot stall a
library run.
"""

from __future__ import annotations

import json
import posixpath
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

from lxml import etree

NS = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "epub": "http://www.idpf.org/2007/ops",
}

# Blocks that carry no narratable prose.
SKIP_TAGS = {"script", "style", "head", "title", "svg", "img", "image", "figure"}
BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "td"}

# Chapter titles that are navigation, not content. Matched casefolded.
NON_CONTENT_TITLES = {
    "содержание", "оглавление", "contents", "table of contents", "toc",
    "cover", "обложка", "colophon", "выходные данные",
    "об авторе", "about the author", "copyright", "авторские права",
    "annotation", "аннотация",
}

FOOTNOTE_MARKER = re.compile(r"\[\s*\d{1,3}\s*\]")
WS = re.compile(r"[ \t   ]+")
MULTI_NL = re.compile(r"\n{3,}")


@dataclass
class Chapter:
    index: int
    title: str            # flattened, human-readable
    nav_path: list[str]   # hierarchy as it appeared in the TOC
    src: str              # "file.xhtml#anchor"
    paragraphs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.paragraphs)


@dataclass
class Book:
    title: str
    author: str
    language: str
    publisher: str = ""
    date: str = ""
    identifier: str = ""
    cover_path: str = ""          # path inside the epub
    chapters: list[Chapter] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(c.char_count for c in self.chapters)


def _text_of(el) -> str:
    return "".join(el.itertext())


def _localname(el) -> str:
    tag = el.tag
    if not isinstance(tag, str):
        return ""
    return etree.QName(tag).localname.lower()


def _clean_inline(s: str) -> str:
    """Normalise whitespace and drop footnote call-outs from narratable text."""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("­", "")           # soft hyphen
    s = FOOTNOTE_MARKER.sub("", s)
    s = WS.sub(" ", s.replace("\n", " ").replace("\r", " "))
    return s.strip()


def _clean_title(s: str) -> str:
    s = _clean_inline(s)
    return s.strip(" .—-–\t")


class Epub:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.zf = zipfile.ZipFile(self.path)
        self.opf_path = self._find_opf()
        self.root = posixpath.dirname(self.opf_path)
        self.opf = self._xml(self.opf_path)

    def close(self) -> None:
        self.zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- zip / xml helpers ------------------------------------------------
    def _xml(self, name: str):
        return etree.fromstring(self.zf.read(name))

    def _resolve(self, href: str) -> str:
        return posixpath.normpath(posixpath.join(self.root, href)) if self.root else href

    def _find_opf(self) -> str:
        c = self._xml("META-INF/container.xml")
        rf = c.find(".//container:rootfile", NS)
        if rf is None or not rf.get("full-path"):
            raise ValueError("EPUB has no rootfile in container.xml")
        return rf.get("full-path")

    # -- metadata --------------------------------------------------------
    def _meta(self) -> dict:
        def dc(tag: str) -> str:
            el = self.opf.find(f".//dc:{tag}", NS)
            return _clean_inline(_text_of(el)) if el is not None else ""

        return {
            "title": dc("title"),
            "author": dc("creator"),
            "language": (dc("language") or "").lower(),
            "publisher": dc("publisher"),
            "date": dc("date"),
            "identifier": dc("identifier"),
        }

    def _manifest(self) -> dict[str, dict]:
        out = {}
        for item in self.opf.findall(".//opf:manifest/opf:item", NS):
            iid = item.get("id")
            if iid:
                out[iid] = {
                    "href": item.get("href", ""),
                    "type": item.get("media-type", ""),
                    "props": (item.get("properties") or "").split(),
                }
        return out

    def _spine(self, manifest: dict) -> list[str]:
        """Spine hrefs, epub-relative, in reading order."""
        out = []
        for ref in self.opf.findall(".//opf:spine/opf:itemref", NS):
            item = manifest.get(ref.get("idref", ""))
            if item and item["href"]:
                out.append(self._resolve(item["href"]))
        return out

    def _cover(self, manifest: dict) -> str:
        # EPUB3 properties, then the EPUB2 <meta name="cover"> convention.
        for item in manifest.values():
            if "cover-image" in item["props"]:
                return self._resolve(item["href"])
        m = self.opf.find('.//opf:metadata/opf:meta[@name="cover"]', NS)
        if m is not None and (item := manifest.get(m.get("content", ""))):
            return self._resolve(item["href"])
        # Last resort: an image whose name looks like a cover.
        for item in manifest.values():
            if item["type"].startswith("image/") and "cover" in item["href"].lower():
                return self._resolve(item["href"])
        return ""

    # -- table of contents ------------------------------------------------
    def _toc_ncx(self, manifest: dict) -> list[tuple[list[str], str]]:
        ncx_href = next(
            (i["href"] for i in manifest.values()
             if i["type"] == "application/x-dtbncx+xml"), None)
        if not ncx_href:
            return []
        doc = self._xml(self._resolve(ncx_href))
        base = posixpath.dirname(self._resolve(ncx_href))
        entries: list[tuple[list[str], str]] = []

        def walk(node, trail: list[str]) -> None:
            for np in node.findall("ncx:navPoint", NS):
                label = np.find("ncx:navLabel/ncx:text", NS)
                content = np.find("ncx:content", NS)
                name = _clean_title(_text_of(label)) if label is not None else ""
                trail2 = trail + [name] if name else trail
                if content is not None and content.get("src"):
                    src = content.get("src")
                    file_part, _, frag = src.partition("#")
                    resolved = posixpath.normpath(posixpath.join(base, file_part))
                    entries.append((trail2, f"{resolved}#{frag}" if frag else resolved))
                walk(np, trail2)

        nav_map = doc.find("ncx:navMap", NS)
        if nav_map is not None:
            walk(nav_map, [])
        return entries

    def _toc_nav(self, manifest: dict) -> list[tuple[list[str], str]]:
        """EPUB3 nav document fallback."""
        nav_href = next(
            (i["href"] for i in manifest.values() if "nav" in i["props"]), None)
        if not nav_href:
            return []
        doc = self._xml(self._resolve(nav_href))
        base = posixpath.dirname(self._resolve(nav_href))
        entries: list[tuple[list[str], str]] = []

        nav = next(
            (n for n in doc.iter()
             if _localname(n) == "nav"
             and n.get(f"{{{NS['epub']}}}type") == "toc"), None)
        if nav is None:
            return []

        def walk(ol, trail: list[str]) -> None:
            for li in [c for c in ol if _localname(c) == "li"]:
                a = next((d for d in li.iter() if _localname(d) == "a"), None)
                name = _clean_title(_text_of(a)) if a is not None else ""
                trail2 = trail + [name] if name else trail
                if a is not None and a.get("href"):
                    file_part, _, frag = a.get("href").partition("#")
                    resolved = posixpath.normpath(posixpath.join(base, file_part))
                    entries.append((trail2, f"{resolved}#{frag}" if frag else resolved))
                for sub in [c for c in li if _localname(c) == "ol"]:
                    walk(sub, trail2)

        for ol in [c for c in nav if _localname(c) == "ol"]:
            walk(ol, [])
        return entries

    # -- body text --------------------------------------------------------
    def _paragraphs_by_anchor(
        self, href: str, anchors: set[str]
    ) -> tuple[dict[str, list[str]], list[str]]:
        """Split one XHTML file's paragraphs across the anchors it contains.

        Walks in document order and attributes each block to the most recently
        seen anchor. This is robust to anchors being siblings *or* wrappers,
        which differs between converters.
        """
        warnings: list[str] = []
        try:
            doc = self._xml(href)
        except (KeyError, etree.XMLSyntaxError) as e:
            return {}, [f"could not parse {href}: {e}"]

        body = next((el for el in doc.iter() if _localname(el) == "body"), doc)
        buckets: dict[str, list[str]] = {"": []}
        current = ""
        seen: set[str] = set()

        for el in body.iter():
            name = _localname(el)
            if name in SKIP_TAGS:
                continue
            eid = el.get("id")
            if eid and eid in anchors:
                current = eid
                seen.add(eid)
                buckets.setdefault(current, [])
            if name in BLOCK_TAGS:
                # Only take blocks with no nested block children, so text is
                # not emitted twice for wrapper divs.
                if any(_localname(c) in BLOCK_TAGS for c in el.iter() if c is not el):
                    continue
                txt = _clean_inline(_text_of(el))
                if txt:
                    buckets.setdefault(current, []).append(txt)

        missing = anchors - seen
        if missing:
            warnings.append(f"{href}: anchors not found in document: {sorted(missing)}")
        return buckets, warnings


def _is_non_content(title: str, nav_path: list[str]) -> bool:
    t = _clean_title(title).casefold()
    if t in NON_CONTENT_TITLES:
        return True
    return bool(nav_path) and nav_path[-1].casefold() in NON_CONTENT_TITLES


def _flatten_title(nav_path: list[str]) -> str:
    """"Плененная Вселенная" / "Начало" / "1"  ->  "Плененная Вселенная — Начало — 1".

    Bare numeric leaves are meaningless as M4B chapter names on their own, so
    they keep their parent context. Named leaves stand alone.
    """
    parts = [p for p in nav_path if p]
    if not parts:
        return "Без названия"
    leaf = parts[-1]
    if len(parts) == 1:
        return leaf
    if re.fullmatch(r"[0-9IVXLCDM]+[.)]?", leaf, re.IGNORECASE):
        return " — ".join(parts)
    # Named leaf: keep the top-level work for context if the tree is deep.
    return f"{parts[0]} — {leaf}" if len(parts) > 2 else leaf


def load(path: str | Path) -> Book:
    with Epub(Path(path)) as ep:
        meta = ep._meta()
        manifest = ep._manifest()
        spine = ep._spine(manifest)
        book = Book(
            title=meta["title"] or Path(path).stem,
            author=meta["author"],
            language=meta["language"] or "und",
            publisher=meta["publisher"],
            date=meta["date"],
            identifier=meta["identifier"],
            cover_path=ep._cover(manifest),
        )
        if not meta["language"]:
            book.warnings.append("no dc:language; language routing will need a manual hint")

        toc = ep._toc_ncx(manifest) or ep._toc_nav(manifest)
        if not toc:
            book.warnings.append("no usable TOC; falling back to one chapter per spine file")
            toc = [([Path(h).stem], h) for h in spine]

        # Group TOC entries by file, preserving spine order.
        spine_order = {h: i for i, h in enumerate(spine)}
        by_file: dict[str, list[tuple[list[str], str]]] = {}
        for nav_path, src in toc:
            file_part = src.partition("#")[0]
            by_file.setdefault(file_part, []).append((nav_path, src))

        unknown = [f for f in by_file if f not in spine_order]
        if unknown:
            book.warnings.append(f"TOC references {len(unknown)} file(s) outside the spine")

        idx = 0
        for file_part in sorted(by_file, key=lambda f: spine_order.get(f, 10**6)):
            entries = by_file[file_part]
            anchors = {s.partition("#")[2] for _, s in entries if "#" in s}
            buckets, warns = ep._paragraphs_by_anchor(file_part, anchors)

            for nav_path, src in entries:
                frag = src.partition("#")[2]
                title = _flatten_title(nav_path)
                if _is_non_content(title, nav_path):
                    continue
                paras = buckets.get(frag, []) if frag else buckets.get("", [])
                # Single-anchor files often put everything in the "" bucket.
                if not paras and frag and len(entries) == 1:
                    paras = buckets.get("", [])
                if not paras:
                    continue
                # A chapter's own heading is usually its first block; drop it so
                # the narrator does not read "12" before chapter 12.
                if paras and _clean_title(paras[0]).casefold() == \
                        _clean_title(nav_path[-1] if nav_path else "").casefold():
                    paras = paras[1:]
                if not paras:
                    continue
                idx += 1
                ch = Chapter(index=idx, title=title, nav_path=nav_path,
                             src=src, paragraphs=paras, warnings=warns)
                book.chapters.append(ch)

        # Spine files with prose that the TOC never mentioned.
        covered = set(by_file)
        for href in spine:
            if href in covered:
                continue
            buckets, _ = ep._paragraphs_by_anchor(href, set())
            paras = buckets.get("", [])
            if sum(len(p) for p in paras) > 500:
                book.warnings.append(f"{href}: prose not referenced by the TOC, appended")
                idx += 1
                book.chapters.append(
                    Chapter(index=idx, title=Path(href).stem, nav_path=[Path(href).stem],
                            src=href, paragraphs=paras))

        return book


def extract_cover(path: str | Path, book: Book, dest_dir: Path) -> Path | None:
    if not book.cover_path:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    with Epub(Path(path)) as ep:
        try:
            data = ep.zf.read(book.cover_path)
        except KeyError:
            return None
    out = dest_dir / f"cover{Path(book.cover_path).suffix or '.jpg'}"
    out.write_bytes(data)
    return out


def to_json(book: Book) -> str:
    d = asdict(book)
    d["char_count"] = book.char_count
    for c, cd in zip(book.chapters, d["chapters"]):
        cd["char_count"] = c.char_count
    return json.dumps(d, ensure_ascii=False, indent=2)
