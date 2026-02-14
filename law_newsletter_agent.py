"""
Labor Law Weekly Newsletter Agent
Scrapes legal journals, RSS feeds, and Google Scholar for recent articles.
Supports feedback via GitHub Issues for rejecting irrelevant papers.
Uses title + abstract for filtering and query refinement.
"""

import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
import sys
import json
import re
import unicodedata
from xml.etree import ElementTree as ET
from urllib.parse import quote as url_quote
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model once at startup
print("  Loading embedding model (all-MiniLM-L6-v2)...")
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("  Model loaded.")

SIMILARITY_THRESHOLD = 0.82
STAR_BOOST_THRESHOLD = 0.65  # Papers this similar to starred ones get priority
MIN_KEYWORD_FREQ = 2  # A word must appear in at least this many rejected titles to become a negative keyword

# ============== CONFIGURATION ==============
EMAIL_CONFIG = {
    "sender": os.getenv("EMAIL_SENDER", "your_email@libero.it"),
    "password": os.getenv("EMAIL_PASSWORD", "your_password"),
    "recipient": os.getenv("EMAIL_RECIPIENT", "your_email@libero.it"),
    "smtp_server": "smtp.libero.it",
    "smtp_port": 465,
    "use_ssl": True
}

# GitHub repo where this script lives (for reject issues)
# Format: "username/repo-name"
GITHUB_REPO = os.getenv("GITHUB_REPO", "your-username/law_newsletter")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # optional, needed for private repos

# ============== SOURCES REGISTRY ==============
# Each source defines how to scrape it: rss, ojs, sage, oup, wordpress, or scholar_fallback.
# Sources that cannot be reliably scraped use Google Scholar as fallback.

SOURCES = {
    # --- Italian Labor Law Journals ---
    "RGL": {
        "name": "Riv. Giuridica del Lavoro",
        "url": "https://www.futura-editrice.it/archivio-rgl/",
        "type": "scholar_fallback",
        "scholar_query": '"Rivista Giuridica del Lavoro" diritto lavoro',
    },
    "RIDL": {
        "name": "Riv. Italiana di Diritto del Lavoro",
        "url": "http://www.soluzionilavoro.it/category/osservatorio-riviste/rivista-italiana-di-diritto-del-lavoro/",
        "type": "wordpress",
    },
    "GDLRI": {
        "name": "Giornale Dir. Lavoro e Rel. Ind.",
        "url": "https://www.francoangeli.it/riviste/sommario/19/giornale-di-diritto-del-lavoro-e-di-relazioni-industriali",
        "type": "scholar_fallback",
        "scholar_query": '"Giornale di diritto del lavoro" relazioni industriali',
    },
    "LD": {
        "name": "Lavoro e Diritto",
        "url": "https://www.rivisteweb.it/issn/1120-947X/issues",
        "type": "rivisteweb",
        "issn": "1120-947X",
    },
    "DRI": {
        "name": "Diritto delle Relazioni Industriali",
        "url": "https://moodle.adaptland.it/course/view.php?id=21",
        "type": "scholar_fallback",
        "scholar_query": '"Diritto delle relazioni industriali" ADAPT lavoro',
    },
    "ADL": {
        "name": "Argomenti di Diritto del Lavoro",
        "url": "http://www.soluzionilavoro.it/category/osservatorio-riviste/argomenti-di-diritto-del-lavoro/",
        "type": "wordpress",
    },
    "LDE": {
        "name": "Lavoro Diritti Europa",
        "url": "https://www.lavorodirittieuropa.it/",
        "type": "wordpress",
    },
    "MGL": {
        "name": "Massimario Giur. del Lavoro",
        "url": "https://www.massimariogiurisprudenzadellavoro.it/HomePage",
        "type": "scholar_fallback",
        "scholar_query": '"Massimario di giurisprudenza del lavoro"',
    },
    "DLM": {
        "name": "Diritti Lavori Mercati",
        "url": "https://www.ddllmm.eu/dlm-fascicoli/",
        "type": "scholar_fallback",
        "scholar_query": '"Diritti lavori mercati" diritto lavoro',
    },
    "DSL": {
        "name": "Diritto della Sicurezza sul Lavoro",
        "url": "https://journals.uniurb.it/index.php/dsl",
        "type": "ojs",
        "feed_url": "https://journals.uniurb.it/index.php/dsl/gateway/plugin/WebFeedGatewayPlugin/atom",
    },
    "VTDL": {
        "name": "Variazioni su Temi Dir. Lavoro",
        "url": "https://www.dirittolavorovariazioni.com/HomePage",
        "type": "scholar_fallback",
        "scholar_query": '"Variazioni su temi di diritto del lavoro"',
    },
    "ILLEJ": {
        "name": "Italian Labour Law E-Journal",
        "url": "https://illej.unibo.it/issue/archive",
        "type": "ojs",
        "feed_url": "https://illej.unibo.it/gateway/plugin/WebFeedGatewayPlugin/atom",
    },
    "LLI": {
        "name": "Labour & Law Issues",
        "url": "https://labourlaw.unibo.it/issue/archive",
        "type": "ojs",
        "feed_url": "https://labourlaw.unibo.it/gateway/plugin/WebFeedGatewayPlugin/atom",
    },
    "LNG": {
        "name": "Il Lavoro nella Giurisprudenza",
        "url": "http://www.soluzionilavoro.it/category/osservatorio-riviste/il-lavoro-nella-giurisprudenza/",
        "type": "wordpress",
    },
    "RDSS": {
        "name": "Riv. Diritto Sicurezza Sociale",
        "url": "https://www.rivisteweb.it/issn/1720-562X/issues",
        "type": "rivisteweb",
        "issn": "1720-562X",
    },
    "PD": {
        "name": "Politica del Diritto",
        "url": "https://www.rivisteweb.it/issn/0032-3063/issues",
        "type": "rivisteweb",
        "issn": "0032-3063",
    },

    # --- European & Comparative ---
    "EULL": {
        "name": "EU Law Live",
        "url": "https://eulawlive.com/",
        "type": "wordpress",
    },
    "EID": {
        "name": "Economic and Industrial Democracy",
        "url": "https://journals.sagepub.com/home/eid",
        "type": "sage",
        "journal_code": "eid",
    },
    "BCLR": {
        "name": "Bulletin Comparative Labour Rel.",
        "url": "https://www.beck-shop.de/reihen/bulletin-of-comparative-labour-relations/11876",
        "type": "scholar_fallback",
        "scholar_query": '"Bulletin of Comparative Labour Relations" labor',
    },
    "EJIR": {
        "name": "European J. Industrial Relations",
        "url": "https://journals.sagepub.com/home/ejd",
        "type": "sage",
        "journal_code": "ejd",
    },
    "ELLJ": {
        "name": "European Labour Law Journal",
        "url": "https://journals.sagepub.com/home/ell",
        "type": "sage",
        "journal_code": "ell",
    },
    "ILJ": {
        "name": "Industrial Law Journal",
        "url": "https://academic.oup.com/ilj",
        "type": "oup",
        "journal_code": "ilj",
    },
    "IJCL": {
        "name": "Int. J. Comparative Labour Law",
        "url": "https://kluwerlawonline.com/Journals/International+Journal+of+Comparative+Labour+Law+and+Industrial+Relations/672",
        "type": "scholar_fallback",
        "scholar_query": '"International Journal of Comparative Labour Law" industrial relations',
    },
    "DS": {
        "name": "Droit Social",
        "url": "https://www.dalloz-revues.fr/revues/droit_social-297.htm",
        "type": "scholar_fallback",
        "scholar_query": '"Droit Social" travail droit',
    },
    "RDT": {
        "name": "Revue de Droit du Travail",
        "url": "https://www.dalloz-revues.fr/revues/revue_de_droit_du_travail-35.htm",
        "type": "scholar_fallback",
        "scholar_query": '"Revue de Droit du Travail"',
    },
    "RDCTSS": {
        "name": "Revue de Droit Comparé du Travail",
        "url": "https://journals.openedition.org/rdctss/",
        "type": "openedition",
        "feed_url": "https://journals.openedition.org/rdctss/backend?format=rssdocuments",
    },
}

# Categories with associated source keys and Google Scholar queries
CATEGORIES = {
    "Diritto del Lavoro": {
        "description": "Dottrina e legislazione italiana sul rapporto di lavoro",
        "sources": ["RGL", "RIDL", "ADL", "VTDL", "ILLEJ", "LLI", "LNG"],
        "queries": [
            "diritto del lavoro subordinato contratto",
            "licenziamento giusta causa giurisprudenza",
            "Jobs Act riforma lavoro",
            "contratti collettivi nazionali CCNL",
            "lavoro agile smart working normativa",
            "somministrazione lavoro appalto",
        ]
    },
    "Relazioni Industriali": {
        "description": "Diritto sindacale, contrattazione collettiva, relazioni industriali",
        "sources": ["GDLRI", "LD", "DRI", "DLM", "PD"],
        "queries": [
            "relazioni industriali contrattazione collettiva",
            "diritto sindacale rappresentanza",
            "sciopero diritto astensione collettiva",
            "partecipazione lavoratori impresa",
            "dialogo sociale europeo",
        ]
    },
    "Diritto Europeo e Comparato del Lavoro": {
        "description": "Diritto del lavoro UE, comparato e internazionale",
        "sources": ["EULL", "EID", "EJIR", "ELLJ", "ILJ", "IJCL", "BCLR", "DS", "RDT", "RDCTSS"],
        "queries": [
            "European labour law directive workers",
            "platform work EU regulation gig economy",
            "EU social policy employment directive",
            "comparative labor law dismissal protection",
            "collective bargaining European framework",
            "droit du travail européen directive",
        ]
    },
    "Sicurezza sul Lavoro e Previdenza": {
        "description": "Sicurezza, salute sul lavoro, previdenza e assistenza sociale",
        "sources": ["DSL", "LDE", "MGL", "RDSS"],
        "queries": [
            "sicurezza lavoro prevenzione infortuni",
            "malattia professionale risarcimento",
            "previdenza sociale pensione riforma",
            "welfare aziendale previdenza complementare",
            "salute sicurezza cantiere lavoratori",
            "INAIL infortunio responsabilità datore",
        ]
    },
    "Giurisprudenza del Lavoro": {
        "description": "Sentenze, massime e commenti giurisprudenziali",
        "sources": ["MGL", "LNG", "RIDL"],
        "queries": [
            "Cassazione lavoro sentenza recente",
            "Corte Costituzionale diritto lavoro",
            "giurisprudenza licenziamento reintegrazione",
            "Corte di Giustizia UE lavoro",
            "tribunale lavoro controversia",
        ]
    },
}

MAX_PAPERS_PER_CATEGORY = 16
MAX_PREPRINTS_PER_CATEGORY = 3  # Not used for law, kept for compatibility
MAX_DISCOVERY_PAPERS = 3        # Papers in the Weekly Discovery section

# ============== SERENDIPITY DISCOVERY ==============
# Maps core interest terms → adjacent/emerging fields to explore.

ADJACENCY_MAP = {
    # Labor law adjacent
    "licenziamento": [
        "(unfair dismissal) AND (comparative law OR European directive)",
        "(redundancy OR restructuring) AND (worker protection OR social plan)",
        "(whistleblower protection) AND (dismissal OR retaliation)",
    ],
    "lavoro subordinato": [
        "(gig economy OR platform work) AND (employment status OR worker classification)",
        "(algorithm management) AND (worker rights OR labor law)",
        "(remote work OR telework) AND (right to disconnect OR regulation)",
        "(artificial intelligence) AND (employment OR labor market impact)",
    ],
    "contrattazione collettiva": [
        "(sectoral bargaining) AND (minimum wage OR wage setting)",
        "(trade union density) AND (Europe OR decline OR renewal)",
        "(social dialogue) AND (crisis OR pandemic response)",
        "(transnational collective bargaining) AND (multinational OR European works council)",
    ],
    "sicurezza lavoro": [
        "(psychosocial risks) AND (workplace OR occupational health)",
        "(occupational safety) AND (artificial intelligence OR automation)",
        "(work-related stress) AND (burnout OR regulation)",
        "(climate change) AND (occupational health OR heat stress workers)",
    ],
    "previdenza sociale": [
        "(universal basic income) AND (welfare OR social protection)",
        "(pension reform) AND (sustainability OR demographic change)",
        "(social protection) AND (platform workers OR non-standard employment)",
        "(long-term care) AND (aging population OR social security)",
    ],
    "diritto europeo lavoro": [
        "(EU minimum wage directive) AND (implementation OR impact)",
        "(posted workers directive) AND (enforcement OR social dumping)",
        "(European Pillar Social Rights) AND (implementation OR progress)",
        "(due diligence) AND (supply chain OR corporate sustainability directive)",
    ],
    "smart working": [
        "(hybrid work) AND (labor regulation OR legal framework)",
        "(digital nomad) AND (employment law OR tax implications)",
        "(workplace surveillance) AND (remote work OR privacy rights)",
        "(coworking space) AND (employment relationship OR occupational safety)",
    ],
    "discriminazione lavoro": [
        "(pay transparency) AND (gender gap OR EU directive)",
        "(algorithmic discrimination) AND (hiring OR employment)",
        "(disability) AND (reasonable accommodation OR workplace inclusion)",
        "(age discrimination) AND (employment OR forced retirement)",
    ],
    "appalto": [
        "(subcontracting) AND (joint liability OR worker protection)",
        "(public procurement) AND (social clause OR labor standards)",
        "(supply chain) AND (forced labor OR human rights due diligence)",
    ],
    "diritto sindacale": [
        "(union organizing) AND (digital platform OR tech workers)",
        "(right to strike) AND (essential services OR comparative law)",
        "(works council) AND (co-determination OR employee participation)",
    ],
}

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.json")


# ============== FEEDBACK SYSTEM ==============

# Separator used in GitHub Issue body between title and abstract
_ISSUE_SEPARATOR = "\n---\n"


def _load_feedback_data():
    """Load the entire feedback.json file.

    Returns dict with keys: rejected_titles, starred_papers, sent_history, explored_paths.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return {"rejected_titles": [], "starred_papers": [], "sent_history": [], "explored_paths": []}
    try:
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)
        return {
            "rejected_titles": data.get("rejected_titles", []),
            "starred_papers": data.get("starred_papers", []),
            "sent_history": data.get("sent_history", []),
            "explored_paths": data.get("explored_paths", []),
        }
    except json.JSONDecodeError as e:
        print("\n" + "!" * 60)
        print("  ERROR: feedback.json is not valid JSON!")
        print(f"  Details: {e}")
        print("!" * 60 + "\n")
        return {"rejected_titles": [], "starred_papers": [], "sent_history": [], "explored_paths": []}
    except IOError:
        return {"rejected_titles": [], "starred_papers": [], "sent_history": [], "explored_paths": []}


def _save_feedback_data(data):
    """Save the entire feedback.json file."""
    try:
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"  Error saving feedback.json: {e}")


def _parse_paper_list(raw):
    """Parse a list of papers from feedback.json (handles old and new formats)."""
    papers = []
    for item in raw:
        if isinstance(item, dict):
            papers.append({"title": item.get("title", ""), "abstract": item.get("abstract", "")})
        else:
            papers.append({"title": str(item), "abstract": ""})
    return papers


def load_feedback_file():
    """Load rejected papers from feedback.json."""
    data = _load_feedback_data()
    return _parse_paper_list(data["rejected_titles"])


def load_starred_papers():
    """Load starred (positive feedback) papers from feedback.json."""
    data = _load_feedback_data()
    return _parse_paper_list(data["starred_papers"])


def load_sent_history():
    """Load set of normalized titles of all previously sent papers."""
    data = _load_feedback_data()
    return set(data["sent_history"])


def save_feedback_file(rejected):
    """Save rejected papers list to feedback.json (preserves other keys)."""
    data = _load_feedback_data()
    data["rejected_titles"] = rejected
    _save_feedback_data(data)
    print(f"  Saved {len(rejected)} rejected papers to feedback.json")


def save_starred_papers(starred):
    """Save starred papers list to feedback.json (preserves other keys)."""
    data = _load_feedback_data()
    data["starred_papers"] = starred
    _save_feedback_data(data)
    print(f"  Saved {len(starred)} starred papers to feedback.json")


def save_sent_history(sent_titles):
    """Save sent paper titles to feedback.json (preserves other keys)."""
    data = _load_feedback_data()
    data["sent_history"] = list(sent_titles)
    _save_feedback_data(data)
    print(f"  Saved {len(sent_titles)} sent titles to history")


def _parse_issue_body(body):
    """Parse a GitHub Issue body into title and abstract."""
    body = body.strip()
    if _ISSUE_SEPARATOR in body:
        parts = body.split(_ISSUE_SEPARATOR, 1)
        return {"title": parts[0].strip(), "abstract": parts[1].strip()}
    else:
        return {"title": body, "abstract": ""}


def _close_github_issue(issue_number):
    """Close a GitHub Issue by number."""
    if not GITHUB_TOKEN:
        print(f"    Cannot close issue #{issue_number}: GITHUB_TOKEN not set")
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{issue_number}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}"
    }

    try:
        resp = requests.patch(url, headers=headers, json={"state": "closed"}, timeout=15)
        if resp.status_code == 200:
            return True
        else:
            print(f"    Failed to close issue #{issue_number}: {resp.status_code}")
            return False
    except Exception as e:
        print(f"    Error closing issue #{issue_number}: {e}")
        return False


def sync_and_cleanup():
    """Sync open GitHub Issues into feedback.json, then close them.

    Handles both 'reject' and 'star' labeled issues.
    """
    if not GITHUB_REPO or GITHUB_REPO == "your-username/law_newsletter":
        print("  Skipping sync: GITHUB_REPO not configured")
        return

    print("\n  Syncing GitHub Issues → feedback.json...")

    # Load existing data
    existing_rejected = load_feedback_file()
    existing_starred = load_starred_papers()
    rejected_titles = {_normalize_title(p["title"]) for p in existing_rejected}
    starred_titles = {_normalize_title(p["title"]) for p in existing_starred}

    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    # Process both reject and star labels
    for label, existing_list, existing_set, label_name in [
        ("reject", existing_rejected, rejected_titles, "rejected"),
        ("star", existing_starred, starred_titles, "starred"),
    ]:
        params = {"labels": label, "state": "open", "per_page": 100}

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code != 200:
                print(f"  GitHub API error ({label}): {resp.status_code}")
                continue
            issues = resp.json()
        except Exception as e:
            print(f"  GitHub sync error ({label}): {e}")
            continue

        if not issues:
            print(f"  No open {label} issues to sync")
            continue

        new_count = 0
        closed_count = 0

        for issue in issues:
            body = issue.get("body", "")
            if not body:
                continue

            paper = _parse_issue_body(body)
            norm = _normalize_title(paper["title"])

            if norm not in existing_set:
                existing_list.append(paper)
                existing_set.add(norm)
                new_count += 1

            if _close_github_issue(issue["number"]):
                closed_count += 1

        print(f"  {label_name.capitalize()}: {new_count} new, {closed_count} issues closed")

    # Save both lists
    save_feedback_file(existing_rejected)
    save_starred_papers(existing_starred)


def load_all_rejected():
    """Load rejected papers from feedback.json (primary source).

    Returns list of dicts: [{"title": ..., "abstract": ...}, ...]
    """
    papers = load_feedback_file()
    if papers:
        print(f"  Loaded {len(papers)} rejected papers from feedback.json")
    return papers


def _normalize_title(title):
    """Normalize a title for robust comparison."""
    t = unicodedata.normalize("NFKC", title)
    t = t.lower().strip()
    t = t.rstrip(".")
    t = re.sub(r"\s+", " ", t)
    return t


def _summarize_abstract(abstract, max_sentences=3):
    """Extract the first 2-3 sentences from an abstract as a brief summary.

    Uses a regex-based sentence splitter that handles common abbreviations.
    """
    if not abstract or not abstract.strip():
        return ""

    text = abstract.strip()
    abbrevs = r"(?<!\bet al)(?<!\bvs)(?<!\bDr)(?<!\bFig)(?<!\bNo)(?<!\bVol)(?<!\bEq)(?<!\bart)(?<!\bco)(?<!\bn)"
    sentences = re.split(rf'{abbrevs}(?<=[.!?])\s+(?=[A-Z])', text)

    selected = sentences[:max(2, min(max_sentences, len(sentences)))]
    summary = " ".join(s.strip() for s in selected if s.strip())

    if len(summary) > 400:
        summary = summary[:397].rsplit(" ", 1)[0] + "..."

    return summary


# Cache for rejected paper embeddings (computed once per run)
_rejected_cache = {"papers": None, "embeddings": None, "texts": None}


def _get_rejected_embeddings(rejected_papers):
    """Compute and cache embeddings for rejected papers (title + abstract)."""
    if _rejected_cache["papers"] is not rejected_papers:
        _rejected_cache["papers"] = rejected_papers
        if rejected_papers:
            texts = [f"{p['title']} {p['abstract']}".strip() for p in rejected_papers]
            _rejected_cache["texts"] = texts
            _rejected_cache["embeddings"] = _EMBED_MODEL.encode(
                texts, normalize_embeddings=True
            )
        else:
            _rejected_cache["texts"] = None
            _rejected_cache["embeddings"] = None
    return _rejected_cache["embeddings"]


def calculate_similarity(new_text, rejected_papers):
    """Check if new_text is semantically similar to any rejected paper using embeddings."""
    if not rejected_papers:
        return False

    rejected_embeddings = _get_rejected_embeddings(rejected_papers)
    if rejected_embeddings is None:
        return False

    new_embedding = _EMBED_MODEL.encode([new_text], normalize_embeddings=True)

    scores = np.dot(rejected_embeddings, new_embedding.T).flatten()
    max_idx = int(np.argmax(scores))
    max_score = float(scores[max_idx])

    if max_score >= SIMILARITY_THRESHOLD:
        rejected_title = rejected_papers[max_idx]["title"]
        print(f"    [Semantic Similarity] Rejected: '{new_text[:80]}...' "
              f"(score={max_score:.3f} vs '{rejected_title}')")
        return True
    return False


def is_rejected(title, abstract, rejected_papers):
    """Check if a paper should be rejected (normalized match or semantic similarity)."""
    norm_title = _normalize_title(title)
    rejected_titles_set = {_normalize_title(p["title"]) for p in rejected_papers}
    if norm_title in rejected_titles_set:
        return True
    combined = f"{title} {abstract}".strip()
    return calculate_similarity(combined, rejected_papers)


# ============== POSITIVE FEEDBACK (STAR SCORING) ==============

_starred_cache = {"papers": None, "embeddings": None}


def _get_starred_embeddings(starred_papers):
    """Compute and cache embeddings for starred papers."""
    if _starred_cache["papers"] is not starred_papers:
        _starred_cache["papers"] = starred_papers
        if starred_papers:
            texts = [f"{p['title']} {p['abstract']}".strip() for p in starred_papers]
            _starred_cache["embeddings"] = _EMBED_MODEL.encode(
                texts, normalize_embeddings=True
            )
        else:
            _starred_cache["embeddings"] = None
    return _starred_cache["embeddings"]


def calculate_star_score(title, abstract, starred_papers):
    """Calculate how similar a paper is to starred (liked) papers."""
    if not starred_papers:
        return 0.0

    starred_embeddings = _get_starred_embeddings(starred_papers)
    if starred_embeddings is None:
        return 0.0

    combined = f"{title} {abstract}".strip()
    new_embedding = _EMBED_MODEL.encode([combined], normalize_embeddings=True)

    scores = np.dot(starred_embeddings, new_embedding.T).flatten()
    max_score = float(np.max(scores))
    return max_score


# ============== DYNAMIC QUERY REFINEMENT ==============

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "not", "no", "nor", "so",
    "as", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "too", "very", "just", "about",
    "above", "after", "again", "also", "any", "before", "between", "both",
    "during", "here", "how", "into", "new", "now", "over", "through",
    "under", "up", "out", "what", "when", "where", "which", "while", "who",
    "why", "using", "based", "via", "among", "across", "within", "without",
    # Domain-generic words that would over-filter law results
    "diritto", "legge", "norma", "articolo", "comma", "sentenza", "nota",
    "commento", "osservazioni", "profili", "aspetti", "questioni", "tema",
    "recente", "brevi", "sulla", "sulle", "delle", "della", "dello", "degli",
    "nella", "nelle", "lavoro", "lavoratore", "lavoratori", "datore",
    "rapporto", "contratto", "tribunale", "corte", "cassazione",
    "law", "labour", "labor", "work", "worker", "workers", "employment",
    "study", "analysis", "review", "case", "article", "note", "comment",
}


def get_negative_keywords(rejected_papers):
    """Extract frequent significant words from rejected paper abstracts."""
    from collections import Counter

    if not rejected_papers:
        return []

    word_counts = Counter()
    for paper in rejected_papers:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        words = set(re.findall(r"[a-zàèéìòù0-9]{3,}", text.lower()))
        words -= STOP_WORDS
        word_counts.update(words)

    keywords = [word for word, count in word_counts.most_common()
                if count >= MIN_KEYWORD_FREQ]

    if keywords:
        print(f"  Negative keywords (freq >= {MIN_KEYWORD_FREQ}): {keywords[:15]}...")

    return keywords


def refine_query(query, negative_keywords, max_negatives=3):
    """Append NOT terms to a query based on negative keywords."""
    if not negative_keywords:
        return query

    query_lower = query.lower()
    not_terms = []
    for kw in negative_keywords:
        if kw not in query_lower:
            not_terms.append(kw)
        if len(not_terms) >= max_negatives:
            break

    if not not_terms:
        return query

    not_clause = " ".join(f"-{term}" for term in not_terms)
    return f"{query} {not_clause}"


# ============== SERENDIPITY DISCOVERY ENGINE ==============

def _extract_interest_profile():
    """Build a set of core interest terms from starred papers and category queries."""
    terms = set()

    for config in CATEGORIES.values():
        for query in config["queries"]:
            words = re.findall(r"[a-zàèéìòù][a-zàèéìòù\s-]+", query.lower())
            for phrase in words:
                phrase = phrase.strip()
                if phrase and phrase not in {"or", "and", "not"}:
                    terms.add(phrase)

    starred = load_starred_papers()
    for paper in starred:
        text = f"{paper['title']} {paper['abstract']}"
        words = set(re.findall(r"[a-zàèéìòù]{4,}", text.lower()))
        words -= STOP_WORDS
        terms.update(words)

    return terms


def _load_explored_paths():
    """Load the list of previously explored adjacency queries."""
    data = _load_feedback_data()
    return set(data.get("explored_paths", []))


def _save_explored_path(query):
    """Record that an adjacency query has been explored."""
    data = _load_feedback_data()
    paths = set(data.get("explored_paths", []))
    paths.add(query)
    data["explored_paths"] = list(paths)
    _save_feedback_data(data)


def generate_serendipity_queries():
    """Pick 2-3 adjacent-field queries based on the user's interest profile."""
    import random

    interest_terms = _extract_interest_profile()
    explored = _load_explored_paths()
    rejected_papers = load_feedback_file()

    rejected_terms = set()
    for paper in rejected_papers:
        words = set(re.findall(r"[a-zàèéìòù]{4,}", paper.get("title", "").lower()))
        words -= STOP_WORDS
        rejected_terms.update(words)

    matched_keys = []
    for key in ADJACENCY_MAP:
        key_lower = key.lower()
        if any(key_lower in term or term in key_lower for term in interest_terms):
            matched_keys.append(key)

    if not matched_keys:
        matched_keys = list(ADJACENCY_MAP.keys())

    random.shuffle(matched_keys)

    selected_queries = []
    for key in matched_keys:
        if len(selected_queries) >= 3:
            break

        candidates = ADJACENCY_MAP[key]
        random.shuffle(candidates)

        for query in candidates:
            if query in explored:
                continue

            query_words = set(re.findall(r"[a-zàèéìòù]{4,}", query.lower())) - STOP_WORDS
            overlap = query_words & rejected_terms
            if len(overlap) > 2:
                continue

            selected_queries.append(query)
            break

    if not selected_queries:
        print("  Discovery: All adjacent paths explored! Resetting exploration history.")
        data = _load_feedback_data()
        data["explored_paths"] = []
        _save_feedback_data(data)
        return generate_serendipity_queries()

    return selected_queries


def collect_discovery_papers(rejected_papers, sent_history):
    """Search for papers in adjacent fields using serendipity queries."""
    print("\n  ✨ Weekly Discovery (Serendipity)...")

    queries = generate_serendipity_queries()
    if not queries:
        print("  No discovery queries generated")
        return {"description": "Esplorando frontiere giuridiche adiacenti", "papers": []}

    papers = []
    seen_titles = set()

    for query in queries:
        print(f"  Discovery query: {query[:70]}...")

        # Use Google Scholar for discovery (broader search)
        for paper in search_google_scholar(query, max_results=5):
            title_lower = paper["title"].lower()
            norm = _normalize_title(paper["title"])

            if title_lower in seen_titles:
                continue
            if norm in sent_history:
                continue
            if is_rejected(paper["title"], paper.get("abstract", ""), rejected_papers):
                continue

            papers.append(paper)
            seen_titles.add(title_lower)

            if len(papers) >= MAX_DISCOVERY_PAPERS:
                break

        _save_explored_path(query)

        if len(papers) >= MAX_DISCOVERY_PAPERS:
            break

    print(f"  Discovery found: {len(papers)} papers from {len(queries)} queries")

    return {
        "description": "Esplorando frontiere giuridiche adiacenti",
        "papers": papers[:MAX_DISCOVERY_PAPERS]
    }


# ============== DATA SOURCES ==============

_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LawNewsletter/1.0; +https://github.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
}


def scrape_rss_feed(feed_url, source_name, max_results=10):
    """Parse an RSS/Atom feed and return recent articles."""
    import feedparser

    try:
        feed = feedparser.parse(feed_url)
        papers = []

        cutoff = datetime.now() - timedelta(days=30)  # Legal journals publish less frequently

        for entry in feed.entries[:max_results * 2]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "")
            summary = entry.get("summary", entry.get("description", ""))

            if not title or not link:
                continue

            # Clean HTML from summary
            if summary:
                summary = re.sub(r"<[^>]+>", "", summary).strip()
                summary = re.sub(r"\s+", " ", summary)

            # Check date if available
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                try:
                    pub_date = datetime(*published[:6])
                    if pub_date < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass

            papers.append({
                "title": title,
                "url": link,
                "source": source_name,
                "abstract": summary[:1000] if summary else ""
            })

            if len(papers) >= max_results:
                break

        return papers
    except Exception as e:
        print(f"  RSS error ({source_name}): {e}")
        return []


def scrape_sage_journal(journal_code, source_name, max_results=10):
    """Scrape recent articles from a SAGE journal via RSS."""
    feed_url = f"https://journals.sagepub.com/action/showFeed?ui=0&mi=ehikzz&ai=2b4&jc={journal_code}&type=etoc&feed=rss"
    return scrape_rss_feed(feed_url, source_name, max_results)


def scrape_oup_journal(journal_code, source_name, max_results=10):
    """Scrape recent articles from an Oxford University Press journal."""
    from bs4 import BeautifulSoup

    url = f"https://academic.oup.com/{journal_code}/advance-articles"
    try:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=20)
        if resp.status_code != 200:
            # Fallback to RSS
            rss_url = f"https://academic.oup.com/rss/site_6258/advanceAccess.xml"
            return scrape_rss_feed(rss_url, source_name, max_results)

        soup = BeautifulSoup(resp.text, "html.parser")
        papers = []

        for article in soup.select(".al-article-item, .article-list-item, .customLink"):
            title_el = article.select_one("h5 a, .article-title a, a.article-link")
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            if href and not href.startswith("http"):
                href = f"https://academic.oup.com{href}"

            abstract = ""
            abs_el = article.select_one(".abstract, .snippet")
            if abs_el:
                abstract = abs_el.get_text(strip=True)

            if title:
                papers.append({
                    "title": title,
                    "url": href,
                    "source": source_name,
                    "abstract": abstract
                })

            if len(papers) >= max_results:
                break

        return papers
    except Exception as e:
        print(f"  OUP error ({source_name}): {e}")
        return []


def scrape_wordpress(url, source_name, max_results=10):
    """Scrape articles from a WordPress site (tries REST API first, then HTML)."""
    from bs4 import BeautifulSoup

    # Try WordPress REST API first
    api_base = url.rstrip("/").split("/category/")[0] if "/category/" in url else url.rstrip("/")
    api_url = f"{api_base}/wp-json/wp/v2/posts?per_page={max_results}&orderby=date"

    try:
        resp = requests.get(api_url, headers=_REQUEST_HEADERS, timeout=15)
        if resp.status_code == 200:
            posts = resp.json()
            papers = []
            for post in posts[:max_results]:
                title = re.sub(r"<[^>]+>", "", post.get("title", {}).get("rendered", "")).strip()
                link = post.get("link", "")
                excerpt = re.sub(r"<[^>]+>", "", post.get("excerpt", {}).get("rendered", "")).strip()

                if title:
                    papers.append({
                        "title": title,
                        "url": link,
                        "source": source_name,
                        "abstract": excerpt[:1000] if excerpt else ""
                    })
            if papers:
                return papers
    except Exception:
        pass  # Fall through to HTML scraping

    # Fallback: HTML scraping
    try:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=15)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        papers = []

        # Common WordPress selectors
        selectors = [
            "article", ".post", ".entry", ".hentry",
            ".type-post", ".blog-post", ".td-module-container"
        ]

        articles = []
        for sel in selectors:
            articles = soup.select(sel)
            if articles:
                break

        if not articles:
            # Last resort: look for h2/h3 links
            articles = soup.select("h2 a, h3 a, .entry-title a")
            for a_tag in articles[:max_results]:
                title = a_tag.get_text(strip=True)
                href = a_tag.get("href", "")
                if title:
                    papers.append({
                        "title": title,
                        "url": href,
                        "source": source_name,
                        "abstract": ""
                    })
            return papers

        for article in articles[:max_results]:
            title_el = article.select_one(
                ".entry-title a, h2 a, h3 a, .post-title a, .td-module-title a"
            )
            if not title_el:
                title_el = article.select_one("h2, h3, .entry-title, .post-title")

            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            href = title_el.get("href", "") if title_el.name == "a" else ""
            if not href:
                a_tag = title_el.find("a")
                if a_tag:
                    href = a_tag.get("href", "")

            excerpt = ""
            exc_el = article.select_one(
                ".entry-summary, .entry-content, .excerpt, .post-excerpt, .td-excerpt"
            )
            if exc_el:
                excerpt = exc_el.get_text(strip=True)[:500]

            if title:
                papers.append({
                    "title": title,
                    "url": href,
                    "source": source_name,
                    "abstract": excerpt
                })

        return papers
    except Exception as e:
        print(f"  WordPress error ({source_name}): {e}")
        return []


def scrape_rivisteweb(issn, source_name, max_results=10):
    """Scrape recent articles from rivisteweb.it journal issues page."""
    from bs4 import BeautifulSoup

    url = f"https://www.rivisteweb.it/issn/{issn}/issues"
    try:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=15)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        papers = []

        # Find latest issue link
        issue_links = soup.select("a[href*='/doi/'], a[href*='/issue/']")
        if not issue_links:
            # Try direct article links
            issue_links = soup.select("a")

        # Look for article titles on the page
        for link in soup.select("a"):
            href = link.get("href", "")
            text = link.get_text(strip=True)

            # Skip navigation/menu links, look for article-like titles
            if len(text) > 20 and ("/doi/" in href or "/article/" in href):
                full_url = href if href.startswith("http") else f"https://www.rivisteweb.it{href}"
                papers.append({
                    "title": text,
                    "url": full_url,
                    "source": source_name,
                    "abstract": ""
                })

            if len(papers) >= max_results:
                break

        return papers
    except Exception as e:
        print(f"  Rivisteweb error ({source_name}): {e}")
        return []


def scrape_openedition(feed_url, source_name, max_results=10):
    """Scrape recent articles from an OpenEdition journal via RSS."""
    return scrape_rss_feed(feed_url, source_name, max_results)


def search_google_scholar(query, max_results=5):
    """Search Google Scholar via SerpAPI (optional - set SERPAPI_KEY)."""
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return []

    url = "https://serpapi.com/search"
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key,
        "num": max_results,
        "as_ylo": datetime.now().year,
        "scisbd": 1
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        papers = []
        for result in data.get("organic_results", []):
            papers.append({
                "title": result.get("title", "No title"),
                "url": result.get("link", ""),
                "source": "Scholar",
                "abstract": result.get("snippet", "")
            })
        return papers
    except Exception as e:
        print(f"  Scholar error: {e}")
        return []


def scrape_source(source_key):
    """Dispatch to the appropriate scraper based on source type."""
    if source_key not in SOURCES:
        return []

    src = SOURCES[source_key]
    src_type = src["type"]
    src_name = src["name"]

    try:
        if src_type == "sage":
            papers = scrape_sage_journal(src["journal_code"], src_name)
        elif src_type == "oup":
            papers = scrape_oup_journal(src["journal_code"], src_name)
        elif src_type == "ojs":
            papers = scrape_rss_feed(src["feed_url"], src_name)
        elif src_type == "openedition":
            papers = scrape_openedition(src["feed_url"], src_name)
        elif src_type == "wordpress":
            papers = scrape_wordpress(src["url"], src_name)
        elif src_type == "rivisteweb":
            papers = scrape_rivisteweb(src["issn"], src_name)
        elif src_type == "scholar_fallback":
            papers = search_google_scholar(src.get("scholar_query", src_name), max_results=5)
            # Override source name for scholar results
            for p in papers:
                p["source"] = src_name
        else:
            papers = []

        if not papers and src_type != "scholar_fallback":
            # Fallback to Scholar if primary scraper returns nothing
            scholar_query = src.get("scholar_query", f'"{src_name}" diritto lavoro')
            print(f"    Fallback to Scholar for {src_name}...")
            papers = search_google_scholar(scholar_query, max_results=3)
            for p in papers:
                p["source"] = src_name

        return papers
    except Exception as e:
        print(f"  Scraper error ({src_name}): {e}")
        return []


# ============== NEWSLETTER BUILDER ==============

def collect_papers():
    """Collect papers for all categories, filtering rejected and already-sent ones.

    Papers are scored by similarity to starred papers and sorted by relevance.
    """
    newsletter = {}
    rejected_papers = load_all_rejected()
    starred_papers = load_starred_papers()
    sent_history = load_sent_history()
    negative_kw = get_negative_keywords(rejected_papers)
    rejected_count = 0
    dedup_count = 0

    if starred_papers:
        print(f"  Loaded {len(starred_papers)} starred papers for boosting")
    if sent_history:
        print(f"  Loaded {len(sent_history)} previously sent titles for dedup")

    for category, config in CATEGORIES.items():
        print(f"\n  {category}...")
        papers = []
        seen_titles = set()
        category_sources = config.get("sources", [])

        # Step 1: Scrape each source associated with this category
        for source_key in category_sources:
            print(f"    Scraping: {SOURCES.get(source_key, {}).get('name', source_key)}...")
            for paper in scrape_source(source_key):
                title_lower = paper["title"].lower()
                norm = _normalize_title(paper["title"])
                if title_lower in seen_titles:
                    continue
                if norm in sent_history:
                    dedup_count += 1
                    continue
                if is_rejected(paper["title"], paper.get("abstract", ""), rejected_papers):
                    rejected_count += 1
                    continue
                papers.append(paper)
                seen_titles.add(title_lower)

        # Step 2: Google Scholar queries for this category
        for query in config["queries"]:
            refined = refine_query(query, negative_kw)
            if refined != query:
                print(f"    Scholar query (refined): {refined[:80]}...")
            else:
                print(f"    Scholar query: {query[:60]}...")

            for paper in search_google_scholar(refined, 3):
                title_lower = paper["title"].lower()
                norm = _normalize_title(paper["title"])
                if title_lower in seen_titles:
                    continue
                if norm in sent_history:
                    dedup_count += 1
                    continue
                if is_rejected(paper["title"], paper.get("abstract", ""), rejected_papers):
                    rejected_count += 1
                    continue
                papers.append(paper)
                seen_titles.add(title_lower)

        # Score papers by similarity to starred papers and sort
        if starred_papers and papers:
            for paper in papers:
                paper["_star_score"] = calculate_star_score(
                    paper["title"], paper.get("abstract", ""), starred_papers
                )
            papers.sort(key=lambda p: p["_star_score"], reverse=True)

            boosted = [p for p in papers if p["_star_score"] >= STAR_BOOST_THRESHOLD]
            if boosted:
                print(f"  ★ {len(boosted)} papers boosted by star similarity")

        newsletter[category] = {
            "description": config["description"],
            "papers": papers[:MAX_PAPERS_PER_CATEGORY]
        }
        print(f"  Found: {len(papers)} papers")

    if rejected_count:
        print(f"\n  Filtered out {rejected_count} rejected papers")
    if dedup_count:
        print(f"  Skipped {dedup_count} previously sent papers")

    # Add Weekly Discovery (serendipity) section
    discovery = collect_discovery_papers(rejected_papers, sent_history)
    if discovery["papers"]:
        newsletter["Scoperte della Settimana"] = discovery

    return newsletter


def format_newsletter(newsletter):
    """Format newsletter as HTML with Reject links via GitHub Issues."""
    date_str = datetime.now().strftime("%d %B %Y")

    css = """
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 750px; margin: auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #8e44ad; padding-bottom: 10px; }
            h2 { color: #6c3483; margin-top: 35px; padding: 10px; background: #f4ecf7; border-radius: 5px; }
            .section-desc { color: #7f8c8d; font-size: 13px; margin-top: -5px; margin-bottom: 15px; }
            .paper { margin: 15px 0; padding: 15px 18px; background: linear-gradient(to right, #f8f9fa, #ffffff); border-left: 4px solid #8e44ad; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
            .paper:hover { background: linear-gradient(to right, #f3eef7, #ffffff); }
            .paper a.title-link { color: #2c3e50; text-decoration: none; font-weight: 600; font-size: 15px; line-height: 1.4; display: block; }
            .paper a.title-link:hover { color: #6c3483; }
            .source { color: #7f8c8d; font-size: 12px; display: inline-block; margin-top: 8px; background: #f4ecf7; padding: 3px 10px; border-radius: 12px; }
            .reject-link { color: #e74c3c; font-size: 12px; text-decoration: none; margin-left: 10px; cursor: pointer; }
            .reject-link:hover { text-decoration: underline; }
            .star-link { color: #f39c12; font-size: 12px; text-decoration: none; margin-left: 6px; cursor: pointer; }
            .star-link:hover { text-decoration: underline; }
            .abstract { color: #555; font-size: 13px; line-height: 1.5; margin-top: 8px; padding: 8px 10px; background: #f9f9fb; border-radius: 4px; }
            .empty { color: #bdc3c7; font-style: italic; padding: 15px; }
            .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #ecf0f1; color: #95a5a6; font-size: 11px; }
        </style>
    """

    html = f"""
    <html>
    <head>{css}</head>
    <body>
        <div class="container">
        <h1>&#x2696; Diritto del Lavoro Weekly</h1>
        <p style="color: #7f8c8d;"><em>{date_str}</em></p>
    """

    icons = {
        "Diritto del Lavoro": "&#x1F4DC;",
        "Relazioni Industriali": "&#x1F91D;",
        "Diritto Europeo e Comparato del Lavoro": "&#x1F1EA;&#x1F1FA;",
        "Sicurezza sul Lavoro e Previdenza": "&#x1F6E1;",
        "Giurisprudenza del Lavoro": "&#x1F3DB;",
        "Scoperte della Settimana": "&#x2728;",
    }

    for category, data in newsletter.items():
        icon = icons.get(category, "&#x1F4C4;")
        count = len(data["papers"])
        html += f'<h2>{icon} {category} <span style="font-size:14px; color:#95a5a6; font-weight:normal;">({count} articoli)</span></h2>'
        html += f'<p class="section-desc">{data["description"]}</p>'

        if data["papers"]:
            for i, paper in enumerate(data["papers"], 1):
                # Build issue body (shared between reject and star)
                abstract = paper.get("abstract", "")
                if abstract:
                    issue_body_raw = f"{paper['title']}{_ISSUE_SEPARATOR}{abstract}"
                else:
                    issue_body_raw = paper["title"]
                if len(issue_body_raw) > 4000:
                    issue_body_raw = issue_body_raw[:4000] + "..."
                issue_body = url_quote(issue_body_raw)

                # [Reject] link
                reject_title = url_quote(f"Reject: {paper['title'][:80]}")
                reject_url = f"https://github.com/{GITHUB_REPO}/issues/new?labels=reject&title={reject_title}&body={issue_body}"

                # [Star] link
                star_title = url_quote(f"Star: {paper['title'][:80]}")
                star_url = f"https://github.com/{GITHUB_REPO}/issues/new?labels=star&title={star_title}&body={issue_body}"

                # Show ★ badge if paper was boosted by star similarity
                star_score = paper.get("_star_score", 0)
                star_badge = ""
                if star_score >= STAR_BOOST_THRESHOLD:
                    pct = int(star_score * 100)
                    star_badge = f' <span style="color:#f39c12; font-size:11px;" title="Simile ai tuoi articoli preferiti ({pct}% match)">&#x2B50;</span>'

                # Abstract summary (2-3 sentences)
                summary = _summarize_abstract(paper.get("abstract", ""))
                summary_html = f'<p class="abstract">{summary}</p>' if summary else ""

                html += f"""
                <div class="paper">
                    <span style="color:#8e44ad; font-weight:bold; margin-right:8px;">{i}.</span>
                    <a class="title-link" href="{paper['url']}" target="_blank">{paper['title']}</a>{star_badge}
                    {summary_html}
                    <span class="source">&#x1F50E; {paper['source']}</span>
                    <a class="star-link" href="{star_url}" target="_blank" title="Segna come preferito — articoli simili avranno priorità">[&#x2605; Star]</a>
                    <a class="reject-link" href="{reject_url}" target="_blank" title="Rifiuta — articoli simili saranno filtrati">[Reject]</a>
                </div>
                """
        else:
            html += '<p class="empty">Nessun nuovo articolo questa settimana.</p>'

    html += """
        <div class="footer">
            <p>Generato da Law Newsletter Agent<br>
            Fonti: Riviste giuridiche, RSS, Google Scholar<br>
            <em>[&#x2605; Star] = più articoli simili &bull; [Reject] = meno articoli simili &mdash; basta premere "Submit" sulla GitHub Issue</em></p>
        </div>
        </div>
    </body>
    </html>
    """
    return html


def send_email(html_content):
    """Send newsletter via email."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Diritto del Lavoro Weekly - {datetime.now():%d %B %Y}"
    msg["From"] = EMAIL_CONFIG["sender"]
    msg["To"] = EMAIL_CONFIG["recipient"]
    msg.attach(MIMEText(html_content, "html"))

    try:
        if EMAIL_CONFIG.get("use_ssl", False):
            with smtplib.SMTP_SSL(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
                server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
                server.sendmail(EMAIL_CONFIG["sender"], EMAIL_CONFIG["recipient"], msg.as_string())
        else:
            with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
                server.starttls()
                server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
                server.sendmail(EMAIL_CONFIG["sender"], EMAIL_CONFIG["recipient"], msg.as_string())
        print("\n  Newsletter sent!")
        return True
    except Exception as e:
        print(f"\n  Email error: {e}")
        return False


def run_newsletter():
    print(f"\n{'='*50}")
    print(f"  Law Newsletter - {datetime.now():%Y-%m-%d %H:%M}")
    print('='*50)

    # Step 1: Sync GitHub Issues into feedback.json and close them
    sync_and_cleanup()

    # Step 2: Collect, filter, format, and send
    newsletter = collect_papers()
    html = format_newsletter(newsletter)
    success = send_email(html)

    # Step 3: Record all sent papers in history (prevents duplicates next week)
    if success:
        sent_history = load_sent_history()
        for category_data in newsletter.values():
            for paper in category_data["papers"]:
                sent_history.add(_normalize_title(paper["title"]))
        save_sent_history(sent_history)


def main():
    if "--run-once" in sys.argv:
        run_newsletter()
        return

    try:
        import schedule
    except ImportError:
        print("For scheduled mode: pip install schedule")
        print("Or use: python law_newsletter_agent.py --run-once")
        return

    print("Law Newsletter Agent Started")
    print("Scheduled: Every Monday at 8:00 AM")
    schedule.every().monday.at("08:00").do(run_newsletter)

    import time
    while True:
        schedule.run_pending()
        time.sleep(3600)


if __name__ == "__main__":
    main()
