"""
Zi Safe — SDS Safety Summarizer (Streamlit + SerpAPI + PubChem + AI auto-format)
"""

import os
import re
import json
import time
import hashlib
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, unquote

import requests
import streamlit as st
from dotenv import load_dotenv
from pdfminer.high_level import extract_text  # correct import for pdfminer.six
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
import httpx

# ---------- Setup ----------
load_dotenv()

def get_secret(name: str, default: str = ""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

SERPAPI_KEY = get_secret("SERPAPI_KEY", "")
st.set_page_config(page_title="Zi Safe", page_icon="🧪", layout="wide")

BASE_DIR = pathlib.Path(__file__).parent
CACHE_DIR = BASE_DIR / "sds_cache"
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_SUPPLIERS = [
    "sigmaaldrich.com", "fishersci.com", "alfa.com", "vwr.com", "caymanchem.com",
    "thermofisher.com", "oakwoodchemical.com", "spectrumchemical.com", "tcichemicals.com",
    "acros.com", "strem.com", "merckmillipore.com",
]

USER_AGENT = "ZiSafe/1.9"
TIMEOUT = 25
HEADERS = {"User-Agent": USER_AGENT}

# ---------- AI Auto-format (OpenAI-compatible via OpenRouter or OpenAI) ----------
AI_SYSTEM_PROMPT = """You are a chemistry-aware editor that normalizes experimental procedures.
Return STRICT JSON that follows this schema:
{
  "reactants": "semicolon-separated list (no trailing semicolon)",
  "products": "semicolon-separated list",
  "solvents": "semicolon-separated list",
  "normalized_procedure": "the cleaned procedure body with the three label lines removed"
}
Rules:
- Only include chemical names in those three lists (no quantities, units, temperatures, or verbs).
- Keep names as written by the user if they look correct; otherwise fix obvious typos.
- Parentheses and commas inside names are allowed; separate ITEMS with semicolons.
- If a list is unknown, return an empty string "" (not null).
- DO NOT add commentary. Output ONLY the JSON object.
"""

def _get_llm_client():
    api_key = get_secret("LLM_API_KEY", "")
    base_url = get_secret("LLM_BASE_URL", "") or "https://openrouter.ai/api/v1"
    model = get_secret("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        return None, None, None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return base_url, model, headers

def ai_autofix_procedure(raw_text: str) -> Optional[dict]:
    base_url, model, headers = _get_llm_client()
    if not base_url:
        st.error("LLM not configured. Add LLM_API_KEY (and optionally LLM_BASE_URL, LLM_MODEL) to secrets.")
        return None

    messages = [
        {"role": "system", "content": AI_SYSTEM_PROMPT},
        {"role": "user", "content": raw_text or ""},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},  # supported by OpenRouter & OpenAI
    }

    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        obj = json.loads(content)

        def _norm(s):
            if not isinstance(s, str): return ""
            parts = [p.strip() for p in re.split(r"[;\n]+", s) if p.strip()]
            return "; ".join(parts[:30])

        return {
            "reactants": _norm(obj.get("reactants", "")),
            "products": _norm(obj.get("products", "")),
            "solvents": _norm(obj.get("solvents", "")),
            "normalized_procedure": obj.get("normalized_procedure", "").strip(),
        }
    except Exception as e:
        st.error(f"AI formatting failed: {e}")
        return None

# ---------- Models ----------
@dataclass
class SDSDoc:
    chemical: str
    supplier: str
    url: str
    filepath: str
    text: str

class PubChemInfo(BaseModel):
    name_input: str
    cid: Optional[int] = None
    canonical_name: Optional[str] = None
    iupac_name: Optional[str] = None
    synonyms: List[str] = []
    ghs_hcodes: List[str] = []

class AppState(BaseModel):
    procedure_text: str = Field("")
    suppliers: List[str] = Field(default_factory=lambda: DEFAULT_SUPPLIERS)
    use_pubchem_normalize: bool = Field(default=True)
    max_results: int = Field(default=12)
    max_chemicals: int = Field(default=20)
    delay_s: float = Field(default=0.3)
    jurisdiction: str = Field(default="US (Federal)")
    compliance_enabled: bool = Field(default=True)

# ---------- Section detection ----------
SECTION_HINTS = [
    ("hazard", ["Hazard(s) identification", "Hazard identification", "GHS", "Hazard statements", "H\\d{3}", "P\\d{3}"]),
    ("first_aid", ["First aid measures", "First-aid measures", "First Aid"]),
    ("fire_fighting", ["Fire-fighting measures", "Fire fighting measures", "Fire Fighting"]),
    ("accidental_release", ["Accidental release measures", "Spill", "Leak"]),
    ("handling_storage", ["Handling and storage", "Handling", "Storage"]),
    ("exposure_ppe", ["Exposure controls", "Exposure Controls/Personal Protection", "Personal protection", "PPE", "Control parameters"]),
    ("stability_reactivity", ["Stability and reactivity", "Reactivity", "Incompatibilities", "Conditions to avoid"]),
    ("toxicology", ["Toxicological information", "Toxicology"]),
]

# ---------- Operation cues for procedure-aware tips ----------
OPERATION_CUES = {
    "heating": ["heat", "heating", "reflux", "boil", "distill", "evaporate", "dry under vacuum"],
    "acid_base": ["add acid", "add base", "neutralize", "acidify", "basify", "pH"],
    "oxidizer": ["add*peroxide", "KMnO4", "chromium(VI)", "bleach", "nitric acid"],
    "reducing": ["NaBH4", "LiAlH4", "hydride", "hydrogenation"],
    "pressurized": ["pressure", "autoclave", "sealed tube"],
    "exotherm": ["exotherm", "vigorous", "add slowly", "cooling bath", "ice bath"],
    "volatile": ["diethyl ether", "hexane", "acetone", "dichloromethane", "chloroform", "acetonitrile"],
    "cryogenic": ["dry ice", "liquid nitrogen"],
}

# ---------- Helpers ----------
def guess_supplier_from_url(url: str, custom_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    try:
        p = urlparse(url.strip())
    except Exception:
        return None
    host = (p.netloc or "").lower().replace("www.", "").replace("m.", "")
    path = unquote((p.path or "").lower())

    generic = {
        "drive.google.com", "docs.google.com", "google.com", "dropbox.com", "dl.dropboxusercontent.com",
        "onedrive.live.com", "1drv.ms", "sharepoint.com", "box.com", "app.box.com", "github.com",
        "gitlab.com", "bitbucket.org", "readthedocs.io", "medium.com", "wikipedia.org",
        "scholar.google.com", "bing.com", "yahoo.com", "archive.org", "scribd.com",
    }
    if any(host == gh or host.endswith("." + gh) for gh in generic):
        return None

    fragment_map = {
        "sigmaaldrich": "Sigma-Aldrich (Merck)",
        "sigma-aldrich": "Sigma-Aldrich (Merck)",
        "merckmillipore": "Merck",
        "merck": "Merck",
        "fishersci": "Fisher Scientific (Thermo)",
        "thermofisher": "Thermo Fisher Scientific",
        "alfa": "Alfa Aesar (Thermo)",
        "acros": "Acros Organics (Thermo)",
        "tcichemicals": "TCI Chemicals",
        "vwr": "VWR (Avantor)",
        "spectrumchemical": "Spectrum Chemical",
        "carlroth": "Carl Roth",
        "honeywell": "Honeywell Research Chemicals",
        "fluka": "Fluka (Honeywell)",
        "aldrich": "Sigma-Aldrich (Merck)",
        "aeciworld": "AECI",
        "aeci": "AECI",
        "sasol": "Sasol",
        "mintek": "Mintek",
    }
    if custom_map:
        fragment_map.update({k.lower(): v for k, v in custom_map.items()})

    tokens = re.split(r"[.\-/ _]+", host) + re.split(r"[.\-/ _]+", path)
    for frag in sorted(fragment_map, key=len, reverse=True):
        if any(frag in t for t in tokens):
            return fragment_map[frag]

    parts = host.split(".")
    if len(parts) >= 2:
        tld2 = ".".join(parts[-2:])
        if tld2 in {"co.za", "com.au", "co.uk", "com.mx", "com.br"} and len(parts) >= 3:
            base = parts[-3]
        else:
            base = parts[-2] if parts[-1] in {"com", "org", "net"} else parts[0]
    else:
        base = parts[0] if parts else ""

    base = re.sub(r"[^a-z0-9]+", " ", base).strip()
    if not base or base in {"files", "storage", "cdn", "assets"}:
        return None

    supplier = " ".join(w.capitalize() for w in base.split())
    fixes = {"3m": "3M", "Dow": "Dow", "Basf": "BASF", "Dupont": "DuPont", "Aeci": "AECI", "Tci": "TCI", "Vwr": "VWR", "Sds": "SDS", "Sasol": "Sasol"}
    return fixes.get(supplier, supplier) or None

CHEM_REGEXES = [
    r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
    r"\b([A-Z][a-z]+\s+\(.+?\))\b",
    r"\b([A-Z][a-z]+(?:\s+[A-Za-z0-9\-]+){0,3})\s*(?:solution|aq\.|anhydrous)?\b",
    r"\b([A-Za-z0-9\-]+\s*acid)\b",
    r"\b([A-Za-z0-9\-]+\s*oxide)\b",
    r"\b([A-Za-z0-9\-]+\s*chloride)\b",
    r"\b([A-Za-z0-9\-]+\s*bromide)\b",
    r"\b([A-Za-z0-9\-]+\s*iodide)\b",
    r"\b([A-Za-z0-9\-]+\s*hydroxide)\b",
    r"\b([A-Za-z0-9\-]+\s*nitrate)\b",
    r"\b([A-Za-z0-9\-]+\s*sulfate)\b",
    r"\b([A-Za-z0-9\-]+\s*sulphate)\b",
]
COMMON_NON_CHEM = {"water", "ice", "air", "nitrogen", "argon", "vacuum", "steam"}

def cap_title(name: str) -> str:
    return " ".join(p.capitalize() if not p.isupper() else p for p in name.split())

def detect_chemicals(text: str) -> List[str]:
    found = set()
    txt = text.replace("\n", " ")
    for pat in CHEM_REGEXES:
        for m in re.finditer(pat, txt, flags=re.IGNORECASE):
            name = m.group(1).strip().strip(",.;:")
            name_norm = re.sub(r"\s+", " ", name).lower()
            if len(name_norm) < 3:
                continue
            if name_norm in COMMON_NON_CHEM:
                continue
            found.add(name_norm)
    return sorted({cap_title(n) for n in found})[:25]

# ---------- Chemical list parsing (line-scoped) ----------
LABELS = {"reactants": r"Reactants?", "products": r"Products?", "solvents": r"Solvents?"}

def _extract_list_after_label(text: str, label_regex: str) -> str:
    if not text:
        return ""
    m = re.search(fr"(?im)^\s*{label_regex}\s*[:\-]\s*(.+)$", text)
    return (m.group(1).strip() if m else "")

def _clean_semicolon_list(s: str) -> str:
    if not s:
        return ""
    items_raw = re.split(r"[;,\n]+", s)
    cleaned: List[str] = []
    seen = set()
    drop_patterns = [
        r"\b(stir|heat|cool|add|filter|wash|dry|evaporate|reflux|quench|mix|transfer|record|allow)\b",
        r"\b(mL|ml|L|g|mg|µL|°C|deg|hours?|hrs?|mins?|seconds?)\b",
        r"^\d+(\.\d+)?$",
        r"^\d+(\.\d+)?\s*(mL|ml|L|g|mg)$",
    ]
    drop_re = re.compile("|".join(drop_patterns), re.IGNORECASE)
    for it in items_raw:
        t = it.strip()
        if not t or drop_re.search(t) or len(t) < 2:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(t)
    return "; ".join(cleaned[:30])

def fallback_guess_chems(text: str, max_items=12) -> str:
    candidates = detect_chemicals(text or "")
    return "; ".join(candidates[:max_items])

def parse_chem_lists(source_text: str) -> Tuple[str, str, str]:
    rx = _extract_list_after_label(source_text, LABELS["reactants"])
    px = _extract_list_after_label(source_text, LABELS["products"])
    sx = _extract_list_after_label(source_text, LABELS["solvents"])
    rx = _clean_semicolon_list(rx) or ""
    px = _clean_semicolon_list(px) or ""
    sx = _clean_semicolon_list(sx) or ""
    if not (rx or px or sx):
        guess = fallback_guess_chems(source_text)
        rx = guess
    return rx, px, sx

def _hash_text(t: str) -> str:
    return hashlib.sha256((t or "").encode("utf-8")).hexdigest()

# ---------- SDS Search & PDF ----------
def search_sds_serpapi(chemical: str, suppliers: List[str], num: int = 12):
    if not SERPAPI_KEY:
        return []
    q = f"{chemical} SDS filetype:pdf (" + " OR ".join([f"site:{d}" for d in suppliers]) + ")"
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": q, "num": num, "api_key": SERPAPI_KEY},
            headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("organic_results", []):
            link = item.get("link", "")
            if isinstance(link, str) and link.lower().endswith(".pdf"):
                results.append((item.get("title", ""), link))
        return results
    except Exception:
        return []

def download_pdf(url: str) -> Optional[str]:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    fpath = CACHE_DIR / f"{h}.pdf"
    if fpath.exists() and fpath.stat().st_size > 0:
        return str(fpath)
    try:
        with requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True) as resp:
            ctype = resp.headers.get("Content-Type", "").lower()
            if "pdf" not in ctype and not url.lower().endswith(".pdf"):
                return None
            resp.raise_for_status()
            with open(fpath, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        return str(fpath)
    except Exception:
        if fpath.exists():
            try: fpath.unlink()
            except Exception: pass
        return None

def extract_pdf_text(fpath: str) -> str:
    try:
        return extract_text(fpath)
    except Exception:
        return ""

# ---------- SDS parsing & summary ----------
def split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    t = re.sub(r"\r", "", text)
    pattern = re.compile(r"(?im)^\s*(section\s*(\d{1,2})\s*[:\-–]\s*)(.+?)$")
    indices: List[Tuple[int, int, str]] = []
    for m in pattern.finditer(t):
        start = m.start(); secno = int(m.group(2)); title = m.group(3).strip()
        indices.append((start, secno, title))
    ends = [i[0] for i in indices] + [len(t)]
    chunks: Dict[str, str] = {}
    for idx, (start, secno, title) in enumerate(indices):
        end = ends[idx + 1]
        chunks[f"section_{secno}"] = t[start:end].strip()

    for key, hints in SECTION_HINTS:
        best = None; best_score = -1
        for sname, chunk in chunks.items():
            head = chunk.split("\n", 1)[0]
            score = max([fuzz.partial_ratio(head.lower(), h.lower()) for h in hints] + [0])
            if score > best_score:
                best_score = score; best = sname
        if best:
            sections[key] = chunks[best]

    if not sections:
        for key, hints in SECTION_HINTS:
            for h in hints:
                m = re.search(rf"(?is){re.escape(h)}(.{{0,2500}})", t)
                if m:
                    sections[key] = m.group(0); break
    return sections

HAZARD_CODE_RE = re.compile(r"\b(H\d{3}|P\d{3})\b")
GHS_WORDS = [
    "flammable", "corrosive", "toxic", "fatal", "poison", "irritant", "sensitization",
    "carcinogen", "mutagen", "reproductive", "aspiration", "acute toxicity",
]

def extract_key_points(sections: Dict[str, str]) -> Dict[str, List[str]]:
    points: Dict[str, List[str]] = {k: [] for k, _ in SECTION_HINTS}
    for key in points.keys():
        chunk = sections.get(key, "")
        if not chunk:
            continue
        codes = sorted(set(HAZARD_CODE_RE.findall(chunk)))
        if codes:
            points.setdefault(key, []).append("Codes: " + ", ".join(codes))
        for line in chunk.splitlines():
            s = line.strip(" -•:\t")
            if not s:
                continue
            if any(w in s.lower() for w in ["wear", "use", "avoid", "keep", "do not", "wash", "gloves", "goggles", "respirator", "ventilation", "fume hood"]):
                points.setdefault(key, []).append(s)
            if any(g in s.lower() for g in GHS_WORDS):
                points.setdefault(key, []).append(s)
        uniq: List[str] = []
        for p in points.get(key, []):
            if not any(fuzz.partial_ratio(p, q) > 90 for q in uniq):
                uniq.append(p)
        points[key] = uniq[:25]
    return points

def procedure_risk_hints(proc: str) -> List[str]:
    tips: List[str] = []
    low = proc.lower()
    def has(pat: str) -> bool:
        if "*" in pat:
            return re.search(pat.replace("*", ".*"), low) is not None
        return pat in low
    if any(has(c) for c in OPERATION_CUES["heating"]):
        tips.append("Heating/reflux: check flash points; use condenser; avoid flames; heat‑resistant gloves.")
    if any(has(c) for c in OPERATION_CUES["acid_base"]):
        tips.append("Acid/base: add acid to water; control exotherms; monitor pH; beware gas evolution.")
    if any(has(c) for c in OPERATION_CUES["oxidizer"]):
        tips.append("Oxidizers: segregate; non‑sparking tools; quench carefully.")
    if any(has(c) for c in OPERATION_CUES["reducing"]):
        tips.append("Hydrides/reducers: exclude moisture/air; slow addition; Class D extinguisher nearby.")
    if any(has(c) for c in OPERATION_CUES["pressurized"]):
        tips.append("Closed/pressurized: rated vessels; shields; relief; leak‑check.")
    if any(has(c) for c in OPERATION_CUES["exotherm"]):
        tips.append("Exotherms: dosing control; pre‑cool; verify heat removal.")
    if any(has(c) for c in OPERATION_CUES["volatile"]):
        tips.append("Volatile solvents: fume hood; ventilation; ground/bond transfers.")
    if any(has(c) for c in OPERATION_CUES["cryogenic"]):
        tips.append("Cryogens: face shield; insulated gloves; dewars; prevent asphyxiation.")
    return tips

def dedupe_ordered(items: List[str]) -> List[str]:
    seen: List[str] = []
    for it in items:
        if not any(fuzz.partial_ratio(it, s) > 92 for s in seen):
            seen.append(it)
    return seen

def consolidate_summary(docs: List[SDSDoc], proc_text: str, pc_map: Dict[str, PubChemInfo]):
    agg: Dict[str, List[str]] = {k: [] for k, _ in SECTION_HINTS}
    citations: List[str] = []
    ghs_notes: List[str] = []
    for doc in docs:
        sections = split_sections(doc.text)
        points = extract_key_points(sections)
        pc = pc_map.get(doc.chemical) or next((pc_map[k] for k in pc_map if k.lower() == doc.chemical.lower()), None)
        if pc and pc.ghs_hcodes:
            sds_codes = set()
            for v in points.get("hazard", []):
                sds_codes.update(re.findall(r"H\d{3}", v))
            miss = [c for c in pc.ghs_hcodes if c not in sds_codes]
            if miss:
                ghs_notes.append(f"[{doc.chemical}] PubChem extra GHS not in SDS text: {', '.join(sorted(set(miss)))} (verify).")
        for key, vals in points.items():
            for v in vals:
                agg[key].append(f"[{doc.chemical} – {doc.supplier}] " + v)
        citations.append(f"- {doc.chemical} – {doc.url}")
    order = ["hazard", "exposure_ppe", "handling_storage", "accidental_release", "first_aid", "fire_fighting", "stability_reactivity", "toxicology"]
    md = ["# Safety Brief (Auto‑generated from SDS)", "\n**NOTE:** Verify against original SDS and local EHS policies."]
    if proc_text.strip():
        md.append("\n## Procedure‑aware cautions")
        for t in procedure_risk_hints(proc_text):
            md.append(f"- {t}")
    if ghs_notes:
        md.append("\n## GHS cross‑check notes (PubChem)")
        for n in ghs_notes:
            md.append(f"- {n}")
    for key in order:
        items = dedupe_ordered(agg.get(key, []))[:25]
        if items:
            md.append(f"\n## {key.replace('_', ' ').title()}")
            md.extend(f"- {it}" for it in items)
    md.append("\n## Sources")
    md.extend(citations)
    return "\n".join(md), agg

# ---------- Compliance seed ----------
KBASE_PATH = BASE_DIR / "data" / "compliance_kbase.json"
SEED_KBASE = {
    "71-43-2": {
        "name": "Benzene",
        "hazards": {"carcinogen": "1A", "mutagen": True, "repro_toxicant": False, "edc": False},
        "lists": {
            "US (Federal)": {"osha_specific_standard": True, "sara_313": True, "urls": ["https://www.osha.gov/benzene", "https://www.epa.gov/toxics-release-inventory-tri-program"]},
            "California (Prop 65)": {"prop65": True, "urls": ["https://oehha.ca.gov/proposition-65"]},
            "EU (REACH/CLP)": {"clp_carcinogen": "1A", "restriction_annex_xvii": True, "urls": ["https://echa.europa.eu/substance-information/-/substanceinfo/100.028.878"]},
            "South Africa (HCA/GHS)": {"ghs_classification": "Carc.1A", "permit_required": True, "permit_text": "Storage over 50 L may require a local authority flammable substance permit.", "permit_url": "https://www.labour.gov.za/legislation/acts/occupational-health-and-safety-act", "urls": ["https://www.labour.gov.za/legislation/acts/occupational-health-and-safety-act"]},
        },
    },
    "117-81-7": {
        "name": "DEHP",
        "hazards": {"carcinogen": "2", "mutagen": False, "repro_toxicant": True, "edc": True},
        "lists": {
            "US (Federal)": {"tsca_action": True, "urls": ["https://www.epa.gov/assessing-and-managing-chemicals-under-tsca"]},
            "California (Prop 65)": {"prop65": True, "urls": ["https://oehha.ca.gov/proposition-65"]},
            "EU (REACH/CLP)": {"svhc": True, "annex_xiv_authorisation": True, "urls": ["https://echa.europa.eu/candidate-list-table"]},
            "South Africa (HCA/GHS)": {"ghs_classification": "Repr.1B", "urls": ["https://www.sabs.co.za/"]},
        },
    },
    "50-00-0": {
        "name": "Formaldehyde",
        "hazards": {"carcinogen": "1B", "mutagen": True, "repro_toxicant": False, "edc": False},
        "lists": {
            "US (Federal)": {"osha_specific_standard": True, "urls": ["https://www.osha.gov/formaldehyde"]},
            "California (Prop 65)": {"prop65": True, "urls": ["https://oehha.ca.gov/proposition-65"]},
            "EU (REACH/CLP)": {"clp_carcinogen": "1B", "urls": ["https://echa.europa.eu/"]},
            "South Africa (HCA/GHS)": {"ghs_classification": "Carc.1B", "urls": ["https://www.labour.gov.za/"]},
        },
    },
}

def load_kbase() -> Dict[str, dict]:
    data = {}
    try:
        if KBASE_PATH.exists():
            with KBASE_PATH.open("r", encoding="utf-8") as f:
                j = json.load(f)
                if isinstance(j, dict):
                    data.update(j)
    except Exception:
        pass
    for k, v in SEED_KBASE.items():
        data.setdefault(k, v)
    return data

KBASE = load_kbase()
KBASE_NAME_INDEX = {(rec.get("name", "").strip().lower()): cas for cas, rec in KBASE.items() if rec.get("name")}
H_PATTERN = re.compile(r"\bH(3\d{2}|4\d{2})i?\b", re.IGNORECASE)

def extract_h_statements(text: str) -> set:
    if not text:
        return set()
    return set(m.group(0).upper().replace("H", "").replace("I", "I") for m in H_PATTERN.finditer(text))

def infer_flags_from_h(hs: set) -> dict:
    return {"carcinogen": "1" if any(h in {"350", "350I"} for h in hs) else None,
            "repro_toxicant": any(h in {"360", "361"} for h in hs),
            "mutagen": any(h in {"340", "341"} for h in hs),
            "edc": False}

@dataclass
class Alert:
    level: str
    substance: str
    cas: str
    messages: List[str]
    links: List[str]

def lookup_kbase(cas: Optional[str], name: Optional[str]) -> Optional[dict]:
    if cas and cas in KBASE:
        return KBASE[cas]
    if name:
        cas2 = KBASE_NAME_INDEX.get(name.strip().lower())
        if cas2 in KBASE:
            return KBASE[cas2]
    return None

def best_cas_from_pubchem(info: Optional[PubChemInfo]) -> Optional[str]:
    if not info:
        return None
    for s in info.synonyms:
        s = s.strip()
        if re.match(r"^\d{2,7}-\d{2}-\d$", s):
            return s
    return None

def evaluate_substance(sub_name: str, cas: Optional[str], jurisdiction: str, h_codes: set) -> Optional[Alert]:
    k = lookup_kbase(cas, sub_name)
    messages: List[str] = []
    links: List[str] = []
    level = "info"

    inferred = infer_flags_from_h(h_codes) if h_codes else {}
    if inferred.get("carcinogen"):
        level = "warning"; messages.append("Classified as carcinogenic (H350 detected in SDS).")
    if inferred.get("repro_toxicant"):
        level = "warning"; messages.append("Reproductive / endocrine concern (H360/H361 detected in SDS).")
    if inferred.get("mutagen"):
        level = "warning"; messages.append("Mutagenicity concern (H340/H341 detected in SDS).")

    if k:
        hz = k.get("hazards", {})
        if hz.get("carcinogen") in {"1", "1A", "1B"}:
            level = "warning"; messages.append(f"Carcinogen category {hz['carcinogen']} (authoritative list).")
        if hz.get("repro_toxicant") or hz.get("edc"):
            level = "warning"; messages.append("Endocrine/reproductive toxicant flagged by authority list.")

        jr = k.get("lists", {}).get(jurisdiction, {})
        if jr.get("annex_xiv_authorisation"):
            messages.append("Authorization required before use (Annex XIV).")
        if jr.get("restriction_annex_xvii"):
            messages.append("Restricted use conditions apply (Annex XVII).")
        if jr.get("osha_specific_standard"):
            messages.append("Subject to an OSHA substance‑specific standard (monitoring/medical surveillance may apply).")
        if jr.get("prop65"):
            messages.append("Listed under California Proposition 65 — warnings/notifications may apply.")
        if jr.get("tsca_action"):
            messages.append("Subject to TSCA rulemaking or action (check conditions).")
        if jr.get("ghs_classification"):
            messages.append(f"GHS classification in jurisdiction: {jr['ghs_classification']}.")
        if jr.get("permit_required"):
            messages.append(jr.get("permit_text", "Permit required in this jurisdiction."))
            if jr.get("permit_url"):
                links.append(jr["permit_url"])
        links.extend(jr.get("urls", []))

    if not messages:
        return None
    return Alert(level=level, substance=sub_name, cas=(cas or "—"), messages=messages, links=links)

def render_alerts(detected_substances: List[Tuple[str, Optional[str]]], jurisdiction: str, docs: List[SDSDoc], pc_map: Dict[str, PubChemInfo]):
    all_text = "\n".join(d.text for d in docs)
    h_codes = extract_h_statements(all_text)

    alerts: List[Alert] = []
    for name, cas in detected_substances:
        a = evaluate_substance(name, cas, jurisdiction, h_codes)
        if a:
            alerts.append(a)

    if not alerts:
        return

    any_warning = any(a.level == "warning" for a in alerts)
    header = "⚠️ Compliance Alerts — Action Recommended" if any_warning else "ℹ️ Compliance Notes"

    box = st.warning if any_warning else st.info
    with st.container():
        box(header)
        for a in alerts:
            st.markdown(f"**{a.substance}** ({a.cas})")
            for m in sorted(set(a.messages)):
                st.markdown(f"- {m}")
            if a.links:
                st.markdown("  Links:")
                for url in sorted(set(a.links)):
                    st.markdown(f"  - [{url}]({url})")
        st.caption("This tool supports safety reviews; always verify against the original SDS and official regulatory texts for your site and use case.")

# ---------- UI ----------
st.title("🧪 Zi Safe — SDS Safety Summarizer")
st.caption("Build: 2025‑08‑13 • Autofill + AI formatter")

with st.expander("How it works & disclaimer", expanded=False):
    st.markdown("""This tool searches vendor sites for SDS PDFs, extracts relevant sections, and builds a consolidated safety brief.
SDS formats vary; always cross‑check with the original PDFs and your institution's rules. For critical work, get an EHS review.
By using this tool, you confirm you have rights to access the referenced SDS.""")

state = AppState()

col1, col2 = st.columns([2, 1])
with col1:
    proc_text = st.text_area("Paste your procedure (free text)", height=220,
                             placeholder="e.g., Reactants: acetone; benzaldehyde; NaOH\nSolvents: ethanol; water\nProducts: benzyl alcohol; sodium benzoate\nThen warm, add acetic anhydride; cool, filter...")
with col2:
    supplier_str = st.text_input("Restrict search to suppliers (comma‑separated domains)",
                                 ", ".join(DEFAULT_SUPPLIERS))
    state.suppliers = [s.strip() for s in supplier_str.split(",") if s.strip()]
    state.use_pubchem_normalize = st.toggle("Normalize via PubChem", value=True,
                                            help="Canonical names, CID & GHS H‑codes")

with st.sidebar:
    st.subheader("Search Settings")
    state.max_results = st.number_input("Results per chemical", 5, 50, 12, step=1)
    state.max_chemicals = st.number_input("Max chemicals per run", 1, 100, 20, step=1)
    state.delay_s = st.slider("Delay between requests (s)", 0.0, 1.0, 0.3, 0.1)
    st.subheader("Compliance")
    state.compliance_enabled = st.toggle("Enable compliance alerts", value=True)
    state.jurisdiction = st.selectbox("Jurisdiction", ["US (Federal)", "California (Prop 65)", "EU (REACH/CLP)", "South Africa (HCA/GHS)"], index=0)

# ---------- AI auto-format controls ----------
st.caption("Need help formatting? Click AI to normalize your text into the exact format this app expects.")
ai_col1, ai_col2 = st.columns([1,1])
with ai_col1:
    run_ai = st.button("✨ AI: Auto‑format procedure", use_container_width=True)
with ai_col2:
    use_ai_replace = st.toggle("Replace procedure with AI‑cleaned version", value=False,
                               help="If ON, your procedure box will be replaced with the normalized version (labels removed).")

if run_ai:
    with st.spinner("AI is formatting your procedure…"):
        res = ai_autofix_procedure(proc_text)
    if res:
        st.session_state["reactants_field"] = res["reactants"]
        st.session_state["products_field"] = res["products"]
        st.session_state["solvents_field"] = res["solvents"]
        if use_ai_replace:
            proc_text = res["normalized_procedure"]
            st.session_state["autofill_source_hash"] = None  # force a refresh of defaults

# ---------- One-time autofill when procedure changes ----------
current_hash = _hash_text(proc_text)
if st.session_state.get("autofill_source_hash") != current_hash:
    r_default, p_default, s_default = parse_chem_lists(proc_text)
    st.session_state["reactants_field"] = r_default
    st.session_state["products_field"]  = p_default
    st.session_state["solvents_field"]  = s_default
    st.session_state["autofill_source_hash"] = current_hash

# ---------- Chemical fields ----------
st.subheader("Chemicals (edit as needed)")
reactants = st.text_area(
    "Reactants (semicolon‑separated)",
    key="reactants_field",
    placeholder="e.g., acetone; benzaldehyde; sodium hydroxide",
    height=80,
    help="Use semicolons (;) so commas inside names are preserved."
)
products = st.text_area(
    "Products (semicolon‑separated)",
    key="products_field",
    placeholder="e.g., benzyl alcohol; sodium benzoate",
    height=80
)
solvents = st.text_area(
    "Solvents (semicolon‑separated)",
    key="solvents_field",
    placeholder="e.g., ethanol; water; toluene",
    height=80
)

# Build the combined chemical list (dedup, keep order)
def _split_semis(s: str) -> List[str]:
    return [c.strip() for c in (s or "").split(";") if c.strip()]

combined = _split_semis(reactants) + _split_semis(products) + _split_semis(solvents)
seen = set()
chemical_list: List[str] = []
for c in combined:
    key = c.lower()
    if key not in seen:
        seen.add(key)
        chemical_list.append(c)

# --- Diagnostics & manual SDS input ---
with st.expander("Debug & Manual SDS input", expanded=False):
    st.write("Use this to test your SerpAPI key and paste SDS URLs directly if needed.")
    st.markdown(f"- **SERPAPI_KEY detected?** {'✅' if SERPAPI_KEY else '❌'}")
    test_query = st.text_input("Test query (for SerpAPI)", value="benzene SDS filetype:pdf site:sigmaaldrich.com")
    if st.button("Run SerpAPI test"):
        if not SERPAPI_KEY:
            st.error("No SERPAPI_KEY loaded. Put it in .env (local) or Secrets (Streamlit Cloud).")
        else:
            try:
                r = requests.get(
                    "https://serpapi.com/search.json",
                    params={"engine": "google", "q": test_query, "num": 5, "api_key": SERPAPI_KEY},
                    headers={"User-Agent": "ZiSafe/diag"},
                    timeout=20,
                )
                st.write("HTTP status:", r.status_code)
                data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                org = data.get("organic_results", [])
                st.write(f"Found {len(org)} organic_results")
                for i, it in enumerate(org[:5], 1):
                    st.write(i, it.get("title"), it.get("link"))
            except Exception as e:
                st.exception(e)

    st.markdown("---")
    st.write("Paste one or more SDS PDF URLs (one per line). These will be used in addition to search results.")
    manual_urls = st.text_area("SDS PDF URLs", height=100, placeholder="https://...pdf\nhttps://...pdf")

# ---------- Run search & summarizer ----------
run = st.button("🔎 Fetch SDS & Summarize", type="primary", disabled=(len(chemical_list) == 0))

results_docs: List[SDSDoc] = []
status_placeholder = st.empty()

if run:
    if not SERPAPI_KEY:
        st.error("SERPAPI_KEY missing. On Streamlit Cloud, set it in Secrets. Locally, add it to .env or .streamlit/secrets.toml.")
    else:
        with st.spinner("Searching SDS and building your safety brief…"):
            # Search via SerpAPI
            for chem in chemical_list[: state.max_chemicals]:
                status_placeholder.info(f"Searching SDS for **{chem}**…")
                time.sleep(state.delay_s)
                hits = search_sds_serpapi(chem, state.suppliers, num=state.max_results)
                chosen = False
                for title, link in hits:
                    fpath = download_pdf(link)
                    if not fpath:
                        continue
                    text = extract_pdf_text(fpath)
                    if len(text) < 500 and "hazard" not in text.lower() and "first aid" not in text.lower():
                        continue
                    supplier_guess = guess_supplier_from_url(link) or "unknown"
                    results_docs.append(SDSDoc(chemical=chem, supplier=supplier_guess, url=link, filepath=fpath, text=text))
                    chosen = True
                    break
                if not chosen:
                    status_placeholder.warning(f"No SDS PDF found or downloadable for {chem}. Try adjusting the name or suppliers.")

            # Manual URL handling
            extra_links = [u.strip() for u in (manual_urls or "").splitlines() if u.strip()]
            for link in extra_links:
                fpath = download_pdf(link)
                if not fpath:
                    st.warning(f"Manual URL not usable as PDF: {link}")
                    continue
                text = extract_pdf_text(fpath)
                if len(text) < 200:
                    st.warning(f"Manual SDS appears to have little/no text (maybe scanned): {link}")
                supplier_guess = guess_supplier_from_url(link) or "manual"
                results_docs.append(SDSDoc(chemical="(manual SDS)", supplier=supplier_guess, url=link, filepath=fpath, text=text))

            status_placeholder.empty()

# ---------- Render results ----------
if results_docs:
    # PubChem normalization
    pc_map: Dict[str, PubChemInfo] = {}
    if chemical_list and state.use_pubchem_normalize:
        with st.spinner("Querying PubChem for canonical names & GHS…"):
            # minimal PubChem calls to avoid rate limits
            def pubchem_name_to_cid(name: str) -> Optional[int]:
                try:
                    r = requests.get(
                        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/cids/JSON",
                        headers=HEADERS, timeout=TIMEOUT)
                    if r.status_code != 200:
                        return None
                    cids = r.json().get("IdentifierList", {}).get("CID", [])
                    return cids[0] if cids else None
                except Exception:
                    return None

            def pubchem_cid_summary(cid: int):
                info = PubChemInfo(name_input=str(cid), cid=cid)
                try:
                    r = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/",
                                     headers=HEADERS, timeout=TIMEOUT)
                    if r.status_code != 200:
                        return info
                    record = r.json().get("Record", {})
                    sections = record.get("Section", [])
                    def find_sec(lst, name):
                        for s in lst:
                            if s.get("TOCHeading") == name: return s
                        return None
                    comp_id = find_sec(sections, "Names and Identifiers")
                    if comp_id:
                        for s in comp_id.get("Section", []):
                            if s.get("TOCHeading") == "IUPAC Name":
                                for it in s.get("Information", []):
                                    info.iupac_name = it.get("Value", {}).get("StringWithMarkup", [{}])[0].get("String")
                            if s.get("TOCHeading") == "Synonyms":
                                for it in s.get("Information", []):
                                    for v in it.get("Value", {}).get("StringWithMarkup", []):
                                        sv = v.get("String"); 
                                        if sv: info.synonyms.append(sv)
                            if s.get("TOCHeading") == "Record Title":
                                for it in s.get("Information", []):
                                    info.canonical_name = it.get("Value", {}).get("StringWithMarkup", [{}])[0].get("String")
                    safety = find_sec(sections, "Safety and Hazards")
                    if safety:
                        for s in safety.get("Section", []):
                            if s.get("TOCHeading") == "GHS Classification":
                                for it in s.get("Information", []):
                                    for tb in it.get("Value", {}).get("StringWithMarkup", []):
                                        for code in re.findall(r"\bH\d{3}\b", tb.get("String", "")):
                                            if code not in info.ghs_hcodes:
                                                info.ghs_hcodes.append(code)
                except Exception:
                    pass
                info.synonyms = sorted(set(info.synonyms))[:50]
                return info

            for n in chemical_list:
                cid = pubchem_name_to_cid(n)
                pc_map[n] = pubchem_cid_summary(cid) if cid else PubChemInfo(name_input=n)
                pc_map[n].name_input = n

    # Compliance alerts
    detected_substances: List[Tuple[str, Optional[str]]] = []
    for chem in chemical_list:
        info = pc_map.get(chem)
        # try to infer CAS from PubChem synonyms or KBASE index
        cas = None
        if info:
            for s in info.synonyms:
                s2 = s.strip()
                if re.match(r"^\d{2,7}-\d{2}-\d$", s2):
                    cas = s2; break
        if not cas:
            name_key = (info.canonical_name or chem).strip().lower() if info else chem.strip().lower()
            cas = KBASE_NAME_INDEX.get(name_key)
        detected_substances.append((chem, cas))

    if state.compliance_enabled:
        render_alerts(detected_substances, state.jurisdiction, results_docs, pc_map)

    md_summary, _ = consolidate_summary(results_docs, proc_text, pc_map)
    st.subheader("Consolidated Safety Summary")
    st.markdown(md_summary)
    st.download_button("💾 Download Markdown", data=md_summary.encode("utf-8"), file_name="zi_safe_safety_brief.md")

    st.subheader("SDS Sources Found")
    for d in results_docs:
        st.write(f"**{d.chemical}** – {d.supplier}")
        st.code(d.url)
else:
    st.info("Enter your procedure and fill the Reactants/Products/Solvents (semicolon‑separated), then click ‘Fetch SDS & Summarize’.")
