"""
SDS Safety Summarizer – Streamlit App (SerpAPI + PubChem, fixed)

This build addresses a runtime error calling guess_supplier_from_url by:
- Defining guess_supplier_from_url earlier in the file
- Using Python 3.8+ compatible type hints (Optional/Dict instead of | None / dict[])
- Handling None supplier gracefully at call sites
"""

import os
import re
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional  # 3.8+ compatible
from urllib.parse import urlparse, unquote

import requests
from pdfminer.high_level import extract_text
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from dotenv import load_dotenv

import streamlit as st

# -----------------------
# Config & Utilities
# -----------------------
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ↓ Helper so it works both locally (.env) and on Streamlit Cloud (st.secrets)
def get_secret(name: str, default: str = ""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

SERPAPI_KEY = get_secret("SERPAPI_KEY", "")

load_dotenv()
CACHE_DIR = os.path.join(os.path.dirname(__file__), "sds_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
DEFAULT_SUPPLIERS = [
    "sigmaaldrich.com", "fishersci.com", "alfa.com", "vwr.com", "caymanchem.com",
    "thermofisher.com", "oakwoodchemical.com", "spectrumchemical.com", "tcichemicals.com",
    "acros.com", "strem.com", "merckmillipore.com"
]

USER_AGENT = "SDS-Summarizer/1.4 (+local)"
TIMEOUT = 25
HEADERS = {"User-Agent": USER_AGENT}

SECTION_HINTS = [
    ("hazard", ["Hazard(s) identification", "Hazard identification", "GHS", "Hazard statements", "H\\d{3}", "P\\d{3}" ]),
    ("first_aid", ["First aid measures", "First-aid measures", "First Aid"]),
    ("fire_fighting", ["Fire-fighting measures", "Fire fighting measures", "Fire Fighting"]),
    ("accidental_release", ["Accidental release measures", "Spill", "Leak" ]),
    ("handling_storage", ["Handling and storage", "Handling", "Storage"]),
    ("exposure_ppe", ["Exposure controls", "Exposure Controls/Personal Protection", "Personal protection", "PPE", "Control parameters"]),
    ("stability_reactivity", ["Stability and reactivity", "Reactivity", "Incompatibilities", "Conditions to avoid"]),
    ("toxicology", ["Toxicological information", "Toxicology"]),
]

OPERATION_CUES = {
    "heating": ["heat", "heating", "reflux", "boil", "distill", "evaporate", "dry under vacuum"],
    "acid_base": ["add acid", "add base", "neutralize", "acidify", "basify", "pH"],
    "oxidizer": ["add*peroxide", "KMnO4", "chromium(VI)", "bleach", "nitric acid"],
    "reducing": ["NaBH4", "LiAlH4", "hydride", "hydrogenation"],
    "pressurized": ["pressure", "autoclave", "sealed tube"],
    "exotherm": ["exotherm", "vigorous", "add slowly", "cooling bath", "ice bath"],
    "volatile": ["diethyl ether", "hexane", "acetone", "dichloromethane", "chloroform", "acetonitrile"],
    "cryogenic": ["dry ice", "liquid nitrogen"]
}

# -----------------------
# Supplier inference (3.8+ compatible)
# -----------------------
def guess_supplier_from_url(url: str, custom_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Guess the supplier/manufacturer name from a URL.
    Returns a clean supplier string, or None if it can't confidently guess.
    """
    if not url or not isinstance(url, str):
        return None

    try:
        p = urlparse(url.strip())
    except Exception:
        return None

    host = (p.netloc or "").lower()
    path = unquote((p.path or "").lower())
    host = host.replace("www.", "").replace("m.", "")

    generic_hosts = {
        "drive.google.com", "docs.google.com", "google.com",
        "dropbox.com", "dl.dropboxusercontent.com",
        "onedrive.live.com", "1drv.ms",
        "sharepoint.com", "box.com", "app.box.com",
        "github.com", "gitlab.com", "bitbucket.org",
        "readthedocs.io", "medium.com", "wikipedia.org",
        "scholar.google.com", "bing.com", "yahoo.com",
        "archive.org", "scribd.com"
    }
    if any(host == gh or host.endswith("." + gh) for gh in generic_hosts):
        return None

    fragment_map = {
        "sigmaaldrich": "Sigma-Aldrich (Merck)",
        "sigma-aldrich": "Sigma-Aldrich (Merck)",
        "merckmillipore": "Merck",
        "millipore": "Merck",
        "merckgroup": "Merck",
        "merck": "Merck",
        "supelco": "Supelco (Merck)",
        "thermofisher": "Thermo Fisher Scientific",
        "thermo-fisher": "Thermo Fisher Scientific",
        "thermo": "Thermo Fisher Scientific",
        "fishersci": "Fisher Scientific (Thermo Fisher)",
        "fisherbrand": "Fisherbrand (Thermo Fisher)",
        "alfa": "Alfa Aesar (Thermo Fisher)",
        "alfa-aesar": "Alfa Aesar (Thermo Fisher)",
        "acros": "Acros Organics (Thermo Fisher)",
        "tcichemicals": "TCI Chemicals",
        "tci-chemicals": "TCI Chemicals",
        "tcichemical": "TCI Chemicals",
        "tci": "TCI Chemicals",
        "vwr": "VWR (Avantor)",
        "avantor": "Avantor",
        "avantorsciences": "Avantor",
        "macron": "Macron Fine Chemicals (Avantor)",
        "spectrumchemical": "Spectrum Chemical",
        "coleparmer": "Cole-Parmer",
        "carlroth": "Carl Roth",
        "scharlab": "Scharlab",
        "scharlau": "Scharlau (Scharlab)",
        "basf": "BASF",
        "dow": "Dow",
        "dupont": "DuPont",
        "evonik": "Evonik",
        "arkema": "Arkema",
        "solvay": "Solvay",
        "clariant": "Clariant",
        "croda": "Croda",
        "lubrizol": "Lubrizol",
        "3m": "3M",
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

    def tokenize(s: str) -> List[str]:
        return re.split(r"[.\-/ _]+", s)

    tokens = tokenize(host) + tokenize(path)

    for frag in sorted(fragment_map.keys(), key=len, reverse=True):
        if any(frag in t for t in tokens):
            return fragment_map[frag]

    domain_bits = host.split(".")
    if len(domain_bits) >= 2:
        tld2 = ".".join(domain_bits[-2:])
        if tld2 in {"co.za", "com.au", "co.uk", "com.mx", "com.br"} and len(domain_bits) >= 3:
            base = domain_bits[-3]
        else:
            base = domain_bits[-2] if domain_bits[-1] in {"com", "org", "net"} else domain_bits[0]
    else:
        base = domain_bits[0] if domain_bits else ""

    base = re.sub(r"[^a-z0-9]+", " ", base).strip()
    if not base:
        return None

    if base in {"files", "storage", "cdn", "assets"}:
        return None

    supplier = " ".join(w.capitalize() for w in base.split())
    fixes = {"3m": "3M", "Dow": "Dow", "Basf": "BASF", "Dupont": "DuPont", "Aeci": "AECI",
             "Tci": "TCI", "Vwr": "VWR", "Sds": "SDS", "Sasol": "Sasol"}
    supplier = fixes.get(supplier, supplier)
    return supplier or None

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
    chemical_list: List[str] = Field(default_factory=list)
    suppliers: List[str] = Field(default_factory=lambda: DEFAULT_SUPPLIERS)
    use_pubchem_normalize: bool = Field(default=True)

# -----------------------
# Chemical detection
# -----------------------
CHEM_REGEXES = [
    r"\\b([A-Z][a-z]+\\s+[A-Z][a-z]+)\\b",
    r"\\b([A-Z][a-z]+\\s+\\(.+?\\))\\b",
    r"\\b([A-Z][a-z]+(?:\\s+[A-Za-z0-9\\-]+){0,3})\\s*(?:solution|aq\\.|anhydrous)?\\b",
    r"\\b([A-Za-z0-9\\-]+\\s*acid)\\b",
    r"\\b([A-Za-z0-9\\-]+\\s*oxide)\\b",
    r"\\b([A-Za-z0-9\\-]+\\s*chloride)\\b",
    r"\\b([A-Za-z0-9\\-]+\\s*bromide)\\b",
    r"\\b([A-Za-z0-9\\-]+\\s*iodide)\\b",
    r"\\b([A-Za-z0-9\\-]+\\s*hydroxide)\\b",
    r"\\b([A-Za-z0-9\\-]+\\s*nitrate)\\b",
    r"\\b([A-Za-z0-9\\-]+\\s*sulfate)\\b",
    r"\\b([A-Za-z0-9\\-]+\\s*sulphate)\\b",
]

COMMON_NON_CHEM = {"water", "ice", "air", "nitrogen", "argon", "vacuum", "steam"}

def detect_chemicals(text: str) -> List[str]:
    found = set()
    txt = text.replace("\\n", " ")
    for pat in CHEM_REGEXES:
        for m in re.finditer(pat, txt, flags=re.IGNORECASE):
            name = m.group(1).strip().strip(",.;:")
            name_norm = re.sub(r"\\s+", " ", name).lower()
            if len(name_norm) < 3:
                continue
            if name_norm in COMMON_NON_CHEM:
                continue
            found.add(name_norm)
    cleaned = sorted({cap_title(n) for n in found})
    return cleaned[:25]

def cap_title(name: str) -> str:
    parts = name.split()
    return " ".join(p.capitalize() if not p.isupper() else p for p in parts)

# -----------------------
# SerpAPI search & PDF handling
# -----------------------
def search_sds_serpapi(chemical: str, suppliers: List[str], num: int = 12):
    if not SERPAPI_KEY:
        return []
    domain_query = " OR ".join([f"site:{d}" for d in suppliers])
    q = f"{chemical} SDS filetype:pdf ({domain_query})"
    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": q, "num": num, "api_key": SERPAPI_KEY}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("organic_results", []):
            link = item.get("link", "")
            title = item.get("title", "")
            if isinstance(link, str) and link.lower().endswith(".pdf"):
                results.append((title, link))
        return results
    except Exception:
        return []

def download_pdf(url: str) -> Optional[str]:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    fpath = os.path.join(CACHE_DIR, f"{h}.pdf")
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        return fpath
    try:
        with requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True) as resp:
            ctype = resp.headers.get("Content-Type", "").lower()
            if "pdf" not in ctype and not url.lower().endswith(".pdf"):
                return None
            resp.raise_for_status()
            with open(fpath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return fpath
    except Exception:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass
        return None

def extract_pdf_text(fpath: str) -> str:
    try:
        return extract_text(fpath)
    except Exception:
        return ""

# -----------------------
# PubChem normalization & GHS
# -----------------------
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest"

class PubChemError(Exception):
    pass

def pubchem_name_to_cid(name: str) -> Optional[int]:
    try:
        r = requests.get(
            f"{PUBCHEM_BASE}/pug/compound/name/{requests.utils.quote(name)}/cids/JSON",
            headers=HEADERS, timeout=TIMEOUT
        )
        if r.status_code != 200:
            return None
        data = r.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        return cids[0] if cids else None
    except Exception:
        return None

def pubchem_cid_summary(cid: int):
    info = PubChemInfo(name_input=str(cid), cid=cid)
    try:
        r = requests.get(f"{PUBCHEM_BASE}/pug_view/data/compound/{cid}/JSON/", headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return info
        data = r.json()
        record = data.get("Record", {})
        sections = record.get("Section", [])
        def find_sections(sec_list, name):
            for s in sec_list:
                if s.get("TOCHeading") == name:
                    return s
            return None
        comp_id = find_sections(sections, "Names and Identifiers")
        if comp_id:
            for s in comp_id.get("Section", []):
                if s.get("TOCHeading") == "IUPAC Name":
                    for it in s.get("Information", []):
                        info.iupac_name = it.get("Value", {}).get("StringWithMarkup", [{}])[0].get("String")
                if s.get("TOCHeading") == "Synonyms":
                    for it in s.get("Information", []):
                        vals = it.get("Value", {}).get("StringWithMarkup", [])
                        for v in vals:
                            sname = v.get("String")
                            if sname:
                                info.synonyms.append(sname)
                if s.get("TOCHeading") == "Record Title":
                    for it in s.get("Information", []):
                        info.canonical_name = it.get("Value", {}).get("StringWithMarkup", [{}])[0].get("String")
        safety = find_sections(sections, "Safety and Hazards")
        if safety:
            ghs = None
            for s in safety.get("Section", []):
                if s.get("TOCHeading") == "GHS Classification":
                    ghs = s
                    break
            if ghs:
                for it in ghs.get("Information", []):
                    textblocks = it.get("Value", {}).get("StringWithMarkup", [])
                    for tb in textblocks:
                        s = tb.get("String", "")
                        for code in re.findall(r"\\bH\\d{3}\\b", s):
                            if code not in info.ghs_hcodes:
                                info.ghs_hcodes.append(code)
    except Exception:
        pass
    info.synonyms = sorted(set(info.synonyms))[:50]
    return info

def normalize_via_pubchem(names: List[str]) -> Dict[str, PubChemInfo]:
    out: Dict[str, PubChemInfo] = {}
    for n in names:
        cid = pubchem_name_to_cid(n)
        if cid:
            out[n] = pubchem_cid_summary(cid)
            out[n].name_input = n
        else:
            out[n] = PubChemInfo(name_input=n)
    return out

# -----------------------
# SDS parsing & consolidation
# -----------------------
def split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    t = re.sub(r"\\r", "", text)
    pattern = re.compile(r"(?im)^\\s*(section\\s*(\\d{1,2})\\s*[:\\-–]\\s*)(.+?)$")
    indices: List[Tuple[int, int, int, str]] = []
    for m in pattern.finditer(t):
        start = m.start()
        secno = int(m.group(2))
        title = m.group(3).strip()
        indices.append((start, 0, secno, title))
    ends = [i[0] for i in indices] + [len(t)]
    chunks = {}
    for idx, (start, _, secno, title) in enumerate(indices):
        end = ends[idx+1]
        chunk = t[start:end].strip()
        chunks[f"section_{secno}"] = chunk
    for key, hints in SECTION_HINTS:
        best_score = -1
        best_sec = None
        for sname, chunk in chunks.items():
            head = chunk.split("\\n", 1)[0]
            score = max([fuzz.partial_ratio(head.lower(), h.lower()) for h in hints] + [0])
            if score > best_score:
                best_score = score
                best_sec = sname
        if best_sec:
            sections[key] = chunks[best_sec]
    if not sections:
        for key, hints in SECTION_HINTS:
            for h in hints:
                m = re.search(rf"(?is){re.escape(h)}(.{{0,2500}})", t)
                if m:
                    sections[key] = m.group(0)
                    break
    return sections

HAZARD_CODE_RE = re.compile(r"\\b(H\\d{3}|P\\d{3})\\b")
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
            s = line.strip(" -•:\\t")
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

def procedure_risk_hints(proc_text: str) -> List[str]:
    tips = []
    low = proc_text.lower()
    def has(pat):
        if "*" in pat:
            pat = pat.replace("*", ".*")
            return re.search(pat, low) is not None
        return pat in low
    if any(has(c) for c in OPERATION_CUES["heating"]):
        tips.append("Heating/reflux: verify closed‑cup flash points; use condenser and do not heat above solvent BP; avoid open flames; use heat‑resistant gloves.")
    if any(has(c) for c in OPERATION_CUES["acid_base"]):
        tips.append("Acid/base steps: ALWAYS add acid to water (not the reverse); control exotherms with an ice bath; monitor pH; beware gas evolution.")
    if any(has(c) for c in OPERATION_CUES["oxidizer"]):
        tips.append("Oxidizers: segregate from organics/reducers; use non‑sparking tools; quench carefully; have appropriate spill neutralizer ready.")
    if any(has(c) for c in OPERATION_CUES["reducing"]):
        tips.append("Strong reducing agents/hydrides: exclude moisture/air; add slowly under inert atmosphere; keep Class D extinguishing media accessible.")
    if any(has(c) for c in OPERATION_CUES["pressurized"]):
        tips.append("Pressurized/closed systems: use rated glassware/autoclaves; shield assemblies; document pressure relief; check for leaks before heating.")
    if any(has(c) for c in OPERATION_CUES["exotherm"]):
        tips.append("Exothermic addition: dose via dropping funnel or syringe pump; pre‑cool; verify heat removal capacity.")
    if any(has(c) for c in OPERATION_CUES["volatile"]):
        tips.append("Volatile solvents: work in a fume hood; verify ventilation > 6 ACH; ground/ bond during solvent transfer.")
    if any(has(c) for c in OPERATION_CUES["cryogenic"]):
        tips.append("Cryogens: wear face shield and insulated gloves; use dewars; avoid asphyxiation risks in confined spaces.")
    return tips

def dedupe_ordered(items: List[str]) -> List[str]:
    seen = []
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
        pc = pc_map.get(doc.chemical) or next((pc_map[k] for k in pc_map if k.lower()==doc.chemical.lower()), None)
        if pc and pc.ghs_hcodes:
            sds_codes = set()
            for v in points.get("hazard", []):
                sds_codes.update(re.findall(r"H\\d{3}", v))
            miss = [c for c in pc.ghs_hcodes if c not in sds_codes]
            if miss:
                ghs_notes.append(f"[{doc.chemical}] PubChem lists additional GHS codes not seen in the extracted SDS: {', '.join(sorted(set(miss)))} (verify in original documents).")
        for key, vals in points.items():
            for v in vals:
                tag = f"[{doc.chemical} – {doc.supplier}] "
                agg[key].append(tag + v)
        citations.append(f"- {doc.chemical} – {doc.url}")
    order = ["hazard", "exposure_ppe", "handling_storage", "accidental_release", "first_aid", "fire_fighting", "stability_reactivity", "toxicology"]
    md = ["# Safety Brief (Auto‑generated from SDS)"]
    md.append("\n**NOTE:** Always verify against original SDS and your local EHS policies.")
    md.append("\n## Procedure‑aware cautions")
    for t in procedure_risk_hints(proc_text):
        md.append(f"- {t}")
    if ghs_notes:
        md.append("\n## GHS cross‑check notes (PubChem)")
        for n in ghs_notes:
            md.append(f"- {n}")
    for key in order:
        pretty = key.replace("_", " ").title()
        items = dedupe_ordered(agg.get(key, []))[:25]
        if items:
            md.append(f"\n## {pretty}")
            for it in items:
                md.append(f"- {it}")
    md.append("\n## Sources")
    md.extend(citations)
    return "\n".join(md), agg

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="SDS Safety Summarizer", page_icon="🧪", layout="wide")
st.title("🧪 SDS Safety Summarizer")
st.caption("Generate a safety brief from your chemistry procedure by fetching SDS for reactants & products.")

with st.expander("How it works & disclaimer", expanded=False):
    st.markdown(
        """
        This tool searches vendor sites for SDS PDFs, extracts relevant sections, and builds a consolidated safety brief.
        SDS formats vary; always cross‑check with the original PDFs and your institution's safety rules. For critical work,
        get an EHS professional review. By using this tool, you confirm you have rights to access the referenced SDS.
        """
    )

state = AppState()

col1, col2 = st.columns([2,1])
with col1:
    state.procedure_text = st.text_area(
        "Paste your procedure (free text)",
        height=220,
        placeholder="e.g., In a 250 mL round‑bottom flask, dissolve sodium chloride in water..."
    )
with col2:
    supplier_str = st.text_input("Restrict search to suppliers (comma‑separated domains)", ", ".join(DEFAULT_SUPPLIERS))
    state.suppliers = [s.strip() for s in supplier_str.split(",") if s.strip()]
    state.use_pubchem_normalize = st.toggle("Normalize via PubChem", value=True, help="Canonical names, CID & GHS H‑codes")

# Detect chemicals
if state.procedure_text.strip():
    detected = detect_chemicals(state.procedure_text)
else:
    detected = []

st.subheader("Chemicals (edit as needed)")
chem_input = st.text_input(
    "Reactants, products, solvents (comma‑separated)",
    value=", ".join(detected),
    help="We’ll fetch SDS for these names. You can refine names for more precise matches."
)
chemical_list = [c.strip() for c in chem_input.split(",") if c.strip()]

# Optional: PubChem normalization preview
pc_map: Dict[str, PubChemInfo] = {}
if chemical_list and state.use_pubchem_normalize:
    with st.spinner("Querying PubChem for canonical names & GHS…"):
        pc_map = normalize_via_pubchem(chemical_list)
    with st.expander("PubChem normalization", expanded=False):
        for orig in chemical_list:
            info = pc_map.get(orig)
            if not info:
                continue
            title = info.canonical_name or info.iupac_name or orig
            cid = info.cid or "—"
            ghs = ", ".join(info.ghs_hcodes) if info.ghs_hcodes else "—"
            st.markdown(f"**{orig}** → *{title}* (CID: {cid}) • GHS: {ghs}")

run = st.button("🔎 Fetch SDS & Summarize", type="primary", disabled=(len(chemical_list)==0))

results_docs: List[SDSDoc] = []
status_placeholder = st.empty()

if run:
    if not SERPAPI_KEY:
        st.error("SERPAPI_KEY missing. Open the .env file in this folder and set your SerpAPI API key.")
    else:
        with st.spinner("Searching SDS and building your safety brief…"):
            for chem in chemical_list:
                status_placeholder.info(f"Searching SDS for **{chem}**…")
                hits = search_sds_serpapi(chem, state.suppliers, num=12)
                chosen = False
                for title, link in hits:
                    fpath = download_pdf(link)
                    if not fpath:
                        continue
                    text = extract_pdf_text(fpath)
                    if len(text) < 500 or ("hazard" not in text.lower() and "first aid" not in text.lower()):
                        continue
                    supplier_guess = guess_supplier_from_url(link) or "unknown"
                    results_docs.append(SDSDoc(chemical=chem, supplier=supplier_guess, url=link, filepath=fpath, text=text))
                    chosen = True
                    break
                if not chosen:
                    status_placeholder.warning(f"No SDS PDF found or downloadable for {chem}. Try adjusting the name or suppliers.")
            status_placeholder.empty()

# Render summaries
if results_docs:
    md_summary, agg = consolidate_summary(results_docs, state.procedure_text, pc_map)
    st.subheader("Consolidated Safety Summary")
    st.markdown(md_summary)

    st.download_button("💾 Download Markdown", data=md_summary.encode("utf-8"), file_name="safety_brief.md")

    st.subheader("SDS Sources Found")
    for d in results_docs:
        st.write(f"**{d.chemical}** – {d.supplier}")
        st.code(d.url)
else:
    st.info("Enter your procedure and chemicals, then click ‘Fetch SDS & Summarize’.")
