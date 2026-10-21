"""Text normalisation for business names and addresses.

Everything here is deterministic and country-aware through small per-country
dictionaries. Unknown countries (e.g. France, which never appears in training)
fall back to generic rules, so the country label stays an open set.
"""
from __future__ import annotations

import re
import unicodedata

from anyascii import anyascii

# --------------------------------------------------------------------------- #
# Static vocabularies
# --------------------------------------------------------------------------- #
LEGAL_MAP = {
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cos": "co",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "lmited": "ltd",
    "pvt": "pvt", "private": "pvt", "pvtltd": "pvt",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pc": "pc", "pllc": "pllc",
    "pa": "pa", "opc": "opc", "psc": "psc", "lllp": "lllp",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scp": "scp", "selarl": "selarl", "scop": "scop", "scm": "scm",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
}
NAME_STOP = {"the", "m/s", "ms"}

STREET_COMMON = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "dr": "drive", "drv": "drive", "ln": "lane", "ct": "court", "crt": "court",
    "cir": "circle", "pl": "place", "pkwy": "parkway", "hwy": "highway", "trl": "trail",
    "ter": "terrace", "terr": "terrace", "sq": "square", "apt": "apartment", "apts": "apartments",
    "fl": "floor", "flr": "floor", "bldg": "building", "nr": "near", "opp": "opposite",
    "ste": "suite", "rm": "room", "cres": "crescent", "cv": "cove", "pt": "point", "mt": "mount",
    "ft": "fort", "jct": "junction", "extn": "extension", "ext": "extension", "sec": "sector",
    "dist": "district", "tq": "taluka", "tal": "taluka", "stn": "station", "ngr": "nagar",
    "rly": "railway", "hsg": "housing", "soc": "society", "indl": "industrial", "ind": "industrial",
    "est": "estate", "cplx": "complex", "mkt": "market", "gf": "ground floor", "ff": "first floor",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "mg": "mg",
}
STREET_FR = {
    "r": "rue", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard", "av": "avenue",
    "ave": "avenue", "pl": "place", "imp": "impasse", "ch": "chemin", "che": "chemin",
    "rte": "route", "all": "allee", "st": "saint", "ste": "sainte", "fg": "faubourg",
    "fbg": "faubourg", "qu": "quai", "sq": "square", "res": "residence", "bat": "batiment",
    "crs": "cours", "pass": "passage", "prom": "promenade", "za": "za", "zi": "zi",
}
ADDR_DROP = {"no", "number", "num", "nos", "hno", "dno", "door", "dor", "h", "d", "null", "none", "nan"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk",
    "jammu kashmir": "jk", "ladakh": "la", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld",
}
_IN_EXTRA_CODES = {"ts": "tg", "or": "od", "ch": "ch", "cg": "cg", "ct": "cg", "uk": "uk", "ut": "uk",
                   "dl": "dl", "tn": "tn", "ka": "ka", "mh": "mh", "up": "up", "wb": "wb", "gj": "gj",
                   "rj": "rj", "mp": "mp", "hr": "hr", "pb": "pb", "kl": "kl", "ap": "ap", "tg": "tg",
                   "br": "br", "jh": "jh", "od": "od", "as": "as", "ga": "ga", "hp": "hp", "jk": "jk",
                   "py": "py"}


def _state_table(names: dict, extra_codes: dict | None = None) -> dict:
    t = dict(names)
    for code in set(names.values()):
        t[code] = code
    if extra_codes:
        t.update(extra_codes)
    return t


STATE_TABLES = {
    "us": _state_table(US_STATES),
    "india": _state_table(IN_STATES, _IN_EXTRA_CODES),
}

STREET_WORDS = set(STREET_COMMON.values()) | set(STREET_FR.values()) | {
    "road", "street", "marg", "gali", "lane", "nagar", "colony", "sector", "block", "phase", "plot",
    "flat", "floor", "tower", "wing", "near", "opposite", "behind", "building", "complex", "house",
    "rue", "chemin", "impasse", "allee", "avenue", "boulevard", "unit", "suite", "po", "box", "pmb",
    "apartment", "apartments", "room", "shop", "office", "estate", "area", "industrial", "main",
    "cross", "layout", "stage", "extension", "park", "plaza", "center", "centre", "market",
}

NON_LATIN_RE = re.compile(r"[^\u0000-ɏḀ-ỿ -⁯]")
SPLIT_RE = re.compile(r"[\s,.;:()\[\]{}<>|/\\#\-_'\"!?*+=~`^$%@]+")
ALNUM_RE = re.compile(r"[a-z0-9]+")
NUM_RE = re.compile(r"\d+")
WEB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]{1,62})\.(?:co\.in|org\.in|com|net|org|in|co|fr|io|biz|info|us)\b")
ALIAS_RE = re.compile(
    r"\b(?:f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka|formerly known as|formerly|d\s*/\s*b\s*/\s*a|dba|"
    r"doing business as|t\s*/\s*a|trading as)\b")
CITY_PREFIX_RE = re.compile(r"^(?:city|village|town|township|borough|municipality) of ")


def country_key(country: str) -> str:
    return (country or "").strip().lower()


def has_non_latin(s: str) -> bool:
    return bool(s) and NON_LATIN_RE.search(s) is not None


# --------------------------------------------------------------------------- #
# Transliteration with a learned dictionary
# --------------------------------------------------------------------------- #
def raw_tokens(s: str) -> list[str]:
    return [t for t in SPLIT_RE.split(s) if t]


def translit(s: str, token_dict: dict) -> str:
    """Map non-Latin tokens through the learned dictionary, then anyascii the rest."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    if has_non_latin(s):
        out = []
        for tok in s.split():
            core = tok.strip(",.;:()[]{}<>|#-_'\"!?")
            if core and has_non_latin(core):
                rep = token_dict.get(core)
                if rep is not None:
                    tok = tok.replace(core, " " + rep + " ")
            out.append(tok)
        s = " ".join(out)
    return anyascii(s).lower()


def _collapse_initials(tokens: list[str]) -> list[str]:
    """'l l c' -> 'llc', 's a s' -> 'sas' (runs of single letters become one token)."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


def _dedupe_adjacent(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def _name_tokens(s: str) -> list[str]:
    s = s.replace("&", " and ")
    toks = ALNUM_RE.findall(s)
    toks = _collapse_initials(toks)
    return _dedupe_adjacent(toks)


def _split_legal(tokens: list[str]) -> tuple[list[str], list[str], list[str]]:
    full, core, legal = [], [], []
    for t in tokens:
        lt = LEGAL_MAP.get(t)
        if lt:
            full.append(lt)
            legal.append(lt)
        else:
            full.append(t)
            if t not in NAME_STOP:
                core.append(t)
    if not core:
        core = [t for t in full]
    return full, core, sorted(set(legal))


# --------------------------------------------------------------------------- #
# Normaliser
# --------------------------------------------------------------------------- #
class Normalizer:
    def __init__(self, token_dict: dict | None = None, comp_dict: dict | None = None,
                 gazetteer: dict | None = None):
        self.token_dict = token_dict or {}
        self.comp_dict = comp_dict or {}
        # gazetteer: country_key -> set(city strings)
        self.gazetteer = {k: set(v) for k, v in (gazetteer or {}).items()}

    # ----- names ---------------------------------------------------------- #
    def name(self, raw: str) -> dict:
        s = translit(raw or "", self.token_dict)
        web = ""
        m = WEB_RE.search(s)
        if m:
            web = m.group(1).replace("-", "")
            s = WEB_RE.sub(" ", s)
        s = s.replace("www", " ")
        alt = ""
        parts = ALIAS_RE.split(s, maxsplit=1)
        if len(parts) == 2 and ALNUM_RE.search(parts[0]) and ALNUM_RE.search(parts[1]):
            s, alt = parts[0], parts[1]
        full, core, legal = _split_legal(_name_tokens(s))
        alt_core = ""
        if alt:
            _, a_core, a_legal = _split_legal(_name_tokens(alt))
            alt_core = " ".join(a_core)
            legal = sorted(set(legal) | set(a_legal))
        if not full and web:            # the name was only a website
            full, core = [web], [web]
        return {
            "name_n": " ".join(full),
            "core": " ".join(core),
            "alt": alt_core,
            "legal": " ".join(legal),
            "web": web,
        }

    # ----- addresses ------------------------------------------------------- #
    def addr_components(self, raw: str, country: str) -> list[list[str]]:
        """Normalised token lists, one per comma-separated component (empty ones dropped)."""
        if not raw:
            return []
        ck = country_key(country)
        street = STREET_FR if ck == "france" else STREET_COMMON
        states = STATE_TABLES.get(ck, {})
        comps = []
        for comp in unicodedata.normalize("NFKC", raw).split(","):
            c = comp.strip()
            if not c:
                continue
            if has_non_latin(c) and c in self.comp_dict:
                c = self.comp_dict[c]
            c = translit(c, self.token_dict)
            raw_toks = [t for t in ALNUM_RE.findall(c) if t not in ADDR_DROP]
            if raw_toks and " ".join(raw_toks) in states:      # 'CT' is a state here, not 'court'
                comps.append(raw_toks)
                continue
            toks = []
            for t in raw_toks:
                if t.isdigit():
                    t = t.lstrip("0") or "0"
                elif t[0].isdigit():                 # '1a', '22nd' -> keep, strip zeros
                    t = t.lstrip("0") or t
                exp = street.get(t)
                if exp:
                    toks.extend(exp.split())
                else:
                    toks.append(t)
            if toks:
                comps.append(toks)
        return comps

    def _city_of(self, comp_str: str, ck: str) -> str:
        gz = self.gazetteer.get(ck)
        if not gz:
            return ""
        c = CITY_PREFIX_RE.sub("", comp_str)
        if c in gz:
            return c
        for suf in (" city", " cdp", " town", " village"):
            if c.endswith(suf) and c[: -len(suf)] in gz:
                return c[: -len(suf)]
        return ""

    def address(self, raw: str, country: str) -> dict:
        raw = (raw or "").strip()
        if raw.lower() in ("", "null", "none", "nan"):
            return {"addr_n": "", "nums": "", "cities": "", "state": "", "street": ""}
        ck = country_key(country)
        states = STATE_TABLES.get(ck, {})
        comps = self.addr_components(raw, country)
        out_toks, nums, cities, street_toks = [], [], [], []
        state = ""
        for toks in comps:
            cs = " ".join(toks)
            st = states.get(cs)
            if st and not state:
                state = st
                out_toks.append("st" + st)
                continue
            city = self._city_of(cs, ck)
            if city:
                cities.append(city)
                out_toks.extend(city.split())
                continue
            for t in toks:
                if any(ch.isdigit() for ch in t):          # '4607', '1a', 'e9355', 'no6'
                    nums.extend(NUM_RE.findall(t))
                elif len(t) >= 3 and t not in STREET_WORDS:
                    street_toks.append(t)
            out_toks.extend(toks)
        nums = [n.lstrip("0") or "0" for n in nums]
        seen = set()
        nums = [n for n in nums if not (n in seen or seen.add(n))]
        return {
            "addr_n": " ".join(out_toks),
            "nums": " ".join(nums),
            "cities": "|".join(dict.fromkeys(cities)),
            "state": state,
            "street": " ".join(dict.fromkeys(street_toks)),
        }

    # ----- full record -------------------------------------------------------- #
    def record(self, name_raw: str, addr_raw: str, country: str) -> dict:
        r = self.name(name_raw)
        r.update(self.address(addr_raw, country))
        r["script"] = int(has_non_latin(name_raw))
        r["addr_empty"] = int(r["addr_n"] == "")
        r["keys"] = "\x1f".join(blocking_keys(r))
        return r


def compact(s: str) -> str:
    return s.replace(" ", "")


def blocking_keys(r: dict) -> list[str]:
    """Exact keys used by the key-blocking channels (namespaced by type)."""
    keys = []
    for nm in (r["core"], r["alt"], r["web"]):
        c = compact(nm)
        if len(c) >= 3:
            keys.append("n:" + c)
    nums = r["nums"].split()[:4]
    if nums:
        for city in (r["cities"].split("|") if r["cities"] else []):
            for n in nums:
                keys.append(f"a:{n}|{city}")
        for tok in r["street"].split()[:2]:
            for n in nums:
                keys.append(f"s:{n}|{tok}")
    return list(dict.fromkeys(keys))


# --------------------------------------------------------------------------- #
# Worker entry points (module-level so they pickle on Windows spawn)
# --------------------------------------------------------------------------- #
_WORKER_NORM: Normalizer | None = None


def init_worker(token_dict, comp_dict, gazetteer):
    global _WORKER_NORM
    _WORKER_NORM = Normalizer(token_dict, comp_dict, gazetteer)


NORM_COLS = ["name_n", "core", "alt", "legal", "web", "addr_n", "nums", "cities", "state", "street",
             "script", "addr_empty", "keys"]


def normalize_chunk(rows: list[tuple[str, str, str]]) -> dict:
    """Columnar output (dict of lists) - far smaller than a list of dicts when pickled."""
    assert _WORKER_NORM is not None
    out = {c: [] for c in NORM_COLS}
    for n, a, c in rows:
        r = _WORKER_NORM.record(n, a, c)
        for k in NORM_COLS:
            out[k].append(r[k])
    return out


def components_chunk(rows: list[tuple[str, str]]) -> list[list[str]]:
    """Address components as strings (used to build the city gazetteer)."""
    assert _WORKER_NORM is not None
    return [[" ".join(t) for t in _WORKER_NORM.addr_components(a, c)] for a, c in rows]
