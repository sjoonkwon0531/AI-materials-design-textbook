"""Exercise 3.7.E (a): template patch IF a Korean OPTIMADE endpoint is ever published.
As checked on 2026-10-01, providers.optimade.org/providers.json lists no Korean provider
and kmds.re.kr documents no Open/REST/OPTIMADE API, so the URL below is a placeholder.
"""
providers = {
    "Materials Project": "https://optimade.materialsproject.org/v1/structures",
    "OQMD":              "https://oqmd.org/optimade/v1/structures",
    "JARVIS":            "https://jarvis.nist.gov/optimade/jarvisdft/v1/structures",
}
KMDS_BASE = None   # e.g. "https://<kmds-optimade-host>/v1" once announced (check /info first)
if KMDS_BASE:
    providers["K-MDS"] = KMDS_BASE.rstrip("/") + "/structures"
print(list(providers))
