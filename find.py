import io, re, requests, numpy as np, streamlit as st
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import matplotlib.pyplot as plt
from astroquery.simbad import Simbad

st.set_page_config(page_title="SkyView + SIMBAD", layout="centered")
st.title("Star Finder From Coordinates")

# ---------- helpers ----------
def parse_coord(s: str) -> SkyCoord:
    parts = s.strip().split()
    if len(parts) == 2 and all(re.fullmatch(r"-?\d+(\.\d+)?", p) for p in parts):
        return SkyCoord(float(parts[0]), float(parts[1]), unit='deg')
    return SkyCoord(s, unit=(u.hourangle, u.deg))

def get_skycoords(table):
    cols_lower = [c.lower() for c in table.colnames]
    colmap = {c.lower(): c for c in table.colnames}
    if "ra_d" in cols_lower and "dec_d" in cols_lower:
        return SkyCoord(table[colmap["ra_d"]], table[colmap["dec_d"]], unit=(u.deg, u.deg))
    elif "ra" in cols_lower and "dec" in cols_lower:
        return SkyCoord(table[colmap["ra"]], table[colmap["dec"]], unit=(u.deg, u.deg))
    elif "ra" in table.colnames and "dec" in table.colnames:
        return SkyCoord(table["RA"], table["DEC"], unit=(u.hourangle, u.deg))
    else:
        raise KeyError("RA/DEC columns not found in SIMBAD result")

# ---------- UI ----------
coord_input = st.text_input("Coordinates (RA DEC)",
                            "10 45 03.5966 -59 41 05.985",
                            help="Examples: 10 45 03.59 -59 41 05.98   •   161.265 -59.685")
radius_arcmin = st.slider("Cone radius (arcmin)", 1, 30, 5)

# ---------- main ----------
if st.button("Fetch image & stars"):
    try:
        coord = parse_coord(coord_input)
    except Exception as e:
        st.error(f"Invalid coordinate: {e}")
        st.stop()

    ra_deg, dec_deg = coord.ra.deg, coord.dec.deg
    fov_deg = radius_arcmin / 60

    with st.spinner("Opening SkyView image…"):
        query_url = ("https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
                     f"?Interface=quick&Position={ra_deg:.6f}%2C{dec_deg:.6f}"
                     "&Survey=Digitized+Sky+Survey"
                     f"&Size={2*fov_deg:.3f}&Pixels=600&float=on")
        html = requests.get(query_url, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")
        fits_link = next(urljoin("https://skyview.gsfc.nasa.gov/current/cgi/", a["href"])
                         for a in soup.find_all("a", href=True) if a["href"].endswith(".fits"))
        hdu = fits.open(io.BytesIO(requests.get(fits_link, timeout=30).content))[0]
        data, wcs = hdu.data, WCS(hdu.header)

    st.subheader("SkyView Image")
    img = np.arcsinh(data); img = (img-img.min())/np.ptp(img)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="gray", origin="lower")
    ax.axis("off")
    st.pyplot(fig)

    with st.spinner("Querying SIMBAD"):
        sim = Simbad(); sim.ROW_LIMIT = -1
        sim.add_votable_fields("ra", "dec", "ra(d)", "dec(d)", "flux(V)", "otype", "otype_txt")
        res = sim.query_region(coord, radius_arcmin * u.arcmin)
        if res is None or len(res) == 0:
            st.warning("No objects returned"); st.stop()

        otype_col = next((c for c in ("OTYPE", "OTYPE_S") if c in res.colnames), None)
        stars = res if otype_col is None else res[res[otype_col].astype(str)
                                                  .str.contains(r"\bStar\b|\*$", regex=True)]

    try:
        skycoords = get_skycoords(stars)
        separ = coord.separation(skycoords).arcsec
        stars["DIST"] = separ

        # Filter to DIST <= 50 arcsec
        mask_50 = separ <= 50
        stars = stars[mask_50]
        skycoords = skycoords[mask_50]
        separ = separ[mask_50]

        order = np.argsort(separ)
        stars = stars[order]
        skycoords = skycoords[order]

        px, py = wcs.world_to_pixel(skycoords)
        
        px, py = px[order], py[order]

    except Exception as e:
        st.error(f"Could not compute pixel coords / distance: {e}")
        st.dataframe(stars.to_pandas())
        st.stop()

    main_id_col = next((col for col in stars.colnames if col.lower() == "main_id"), None)
    if main_id_col is None:
        st.error("MAIN_ID column not found in SIMBAD result.")
        st.dataframe(stars.to_pandas())
        st.stop()
    

    st.subheader("Star List")
    ra_col = next((c for c in stars.colnames if c.lower() in ("ra", "ra_d")), None)
    dec_col = next((c for c in stars.colnames if c.lower() in ("dec", "dec_d")), None)
    cols = [main_id_col, "DIST", ra_col, dec_col]
    if "FLUX_V" in stars.colnames:
        cols.append("FLUX_V")
    df = stars.to_pandas()[cols]
    df.columns = ["Name", "Dist  (arcsec)", "RA (deg)", "DEC (deg)"] + (["V‑mag"] if "FLUX_V" in stars.colnames else [])
    st.dataframe(df, use_container_width=True)
