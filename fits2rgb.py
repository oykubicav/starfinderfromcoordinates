import io, re, requests, numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from astroquery.simbad import Simbad

# ----------- AYARLAR ------------
coord_str = "10 45 03.5966 -59 41 05.985"   # örnek koordinat
radius_arcmin = 5
fov_deg = radius_arcmin / 60

# ----------- GÖRSELİ AL ----------
coord = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
ra_deg, dec_deg = coord.ra.deg, coord.dec.deg

query_url = (
    "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
    f"?Interface=quick&Position={ra_deg:.6f}%2C{dec_deg:.6f}"
    "&Survey=Digitized+Sky+Survey"
    f"&Size={2*fov_deg:.3f}&Pixels=600&float=on"
)
html = requests.get(query_url).text
soup = BeautifulSoup(html, "html.parser")
fits_link = next(urljoin("https://skyview.gsfc.nasa.gov/current/cgi/", a["href"])
                 for a in soup.find_all("a", href=True) if a["href"].endswith(".fits"))

hdu = fits.open(io.BytesIO(requests.get(fits_link).content))[0]
data, wcs = hdu.data, WCS(hdu.header)

# ----------- YILDIZLARI GETİR ---------
simbad = Simbad()
simbad.ROW_LIMIT = -1
simbad.add_votable_fields("ra", "dec", "ra(d)", "dec(d)", "otype")

res = simbad.query_region(coord, radius_arcmin * u.arcmin)
if res is None or len(res) == 0:
    raise ValueError("No SIMBAD objects found.")

# ----------- KOORDİNAT HESAPLA --------
ra_col = "RA_d" if "RA_d" in res.colnames else "RA"
dec_col = "DEC_d" if "DEC_d" in res.colnames else "DEC"

skycoords = SkyCoord(res[ra_col], res[dec_col], unit=(u.deg, u.deg))


separ = coord.separation(skycoords).arcsec
res["DIST"] = separ

# İsteğe bağlı: sadece 50 arcsec içindekileri al
mask_50 = separ <= 50
res = res[mask_50]
skycoords = skycoords[mask_50]

# Sort by distance
order = np.argsort(res["DIST"])
res = res[order]
skycoords = skycoords[order]

# ----------- PX COORD ÇEVİR ------------
px, py = wcs.world_to_pixel(skycoords)

# ----------- GÖRSELİ ÇİZ ---------------
plt.figure(figsize=(6, 6))
img = np.arcsinh(data)
plt.imshow(img, cmap="gray", origin="lower")
plt.scatter(px, py, s=50, edgecolors='red', facecolors='none', label="Stars")

# İsim yaz
for i, name in enumerate(res["MAIN_ID"]):
    plt.text(px[i] + 5, py[i] + 5, str(i+1), color="red", fontsize=8)

plt.title("SIMBAD Stars on SkyView Image")
plt.axis("off")
plt.tight_layout()
plt.show()
