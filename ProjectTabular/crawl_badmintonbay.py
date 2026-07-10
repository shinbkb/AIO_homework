import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import time
import os
import re

# ─────────────────────────── CẤU HÌNH ───────────────────────────

DELAY       = 0.5          # giây giữa 2 request
OUTPUT_FILE = "ProjectTabular/data/badmintonbay.csv"
BASE_URL    = "https://www.badmintonbay.com"

# Chế độ test: True → chỉ crawl series đầu tiên, giới hạn TEST_LIMIT sản phẩm
TEST_MODE  = False
TEST_LIMIT = 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.badmintonbay.com/",
}

# Cấu trúc: { tên_hãng: { tên_dòng: url_trang_dòng } }
BRAND_SERIES = {
    "Yonex": {
        "ArcSaber":  "https://www.badmintonbay.com/Badminton-Racket/Yonex-Badminton-Racket/Yonex-ArcSaber-Badminton-Racket/",
        "Astrox":    "https://www.badmintonbay.com/Badminton-Racket/Yonex-Badminton-Racket/Yonex-Astrox-Badminton-Racket/",
        "Duora":     "https://www.badmintonbay.com/Badminton-Racket/Yonex-Badminton-Racket/Yonex-Duora-Badminton-Racket/",
        "Nanoflare": "https://www.badmintonbay.com/Badminton-Racket/Yonex-Badminton-Racket/Yonex-Nanoflare-Badminton-Racket/",
        "NanoRay":   "https://www.badmintonbay.com/Badminton-Racket/Yonex-Badminton-Racket/Yonex-NanoRay-Badminton-Racket/",
        "Voltric":   "https://www.badmintonbay.com/Badminton-Racket/Yonex-Badminton-Racket/Yonex-Voltric-Badminton-Racket/",
        "Carbonex":  "https://www.badmintonbay.com/Badminton-Racket/Yonex-Badminton-Racket/Yonex-Carbonex-Badminton-Racket/",
    },
    "Li-Ning": {
        "Li-Ning": "https://www.badmintonbay.com/Badminton-Racket/Li-Ning-Badminton-Racket/",
    },
    "Apacs": {
        "Apacs": "https://www.badmintonbay.com/Badminton-Racket/Apacs-Badminton-Racket/",
    },
    "RSL": {
        "RSL": "https://www.badmintonbay.com/Badminton-Racket/RSL-Badminton-Racket/",
    },
    "Hundred": {
        "Hundred": "https://www.badmintonbay.com/Badminton-Racket/Hundred-Badminton-Racket/",
    },
    "Abroz": {
        "Abroz": "https://www.badmintonbay.com/Badminton-Racket/Abroz-Badminton-Racket/",
    },
    "Maxx": {
        "Maxx": "https://www.badmintonbay.com/Badminton-Racket/Maxx-Badminton-Racket/",
    },
    "Flex-Power": {
        "Flex-Power": "https://www.badmintonbay.com/Badminton-Racket/Flex-Power-Badminton-Racket/",
    },   
}

# ─────────────────────────── SESSION ───────────────────────────

session = requests.Session()
session.headers.update(HEADERS)


def safe_get(url: str, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            res = session.get(url, timeout=20)
            if res.status_code == 200:
                return res
            if res.status_code == 404:
                return None
        except Exception as e:
            if attempt == retries - 1:
                print(f"\n  [Lỗi request] {url}: {e}")
        time.sleep(DELAY)
    return None


# ─────────────────────────── LẤY LINK SẢN PHẨM ───────────────────────────

# Các brand có sản phẩm trực tiếp dưới brand page (không có sub-series)
_BRAND_ONLY = {"li-ning", "apacs", "mizuno", "kawasaki", "felet",
               "hundred", "maxx", "rsl", "abroz", "flex-power"}


def is_product_url(href: str) -> bool:
    """
    URL sản phẩm trên badmintonbay.com kết thúc bằng .html
    và nằm ít nhất 2 cấp dưới /Badminton-Racket/.
    - Yonex (4 cấp): /Badminton-Racket/Yonex.../Yonex-Astrox.../product.html
    - Li-Ning (3 cấp): /Badminton-Racket/Li-Ning.../product.html
    """
    return bool(re.search(r'/Badminton-Racket/[^/]+/[^/]+\.html$', href)) or \
           bool(re.search(r'/Badminton-Racket/[^/]+/[^/]+/[^/]+\.html$', href))


def get_product_links_from_series(series_url: str, series_keyword: str = "") -> list:
    """
    Lấy tất cả link sản phẩm của 1 dòng vợt.
    Pagination: ?page=2, ?page=3, ...
    """
    kw = series_keyword.lower().replace(" ", "-")
    links = []
    page = 1

    while True:
        url = series_url if page == 1 else f"{series_url}?page={page}"
        res = safe_get(url)
        if res is None:
            break

        soup = BeautifulSoup(res.content, "lxml")
        page_links = []

        # Selector chính: product card links
        product_anchors = soup.select(
            "div.product-layout a.product-img, "
            "div.product-thumb a, "
            "div.product-layout h4 a"
        )

        # Fallback: tất cả <a> href .html trong nội dung
        if not product_anchors:
            product_anchors = [
                a for a in soup.find_all("a", href=True)
                if a["href"].endswith(".html") and "Badminton-Racket" in a["href"]
            ]

        seen = set()
        for a in product_anchors:
            href = a.get("href", "")
            if not href.startswith("http"):
                href = BASE_URL + href
            if is_product_url(href) and href not in seen:
                # Lọc theo series keyword nếu có
                # Bỏ qua filter nếu keyword là tên hãng (không có sub-series)
                if kw and kw not in _BRAND_ONLY and kw not in href.lower():
                    continue
                seen.add(href)
                page_links.append(href)

        if not page_links:
            break

        links.extend(page_links)
        page += 1
        if page > 20:
            break
        time.sleep(DELAY)

    return list(dict.fromkeys(links))


# ─────────────────────────── LẤY SPEC SẢN PHẨM ───────────────────────────

def clean_price(text: str) -> str | None:
    """Chuyển 'US$12.50' → '12.50'"""
    m = re.search(r'[\d,\.]+', text.replace(",", ""))
    return m.group() if m else None


def get_product_specs(url: str) -> dict:
    """
    Crawl trang sản phẩm trên badmintonbay.com:
    - Tên vợt (h1 / h2 .product-name)
    - Giá (USD)
    - Thông số: Weight, Balance Point, Shaft Flex, String Tension,
                Frame Material, Color, Made In
    """
    res = safe_get(url)
    if res is None:
        return {}

    try:
        soup = BeautifulSoup(res.content, "lxml")
        data = {}

        # ── Tên ──
        h1 = soup.select_one("h1.product-title, h2.product-name, h1, h2")
        raw_name = h1.get_text(strip=True) if h1 else None
        if raw_name:
            # Xóa tiền tố "Yonex -", "Victor -", "Li-Ning -", "Apacs -" ...
            cleaned = re.sub(
                r"(?i)^(yonex|victor|li[- ]ning|apacs|mizuno|kawasaki|"
                r"lining|felet|flypower|fleet|kumpoo)\s*[-–]\s*",
                "", raw_name
            ).strip()
            # Xóa tiền tố dòng vợt (Astrox, Nanoflare, ...)
            all_series = [
                "astrox", "nanoflare", "arcsaber", "duora", "voltric",
                "nanoray", "carbonex", "thruster", "jetspeed", "drivex",
                "brave sword", "auraspeed",
            ]
            for s in all_series:
                cleaned = re.sub(r"(?i)^" + re.escape(s) + r"\s*", "", cleaned).strip()
            data["racket_name"] = cleaned if cleaned else raw_name
        else:
            data["racket_name"] = None

        # ── Giá (USD) ──
        price_el = soup.select_one(".price-new, li.price-new, .product-price .price-new")
        if not price_el:
            price_el = soup.select_one(".price")
        data["price_usd"] = clean_price(price_el.get_text()) if price_el else None

        # ── Thông số kỹ thuật ──
        spec_fields = {
            "racket_name": None,
            "weight":       None,
            "balance_point":None,
            "shaft_flex":   None,
            "string_tension":None,
            "frame_material":None,
            "color":        None,
            "made_in":      None,
        }

        # Tìm bảng spec: thường trong tab "Specifications"
        spec_table = None
        tabs = soup.find_all("div", class_=re.compile(r"tab-content|product-tab", re.I))
        for tab in tabs:
            tbl = tab.find("table")
            if tbl:
                spec_table = tbl
                break

        if not spec_table:
            spec_table = soup.find("table")

        if spec_table:
            for row in spec_table.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) < 2:
                    continue
                key = cells[0].get_text(strip=True).lower()
                val = cells[1].get_text(strip=True)

                if "weight" in key:
                    data["weight"] = val
                elif "balance" in key:
                    data["balance_point"] = val
                elif "flex" in key or "shaft" in key:
                    data["shaft_flex"] = val
                elif "tension" in key or "string" in key:
                    data["string_tension"] = val
                elif "frame" in key or "material" in key:
                    data["frame_material"] = val
                elif "color" in key or "colour" in key:
                    data["color"] = val
                elif "made" in key or "origin" in key or "manufactur" in key:
                    data["made_in"] = val

        return data

    except Exception as e:
        print(f"\n  [Lỗi parse] {url}: {e}")
        return {}


# ─────────────────────────── MAIN ───────────────────────────

def main():
    print("=" * 60)
    print("  CRAWL VỢT CẦU LÔNG BADMINTONBAY.COM")
    print("=" * 60)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    all_records = []

    brands_to_crawl = BRAND_SERIES
    if TEST_MODE:
        first_brand = list(BRAND_SERIES.keys())[0]
        first_series_name = list(BRAND_SERIES[first_brand].keys())[0]
        first_series_url  = BRAND_SERIES[first_brand][first_series_name]
        brands_to_crawl = {first_brand: {first_series_name: first_series_url}}
        print(f"\n[TEST MODE] Chỉ crawl: {first_brand} → {first_series_name} (tối đa {TEST_LIMIT} sp)")

    for brand, series_dict in brands_to_crawl.items():
        print(f"\n{'='*50}")
        print(f"[BRAND] {brand}")
        for series_name, series_url in series_dict.items():
            print(f"\n  [SERIES] {series_name}")
            print(f"           URL: {series_url}")

            # 1. Lấy link sản phẩm
            links = get_product_links_from_series(series_url, series_keyword=series_name)
            print(f"           → Found {len(links)} products")

            if not links:
                print("           → Skipped (no products found)")
                continue

            if TEST_MODE:
                links = links[:TEST_LIMIT]
                print(f"           → Testing with {len(links)} products")

            # 2. Crawl từng sản phẩm
            for link in tqdm(links, desc=f"  {brand} / {series_name}", unit="racket", ncols=80):
                specs = get_product_specs(link)
                if specs:
                    specs["brand"]  = brand
                    specs["series"] = series_name
                    all_records.append(specs)
                time.sleep(DELAY)

    # 3. Lưu CSV
    if all_records:
        df = pd.DataFrame(all_records)

        col_order = [
            "brand", "series", "racket_name",
            "weight", "balance_point", "shaft_flex",
            "string_tension", "frame_material", "color",
            "made_in", "price_usd",
        ]
        col_order = [c for c in col_order if c in df.columns]
        df = df[col_order]
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

        print(f"\n{'='*60}")
        print(f"✅ Saved {len(df)} rackets → '{OUTPUT_FILE}'")
        print(f"\nBy brand & series:")
        if "brand" in df.columns and "series" in df.columns:
            print(df.groupby(["brand", "series"]).size().to_string())
        print(f"\nSample data:")
        print(df[["brand", "series", "racket_name",
                   "balance_point", "shaft_flex", "price_usd"]].head(5).to_string())
    else:
        print("\n❌ No data collected.")


if __name__ == "__main__":
    main()
