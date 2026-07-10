
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import time
import os
import re

# ─────────────────────────── CẤU HÌNH ───────────────────────────
DELAY       = 2.0   # giây giữa mỗi request
OUTPUT_FILE = "ProjectTabular/data/badminton_hvshop.csv"

# Chế độ test: True → chỉ crawl 1 dòng vợt đầu tiên, giới hạn TEST_LIMIT sản phẩm
TEST_MODE  = False
TEST_LIMIT = 5    # số sản phẩm tối đa khi test

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://hvshop.vn/",
}

# Cấu trúc: { tên_hãng: { tên_dòng: url_trang_dòng } }
# Hãng có sub-series → mỗi dòng là 1 entry
# Hãng không có sub-series → dùng tên hãng làm tên dòng duy nhất
BRAND_SERIES = {
    # ── YONEX ──
    "Yonex": {
        "Astrox":       "https://hvshop.vn/astrox/",
        "Nanoflare":    "https://hvshop.vn/nanoflare/",
        "Duora":        "https://hvshop.vn/duora/",
        "Arcsaber":     "https://hvshop.vn/arcsaber/",
        "Voltric":      "https://hvshop.vn/voltric/",
        "Nanospeed":    "https://hvshop.vn/nanospeed/",
        "Muscle Power": "https://hvshop.vn/muscle-power/",
        "Armotec":      "https://hvshop.vn/armotec/",
        "Nanoray":      "https://hvshop.vn/nanoray/",
    },
    # ── VICTOR ──
    "Victor": {
        "Thruster":     "https://hvshop.vn/thruster/",
        "Jetspeed":     "https://hvshop.vn/jetspeed-s/",
        "DriveX":       "https://hvshop.vn/drivex/",
        "Brave Sword":  "https://hvshop.vn/brave-sword/",
        "Auraspeed":    "https://hvshop.vn/victor-auraspeed/",
    },
    # ── LI-NING ──
    "Li-Ning": {
        "Axforce":      "https://hvshop.vn/axforce/",
        "Turbo Charging": "https://hvshop.vn/turbo-charging/",
        "3D Calibar":   "https://hvshop.vn/3d-calibar/",
        "Tectonic":     "https://hvshop.vn/tectonic/",
        "Halbertec":    "https://hvshop.vn/halbertec/",
        "Aeronaut":     "https://hvshop.vn/aeronaut/",
        "Windstorm":    "https://hvshop.vn/windstorm/",
        "High Carbon":  "https://hvshop.vn/high-carbon/",
        "Bladex":       "https://hvshop.vn/bladex/",
        "Lightning":    "https://hvshop.vn/lightning/",
    },
    # ── KAWASAKI (có sub-series) ──
    "Kawasaki": {
        "Kawasaki Passion":       "https://hvshop.vn/vot-cau-long-kawasaki/",
        "Kawasaki Power":         "https://hvshop.vn/vot-cau-long-kawasaki/",
        "Kawasaki Ninja":         "https://hvshop.vn/vot-cau-long-kawasaki/",
        "Kawasaki Dragon Swallow":"https://hvshop.vn/vot-cau-long-kawasaki/",
    },
    # ── MIZUNO ──
    "Mizuno": {
        "Mizuno": "https://hvshop.vn/vot-cau-long-mizuno/",
    },
    # ── APACS ──
    "Apacs": {
        "Apacs": "https://hvshop.vn/vot-cau-long-apacs/",
    },
    # ── KUMPOO ──
    "Kumpoo": {
        "Kumpoo": "https://hvshop.vn/vot-cau-long-kumpoo/",
    },
    # ── FELET ──
    "Felet": {
        "Felet": "https://hvshop.vn/vot-cau-long-felet/",
    },
    # ── FLYPOWER ──
    "Flypower": {
        "Flypower": "https://hvshop.vn/vot-cau-long-flypower/",
    },
    # ── FLEET ──
    "Fleet": {
        "Fleet": "https://hvshop.vn/vot-cau-long-fleet/",
    },
    # ── PROACE ──
    "Proace": {
        "Proace": "https://hvshop.vn/vot-cau-long-proace/",
    },
    # ── BUBADU ──
    "Bubadu": {
        "Bubadu": "https://hvshop.vn/vot-cau-long-bubadu/",
    },
    # ── VENSON (VSE) ──
    "Venson": {
        "Venson": "https://hvshop.vn/vot-cau-long-vs/",
    },
    # ── KAMITO ──
    "Kamito": {
        "Kamito": "https://hvshop.vn/vot-cau-long-kamito/",
    },
    # ── IXE ──
    "IXE": {
        "IXE": "https://hvshop.vn/vot-cau-long-ixe/",
    },
    # ── THE 3RD GAME ──
    "The 3rd Game": {
        "The 3rd Game": "https://hvshop.vn/vot-cau-long-the-3rd-game/",
    },
    # ── VICLEO ──
    "Vicleo": {
        "Vicleo": "https://hvshop.vn/vot-cau-long-vicleo/",
    },
    # Kuno
    "Kuno": {
        "Kuno": "https://hvshop.vn/vot-cau-long-kuno/",
    },
    # Adidas
    "Adidas": {
        "Adidas": "https://hvshop.vn/vot-cau-long-adidas/",
    },
}


# Ánh xạ tên field (tiếng Việt trên web → tên cột CSV)
SPEC_MAP = {
    # nơi sản xuất
    "sản xuất":        "noi_san_xuat",
    "xuất xứ":         "noi_san_xuat",
    "nơi sản xuất":    "noi_san_xuat",
    # điểm cân bằng
    "điểm cân bằng":   "diem_can_bang",
    "cân bằng":        "diem_can_bang",
    # độ cứng
    "độ cứng":         "do_cung",
    "độ cứng đũa":     "do_cung",
    # sức căng
    "sức căng":        "suc_cang",
    "lực căng":        "suc_cang",
    # vật liệu khung
    "vật liệu khung":  "vat_lieu_khung",
    "vật liệu":        "vat_lieu_khung",
    "chất liệu":       "vat_lieu_khung",
    "thân vợt":        "vat_lieu_khung",
    # màu sắc
    "màu sắc":         "mau_sac",
    "màu":             "mau_sac",
    # trọng lượng
    "trọng lượng":     "trong_luong",
    # thương hiệu (từ trang web)
    "thương hiệu":     "thuong_hieu_trang",
    # chiều dài
    "chiều dài":       "chieu_dai",
    # swing weight
    "swing weight":    "swing_weight",
    # chu vi cán
    "chu vi cán":      "chu_vi_can",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ─────────────────────────── HÀM TIỆN ÍCH ───────────────────────────

def safe_get(url: str, retries: int = 3):
    """GET có retry + delay, trả None nếu thất bại."""
    for attempt in range(retries):
        try:
            res = SESSION.get(url, timeout=30)
            if res.status_code == 200:
                return res
            print(f"  [HTTP {res.status_code}] {url}")
        except Exception as e:
            print(f"  [Lỗi lần {attempt+1}] {url}: {e}")
        time.sleep(DELAY * (attempt + 1))
    return None


def normalize_key(text: str) -> str:
    """Chuẩn hóa tên label để tra cứu."""
    return text.lower().strip().rstrip(":")


def is_product_url(href: str) -> bool:
    """
    Trả True nếu URL trỏ đến sản phẩm riêng lẻ (không phải category).
    Sản phẩm: /vot-cau-long-yonex-astrox-100va-tour/   → >= 3 từ sau prefix
    Category: /vot-cau-long-yonex/                     → chỉ 1 từ
    """
    path = re.sub(r"https?://[^/]+", "", href).strip("/")
    if not path.startswith("vot-cau-long-"):
        return False
    remainder = path[len("vot-cau-long-"):]
    return len(remainder.split("-")) >= 3


# ─────────────────────────── LẤY LINK SẢN PHẨM ───────────────────────────

def get_product_links_from_series(series_url: str, series_keyword: str = "") -> list:
    """
    Lấy tất cả link sản phẩm của một dòng vợt.
    series_keyword: chỉ giữ link có chứa từ khóa này (ví dụ: 'astrox').
    """
    kw = series_keyword.lower().replace(" ", "-")  # 'brave sword' → 'brave-sword'
    links = []
    page = 1
    while True:
        url = series_url if page == 1 else f"{series_url.rstrip('/')}/page/{page}/"
        res = safe_get(url)
        if res is None:
            break
        soup = BeautifulSoup(res.content, "lxml")
        page_links = []

        def accept(href: str) -> bool:
            """Link hợp lệ: đúng dạng sản phẩm VÀ chứa từ khóa series."""
            if href.startswith("/"):
                href = "https://hvshop.vn" + href
            if not is_product_url(href):
                return False
            if kw and kw not in href.lower():
                return False   # không phải sản phẩm của dòng này
            return True

        # Thử nhiều selector của WooCommerce product grid
        woo_links = soup.select(
            "ul.products li a.woocommerce-LoopProduct-link, "
            "ul.products li h2 a, "
            "div.products article a.woocommerce-loop-product__link, "
            "div.products .product a"
        )
        if woo_links:
            for a in woo_links:
                href = a.get("href", "")
                if accept(href):
                    page_links.append(
                        href if href.startswith("http") else "https://hvshop.vn" + href
                    )
        else:
            # Fallback: quét toàn bộ <a> nhưng BUỘC lọc theo series_keyword
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if accept(href):
                    page_links.append(
                        href if href.startswith("http") else "https://hvshop.vn" + href
                    )

        page_links = list(dict.fromkeys(page_links))
        if not page_links:
            break
        links.extend(page_links)

        # Nếu trang trả về ít hơn 2 sp → khả năng là trang cuối, dừng
        # (không cần tìm nút "next" vì class thay đổi theo theme)
        page += 1
        if page > 20:          # safety: tối đa 20 trang
            break
        time.sleep(DELAY)

    return list(dict.fromkeys(links))



# ─────────────────────────── LẤY THÔNG SỐ SẢN PHẨM ───────────────────────────

def get_product_specs(url: str) -> dict:
    """
    Crawl trang sản phẩm, lấy:
      - Tên vợt
      - Giá (ưu tiên giá sale)
      - Thông số kỹ thuật từ bảng #prod_tskt
    """
    res = safe_get(url)
    if res is None:
        return {}

    try:
        soup = BeautifulSoup(res.content, "lxml")
        data = {"url": url}

        # ---- Tên sản phẩm (bỏ prefix "Vợt Cầu Lông <Hãng> <Dòng>") ----
        h1 = soup.select_one("h1.product_title, h1.entry-title, h1")
        raw_name = h1.get_text(strip=True) if h1 else None
        if raw_name:
            # Bước 1: Xóa "Vợt Cầu Lông <Hãng>"
            cleaned = re.sub(
                r"(?i)^(set\s+)?v[oợ]t\s+c[aầ]u\s+l[oô]ng\s+"
                r"(yonex|victor|li[- ]?ning|lining|mizuno|apacs|kumpoo|"
                r"felet|flypower|fleet|kawasaki|proace|bubadu|venson|"
                r"kamito|ixe|the\s+3rd\s+game)?\s*",
                "", raw_name
            ).strip()
            # Bước 2: Xóa luôn tên dòng vợt ở đầu
            all_series = [
                # Yonex
                "astrox", "nanoflare", "duora", "arcsaber", "voltric",
                "nanospeed", "muscle power", "armotec", "nanoray",
                # Victor
                "thruster", "jetspeed", "drivex", "brave sword",
                "auraspeed", "ultramate",
                # Li-Ning
                "turbo charging", "axforce", "air force", "bladex", "wind lite",
                # Kawasaki sub-series
                "kawasaki passion", "kawasaki power",
                "kawasaki ninja", "kawasaki dragon swallow",
                # Hãng khác (dùng tên hãng làm series)
                "flypower", "fleet", "felet", "kumpoo", "mizuno",
                "apacs", "proace", "bubadu", "venson", "kamito",
                "ixe", "the 3rd game",
            ]
            for s in all_series:
                pattern = r"(?i)^" + re.escape(s) + r"\s*"
                cleaned = re.sub(pattern, "", cleaned).strip()
            data["ten_vot"] = cleaned if cleaned else raw_name
        else:
            data["ten_vot"] = None



        # ---- Giá (ưu tiên giá sale, fallback giá thường) ----
        ins_price = soup.select_one("ins .woocommerce-Price-amount bdi")
        if ins_price:
            data["gia_vnd"] = re.sub(r"[^\d]", "", ins_price.get_text()) or None
        else:
            reg_price = soup.select_one(".woocommerce-Price-amount bdi")
            if reg_price:
                data["gia_vnd"] = re.sub(r"[^\d]", "", reg_price.get_text()) or None
            else:
                data["gia_vnd"] = None

        # ---- Khởi tạo các cột thông số ----
        spec_cols = [
            "noi_san_xuat", "diem_can_bang", "do_cung", "suc_cang",
            "vat_lieu_khung", "mau_sac", "trong_luong",
            "thuong_hieu_trang", "chieu_dai", "swing_weight", "chu_vi_can"
        ]
        for c in spec_cols:
            data[c] = None

        # ---- Đọc bảng thông số ----
        # Ưu tiên: #prod_tskt → woocommerce-product-attributes
        specs_block = soup.find(id="prod_tskt")
        if specs_block:
            rows = specs_block.find_all("tr")
        else:
            tbl = soup.find("table", class_="woocommerce-product-attributes")
            rows = tbl.find_all("tr") if tbl else []

        for row in rows:
            th = row.find("th")
            td = row.find("td")
            if not th or not td:
                continue
            key = normalize_key(th.get_text(strip=True))
            val = td.get_text(separator=" ", strip=True)
            # Tra cứu trong SPEC_MAP
            for kw, col in SPEC_MAP.items():
                if kw in key:
                    data[col] = val
                    break

        return data

    except Exception as e:
        print(f"  [Lỗi parse] {url}: {e}")
        return {}


# ─────────────────────────── MAIN ───────────────────────────

def main():
    print("=" * 60)
    print("  CRAWL VỢT CẦU LÔNG HVSHOP.VN (Theo Hãng → Dòng)")
    print("=" * 60)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    all_records = []

    brands_to_crawl = BRAND_SERIES
    if TEST_MODE:
        # Chỉ lấy brand đầu tiên và series đầu tiên
        first_brand = list(BRAND_SERIES.keys())[0]
        first_series_name = list(BRAND_SERIES[first_brand].keys())[0]
        first_series_url  = BRAND_SERIES[first_brand][first_series_name]
        brands_to_crawl = {first_brand: {first_series_name: first_series_url}}
        print(f"\n[TEST MODE] Chỉ crawl: {first_brand} → {first_series_name} (tối đa {TEST_LIMIT} sp)")

    for brand, series_dict in brands_to_crawl.items():
        print(f"\n{'='*50}")
        print(f"[HÃNG] {brand}")
        for series_name, series_url in series_dict.items():
            print(f"\n  [DÒNG] {series_name}")
            print(f"         URL: {series_url}")

            # 1. Lấy danh sách link sản phẩm (lọc theo từ khóa dòng vợt)
            links = get_product_links_from_series(series_url, series_keyword=series_name)
            print(f"         → Tìm được {len(links)} sản phẩm")

            if not links:
                print("         → Bỏ qua (không có sản phẩm)")
                continue

            # Giới hạn số lượng khi test
            if TEST_MODE:
                links = links[:TEST_LIMIT]
                print(f"         → Test với {len(links)} sản phẩm đầu")

            # 2. Crawl từng sản phẩm
            for link in tqdm(links, desc=f"  {brand} / {series_name}", unit="sp", ncols=80):
                specs = get_product_specs(link)
                if specs:
                    specs["thuong_hieu"] = brand
                    specs["dong_vot"]    = series_name
                    all_records.append(specs)
                time.sleep(DELAY)



    # 3. Lưu CSV
    if all_records:
        df = pd.DataFrame(all_records)

        # Sắp xếp cột theo thứ tự ý nghĩa
        col_order = [
            "thuong_hieu", "dong_vot", "ten_vot",
            "noi_san_xuat", "diem_can_bang", "do_cung",
            "suc_cang", "vat_lieu_khung", "mau_sac",
            "trong_luong", "chieu_dai", "swing_weight", "chu_vi_can",
            "gia_vnd"
        ]
        col_order = [c for c in col_order if c in df.columns]
        df = df[col_order]
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

        print(f"\n{'='*60}")
        print(f"✅ Đã lưu {len(df)} sản phẩm → '{OUTPUT_FILE}'")
        print(f"\nThống kê theo hãng + dòng:")
        if "thuong_hieu" in df.columns and "dong_vot" in df.columns:
            print(df.groupby(["thuong_hieu", "dong_vot"]).size().to_string())
        print(f"\nMẫu dữ liệu:")
        print(df[["thuong_hieu", "dong_vot", "ten_vot",
                   "noi_san_xuat", "diem_can_bang", "do_cung",
                   "mau_sac", "gia_vnd"]].head(5).to_string())
    else:
        print("\n❌ Không có dữ liệu nào.")


if __name__ == "__main__":
    main()