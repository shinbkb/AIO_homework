
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import time
import os
import re

# ─────────────────────────── CẤU HÌNH ───────────────────────────

DELAY       = 0.5   # giây giữa mỗi request
BASE_URL    = "https://shopvnb.com"
OUTPUT_FILE = "ProjectTabular/data/shopvnb.csv"

TEST_MODE  = True
TEST_LIMIT = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi,en;q=0.9,en-US;q=0.8",
    "Accept": "text/html, */*; q=0.01",
    "Referer": "https://shopvnb.com/",
    "X-Requested-With": "XMLHttpRequest",   
}

# URL filter: https://shopvnb.com/vot-cau-long?thuong_hieu=<id>&page=N&is_ajax=1
# ID lấy từ filter checkbox trên trang danh mục /vot-cau-long
BRAND_IDS = {
    "Yonex": "8",
    "Victor": "10",
    "Li-Ning": "90",
    "Mizuno": "34",
    "Kawasaki": "14",
    "Apacs": "15",
    "Kumpoo": "13",
    "Felet": "64",
    "Flypower": "38",
    "Fleet": "48",
    "Proace": "20",
    "Forza": "11",
    "VS": "24",
    "VNB": "43",
    "Kamito": "61",
    "Adidas": "21",
    "Vicleo": "77",
    "The 3rd Game": "63",
    "Taro":"90",
    "Adonex" :"16",
    "Babolat":"17",
    "Pebble Beach":"19",
    "DonexPro":"25",
    "Joto":"31",
    "Iron Man":"32",
    "Ashaway":"33",
    "Sunbatta":"36",
    "Paramount":"39",
    "Tenway":"44",
    "FYKYMI":"46",
    "FUKYMI":"47",
    "Maxta":"51",
    "Lotus":"52",
    "Pro Kennex":"55",
    "Protech":"12",
    "Victec":"60",
    "Jogarbola":"68",
    "Yuko":"70",
    "Kolt":"72",
    "Gosen":"110",
    "IXE":"113",
    "Kaiwin":"120",
    "Gamicy":"121",
    "Redson":"127",
    "Hundred":"136"
}

SPEC_MAP = {
    # thương hiệu
    "thương hiệu":          "thuong_hieu_trang",
    # nơi sản xuất / xuất xứ
    "nơi sản xuất":         "noi_san_xuat",
    "xuất xứ":              "noi_san_xuat",
    "sản xuất":             "noi_san_xuat",
    # điểm / độ cân bằng
    "điểm cân bằng":        "diem_can_bang",
    "độ cân bằng":          "diem_can_bang",
    "cân bằng":             "diem_can_bang",
    # độ cứng đũa
    "độ cứng":              "do_cung",
    "độ cứng đũa":          "do_cung",
    # sức căng / mức căng
    "sức căng":             "suc_cang",
    "mức căng dây tối đa":  "suc_cang",
    "mức căng":             "suc_cang",
    "lực căng":             "suc_cang",
    # vật liệu / chất liệu khung
    "vật liệu khung":       "vat_lieu_khung",
    "chất liệu":            "vat_lieu_khung",
    "vật liệu":             "vat_lieu_khung",
    # màu sắc
    "màu sắc":              "mau_sac",
    "màu":                  "mau_sac",
    # trọng lượng
    "trọng lượng":          "trong_luong",
    # chiều dài
    "chiều dài":            "chieu_dai",
    # chu vi cán
    "chu vi cán vợt":       "chu_vi_can",
    "chu vi cán":           "chu_vi_can",
    # swing weight
    "swing weight":         "swing_weight",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ─────────────────────────── HÀM TIỆN ÍCH ───────────────────────────

def safe_get(url: str, retries: int = 3):
    """GET có retry, trả None nếu thất bại."""
    for attempt in range(retries):
        try:
            res = SESSION.get(url, timeout=30)
            if res.status_code == 200:
                return res
            if res.status_code == 404:
                return None
            print(f"  [HTTP {res.status_code}] {url}")
        except Exception as e:
            print(f"  [Lỗi lần {attempt+1}] {url}: {e}")
        time.sleep(DELAY * (attempt + 1))
    return None


def is_product_url(href: str) -> bool:
   
    if not href.endswith(".html"):
        return False
    path = re.sub(r"https?://[^/]+", "", href).strip("/")
    if not path.startswith("vot-cau-long-"):
        return False
    # Bỏ prefix rồi kiểm tra phần còn lại có >= 2 từ không
    remainder = path[len("vot-cau-long-"):].replace(".html", "")
    return len(remainder.split("-")) >= 2


# ─────────────────────────── LẤY LINK SẢN PHẨM ───────────────────────────

def get_product_links_by_brand_id(brand_id: str) -> list:
    """
    Lấy tất cả link sản phẩm của một brand theo ID filter.
    URL pattern: /vot-cau-long?thuong_hieu=<id>&page=N&is_ajax=1
    """
    base = f"{BASE_URL}/vot-cau-long"
    links = []
    page = 1

    while True:
        url = f"{base}?thuong_hieu={brand_id}&page={page}&is_ajax=1"
        res = safe_get(url)
        if res is None:
            break

        soup = BeautifulSoup(res.content, "lxml")
        page_links = []

        seen = set()
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if not href.startswith("http"):
                href = BASE_URL + "/" + href.lstrip("/")
            if not is_product_url(href):
                continue
            if href not in seen:
                seen.add(href)
                page_links.append(href)

        if not page_links:
            break

        links.extend(page_links)
        page += 1
        if page > 50:
            break
        time.sleep(DELAY)

    return list(dict.fromkeys(links))


# ─────────────────────────── LẤY THÔNG SỐ SẢN PHẨM ───────────────────────────

def parse_specs_from_text(soup: BeautifulSoup) -> dict:
 
    specs = {}

    # Tìm vùng chứa mô tả sản phẩm
    desc_block = (
        soup.find("div", class_=re.compile(r"product-description|description|content", re.I))
        or soup.find("div", id=re.compile(r"description|content|tab", re.I))
        or soup.find("div", class_="tab-content")
        or soup.find("article")
    )

    if not desc_block:
        # Fallback: lấy toàn bộ nội dung trang
        desc_block = soup.find("body")

    if not desc_block:
        return specs

    # Lấy toàn bộ text, tách thành từng dòng
    full_text = desc_block.get_text(separator="\n", strip=True)
    lines = full_text.splitlines()

    for line in lines:
        line = line.strip()
        if ":" not in line:
            continue
        # Tách tại dấu ":" đầu tiên
        parts = line.split(":", 1)
        if len(parts) < 2:
            continue
        raw_key = parts[0].strip().lower().rstrip(":")
        raw_val = parts[1].strip()

        if not raw_val:
            continue

        # Tra cứu trong SPEC_MAP (partial match)
        for map_key, col_name in SPEC_MAP.items():
            if map_key in raw_key or raw_key in map_key:
                if col_name not in specs:   # chỉ lấy lần xuất hiện đầu tiên
                    specs[col_name] = raw_val
                break

    return specs


def get_product_specs(url: str) -> dict:
   
    res = safe_get(url)
    if res is None:
        return {}

    try:
        soup = BeautifulSoup(res.content, "lxml")
        data = {"url": url}

        # ── Tên sản phẩm ──
        h1 = soup.select_one("h1.product-title, h1.product_title, h1.entry-title, h1")
        raw_name = h1.get_text(strip=True) if h1 else None
        if raw_name:
            # Xóa prefix "Vợt Cầu Lông <Hãng>" khỏi tên
            cleaned = re.sub(
                r"(?i)^(set\s+)?v[oợ]t\s+c[aầ]u\s+l[oô]ng\s+"
                r"(yonex|victor|li[\s-]?ning|lining|mizuno|apacs|kumpoo|"
                r"felet|flypower|fleet|kawasaki|proace|forza|vs|vnb|"
                r"bubadu|kamito|kuno|adidas|vicleo|the\s+3rd\s+game)?\s*",
                "", raw_name
            ).strip()
            # Xóa tiếp tên dòng vợt ở đầu nếu còn
            for series in [
                "astrox", "nanoflare", "arcsaber", "voltric", "nanoray", "duora",
                "thruster", "jetspeed", "drivex", "auraspeed", "brave sword",
                "axforce", "turbo charging", "3d calibar", "tectonic",
                "halbertec", "aeronaut", "windstorm", "bladex", "lightning",
            ]:
                cleaned = re.sub(r"(?i)^" + re.escape(series) + r"\s*", "", cleaned).strip()
            data["ten_vot"] = cleaned if cleaned else raw_name
        else:
            data["ten_vot"] = None

        # ── Giá (VND) ──
        # Ưu tiên giá sale (trong <ins>), fallback giá thường
        # Dùng select() + [0] để chỉ lấy phần tử ĐẦU TIÊN,
        # tránh get_text() nối cả giá gốc lẫn giá sale thành chuỗi dài
        def first_price(selector):
            els = soup.select(selector)
            return els[0] if els else None

        price_el = (
            first_price("ins .woocommerce-Price-amount bdi")
            or first_price("ins .woocommerce-Price-amount")
            or first_price(".woocommerce-Price-amount bdi")
            or first_price(".woocommerce-Price-amount")
            or first_price("span.price")
            or first_price(".product-price")
        )
        if price_el:
            # Lấy text của chính phần tử (không đệ quy quá sâu để tránh nhặt nhiều giá)
            raw_price = price_el.get_text(separator="", strip=True)
            digits = re.sub(r"[^\d]", "", raw_price)
            # Nếu chuỗi số quá dài → có thể bị lặp, chỉ lấy nửa đầu
            if digits and len(digits) > 10 and len(digits) % 2 == 0:
                half = digits[:len(digits)//2]
                if half == digits[len(digits)//2:]:
                    digits = half
            data["gia_vnd"] = int(digits) if digits else None
        else:
            data["gia_vnd"] = None

        # ── Khởi tạo các cột thông số ──
        spec_cols = [
            "noi_san_xuat", "diem_can_bang", "do_cung", "suc_cang",
            "vat_lieu_khung", "mau_sac", "trong_luong",
            "chieu_dai", "swing_weight", "chu_vi_can",
            "thuong_hieu_trang",
        ]
        for c in spec_cols:
            data[c] = None

        # ── Parse thông số từ text mô tả ──
        specs = parse_specs_from_text(soup)
        data.update(specs)

        # ── Fallback: thử đọc bảng <table> nếu có ──
        if not any(data.get(c) for c in ["diem_can_bang", "do_cung", "suc_cang"]):
            tbl = soup.find("table")
            if tbl:
                for row in tbl.find_all("tr"):
                    cells = row.find_all(["th", "td"])
                    if len(cells) < 2:
                        continue
                    raw_key = cells[0].get_text(strip=True).lower().rstrip(":")
                    raw_val = cells[1].get_text(separator=" ", strip=True)
                    for map_key, col_name in SPEC_MAP.items():
                        if map_key in raw_key:
                            if not data.get(col_name):
                                data[col_name] = raw_val
                            break

        return data

    except Exception as e:
        print(f"  [Lỗi parse] {url}: {e}")
        return {}


# ─────────────────────────── MAIN ───────────────────────────

def main():
    print("=" * 60)
    print("  CRAWL VỢT CẦU LÔNG SHOPVNB.COM")
    print(f"  Tổng số hãng: {len(BRAND_IDS)}")
    print("=" * 60)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    all_records = []

    brands_to_crawl = BRAND_IDS
    if TEST_MODE:
        first_brand = list(BRAND_IDS.keys())[0]
        brands_to_crawl = {first_brand: BRAND_IDS[first_brand]}
        print(f"\n[TEST MODE] Chỉ crawl: {first_brand} (tối đa {TEST_LIMIT} sp)")

    for brand, brand_id in brands_to_crawl.items():
        print(f"\n{'='*50}")
        print(f"[HÃNG] {brand}  (id={brand_id})")

        # 1. Lấy danh sách link sản phẩm theo brand_id
        links = get_product_links_by_brand_id(brand_id)
        print(f"  → Tìm được {len(links)} sản phẩm")

        if not links:
            print("  → Bỏ qua (không tìm được sản phẩm)")
            continue

        if TEST_MODE:
            links = links[:TEST_LIMIT]
            print(f"  → Test với {len(links)} sản phẩm đầu")

        # 2. Crawl từng sản phẩm
        for link in tqdm(links, desc=f"  {brand}", unit="sp", ncols=80):
            specs = get_product_specs(link)
            if specs:
                specs["thuong_hieu"] = brand
                all_records.append(specs)
            time.sleep(DELAY)

    # 3. Lưu CSV
    if all_records:
        df = pd.DataFrame(all_records)

        col_order = [
            "thuong_hieu", "ten_vot",
            "noi_san_xuat", "diem_can_bang", "do_cung",
            "suc_cang", "vat_lieu_khung", "mau_sac",
            "trong_luong", "chieu_dai", "swing_weight", "chu_vi_can",
            "gia_vnd",
        ]
        col_order = [c for c in col_order if c in df.columns]
        df = df[col_order]
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

        print(f"\n{'='*60}")
        print(f"✅ Đã lưu {len(df)} sản phẩm → '{OUTPUT_FILE}'")
        print(f"\nThống kê theo hãng:")
        if "thuong_hieu" in df.columns:
            print(df["thuong_hieu"].value_counts().to_string())
        print(f"\nMẫu dữ liệu:")
        print(df[["thuong_hieu", "ten_vot",
                   "noi_san_xuat", "diem_can_bang", "do_cung",
                   "mau_sac", "gia_vnd"]].head(5).to_string())
    else:
        print("\n❌ Không có dữ liệu nào.")


if __name__ == "__main__":
    main()
