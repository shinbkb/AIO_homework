"""
Tìm toàn bộ brand vợt cầu lông trên shopvnb.com
Chiến lược:
  1. Gọi URL bộ lọc (AJAX) để lấy danh sách thương hiệu từ sidebar filter
  2. Tìm tất cả link /vot-cau-long-<brand> từ trang chủ danh mục
"""

import requests
from bs4 import BeautifulSoup
import re

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi,en;q=0.9",
    "Referer": "https://shopvnb.com/",
}

BASE = "https://shopvnb.com"

session = requests.Session()
session.headers.update(HEADERS)

found_brands = {}  # slug -> url

# ── Phương án 1: parse từ trang danh mục (có bộ lọc Thương Hiệu) ──
print("=== Phương án 1: Sidebar filter trang danh mục ===")
for url in [
    f"{BASE}/vot-cau-long?&is_ajax=1",
    f"{BASE}/vot-cau-long",
]:
    try:
        res = session.get(url, timeout=30)
        soup = BeautifulSoup(res.content, "lxml")

        # Tìm tất cả link /vot-cau-long-xxx trong sidebar filter
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                href = BASE + "/" + href.lstrip("/")
            # Pattern: /vot-cau-long-<brand> (không có thêm /-<series>)
            m = re.match(
                r"https://shopvnb\.com/(vot-cau-long-[a-z0-9\-]+?)(?:/|\?|$)",
                href
            )
            if m:
                slug = m.group(1)
                # Loại bỏ các slug là trang series (có nhiều từ ghép dài)
                # Brand URL thường: vot-cau-long-yonex  (không có series phụ)
                brand_part = slug.replace("vot-cau-long-", "")
                # Giữ lại slug có ít dấu gạch ngang (brand đơn)
                found_brands[slug] = f"{BASE}/{slug}"

        print(f"  [{res.status_code}] {url} → tìm được {len(found_brands)} slug")
        break
    except Exception as e:
        print(f"  Lỗi: {e}")

# ── Phương án 2: sitemap ──
print("\n=== Phương án 2: Sitemap ===")
for sitemap_url in [
    f"{BASE}/sitemap.xml",
    f"{BASE}/sitemap_index.xml",
    f"{BASE}/sitemap-categories.xml",
]:
    try:
        res = session.get(sitemap_url, timeout=20)
        if res.status_code == 200:
            # Tìm tất cả <loc> chứa vot-cau-long-
            locs = re.findall(r"<loc>(https://shopvnb\.com/vot-cau-long-[^<]+)</loc>", res.text)
            for loc in locs:
                m = re.match(
                    r"https://shopvnb\.com/(vot-cau-long-[a-z0-9\-]+?)(?:/|\?|\.html|$)",
                    loc
                )
                if m:
                    slug = m.group(1)
                    found_brands[slug] = f"{BASE}/{slug}"
            print(f"  [{res.status_code}] {sitemap_url} → +{len(locs)} locs")
        else:
            print(f"  [{res.status_code}] {sitemap_url}")
    except Exception as e:
        print(f"  Lỗi {sitemap_url}: {e}")

# ── Phương án 3: trang filter AJAX trả JSON/HTML riêng ──
print("\n=== Phương án 3: AJAX filter endpoint ===")
ajax_urls = [
    f"{BASE}/vot-cau-long?filter_cat=1&is_ajax=1",
    f"{BASE}/wp-admin/admin-ajax.php",
]
for url in ajax_urls:
    try:
        res = session.get(url, timeout=20)
        if res.status_code == 200 and len(res.content) > 200:
            soup = BeautifulSoup(res.content, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "vot-cau-long-" in href:
                    m = re.search(r"/(vot-cau-long-[a-z0-9\-]+?)(?:/|\?|$)", href)
                    if m:
                        slug = m.group(1)
                        found_brands[slug] = f"{BASE}/{slug}"
            print(f"  [{res.status_code}] {url}")
    except Exception as e:
        print(f"  Lỗi {url}: {e}")

# ── In kết quả ──
print("\n" + "=" * 60)
print(f"TỔNG CỘNG TÌM ĐƯỢC: {len(found_brands)} slug")
print("=" * 60)

# Lọc bỏ các slug quá dài (thường là trang series, không phải brand)
brand_slugs = {
    slug: url for slug, url in found_brands.items()
    if len(slug.replace("vot-cau-long-", "").split("-")) <= 3
}

print(f"\n--- Slug khả năng là BRAND (≤3 từ) ---  [{len(brand_slugs)} mục]")
for slug, url in sorted(brand_slugs.items()):
    brand_name = slug.replace("vot-cau-long-", "").replace("-", " ").title()
    print(f"  {brand_name:25s}  {url}")

other_slugs = {k: v for k, v in found_brands.items() if k not in brand_slugs}
if other_slugs:
    print(f"\n--- Slug dài hơn (có thể là series) ---  [{len(other_slugs)} mục]")
    for slug, url in sorted(other_slugs.items()):
        print(f"  {url}")
