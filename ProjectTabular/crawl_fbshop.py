import requests
from bs4 import BeautifulSoup
import pandas as pd  
import time
import re
from tqdm import tqdm  # <-- Import thêm tqdm ở đây

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_all_brands():
    """Hàm lấy tất cả các link danh mục hãng từ trang chủ vợt cầu lông."""
    url = "https://fbshop.vn/vot-cau-long/"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            links = soup.find_all('a', href=True)
            # Lọc các link có dạng https://fbshop.vn/vot-cau-long-[ten-hang]/
            brand_urls = sorted(list(set(
                a['href'] for a in links 
                if a['href'].count('/') == 4 and a['href'].startswith('https://fbshop.vn/vot-cau-long-')
            )))
            
            brands_info = []
            for b_url in brand_urls:
                parts = b_url.strip('/').split('-')
                if "long" in parts:
                    idx = parts.index("long")
                    brand_name = " ".join(parts[idx+1:]).title()
                else:
                    brand_name = parts[-1].title()
                brands_info.append({"name": brand_name, "url": b_url})
            return brands_info
    except Exception as e:
        print(f"Lỗi khi lấy danh sách hãng: {e}")
    return []

def get_product_links(category_url, max_pages=999):
    """Hàm lấy tất cả link sản phẩm từ các trang danh mục"""
    product_links = []
    
    # --- Bước 1.1: Tìm tổng số trang thực tế của danh mục ---
    try:
        response = requests.get(category_url, headers=HEADERS)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            # Tìm thẻ chứa số trang cuối cùng (vd: <a class="page-numbers">21</a>)
            pagination = soup.select("ul.page-numbers li a.page-numbers")
            if pagination:
                # Lấy số lớn nhất từ các nút chuyển trang (thường là nút áp chót trước nút "Next")
                last_page_text = pagination[-2].text.strip() if '→' in pagination[-1].text or 'Next' in pagination[-1].text else pagination[-1].text.strip()
                if last_page_text.isdigit():
                    actual_max = int(last_page_text)
                    max_pages = min(max_pages, actual_max) # Lấy số nhỏ hơn giữa user nhập và thực tế
    except Exception as e:
        print(f"Không thể lấy tổng số trang, sẽ quét đến tối đa {max_pages} trang. Lỗi: {e}")
    
    # --- Bước 1.2: Bắt đầu lấy link từng trang ---
    for page in tqdm(range(1, max_pages + 1), desc="BƯỚC 1: Đang quét trang danh mục", unit="trang"):
        url = f"{category_url}page/{page}/" if page > 1 else category_url
        
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            if response.status_code != 200:
                break
        except Exception as e:
            tqdm.write(f"Lỗi mạng khi tải trang {page}: {e}")
            break
            
        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.select("a.prd-card-link") 
        
        if not links:
            break
            
        for link in links:
            href = link.get('href')
            if href and href not in product_links:
                product_links.append(href)
                
        time.sleep(1)
        
    return product_links

def parse_title(title, brand):
    """Hàm xử lý tách Tên sản phẩm thành Series và Version"""
    title_clean = title.split('|')[0]
    title_clean = title_clean.lower().replace("vợt cầu lông", "").replace(brand.lower(), "").strip()
    
    version_keywords = ['pro', 'game', 'tour', 'play', 'feel', 'clear', 'ability', 'lite', 'zz', 'z']
    
    series = title_clean.title()
    version = ""
    
    parts = title_clean.split()
    for i, part in enumerate(parts):
        if any(kw in part for kw in version_keywords):
            series = " ".join(parts[:i]).title()
            version = " ".join(parts[i:]).title()
            break
            
    return series, version

def get_product_details(product_url, brand):
    """Hàm vào từng trang sản phẩm để lấy chi tiết kỹ thuật"""
    try:
        response = requests.get(product_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # 1. Tên sản phẩm
        title_element = soup.select_one("h1.t-title")
        full_title = title_element.text.strip().replace('"', '') if title_element else "N/A"
        series, version = parse_title(full_title, brand)
        
        # 2. Giá sản phẩm
        price_element = soup.select_one("span.price-current")
        if not price_element:
            price_element = soup.select_one("span.price-regular")
            
        if price_element is not None:
            price_text = price_element.text.replace(".", "").replace("đ", "").strip()
            # Lấy giá đầu tiên nếu là chuỗi khoảng giá (VD: 3619000 - 4050000)
            price = price_text.split("-")[0].strip() if "-" in price_text else price_text
        else:
            price = "N/A"
        
        # 3. Xuất xứ
        origin = "N/A"
        meta_labels = soup.select("span.meta-label")
        for label in meta_labels:
            if "Xuất xứ" in label.text:
                origin_val = label.find_next_sibling("span", class_="meta-value")
                if origin_val:
                    origin = origin_val.text.strip()
                break

        # 4. Thông số
        # 4. Thông số
        max_tension = "N/A"
        technologies = "N/A"
        
        # --- LẤY MỨC CĂNG TỪ TAB THÔNG SỐ KỸ THUẬT ---
        info_tags = soup.find_all(['span', 'p', 'li'])
        for tag in info_tags:
            text = tag.get_text(strip=True)
            if ("Mức căng" in text or "Sức căng" in text) and max_tension == "N/A":
                if tag.name == 'p' and tag.find_next_sibling("ul"):
                    ul_tag = tag.find_next_sibling("ul")
                    if ul_tag:
                        max_tension = " | ".join([li.get_text(strip=True) for li in ul_tag.find_all("li")])
                elif ":" in text and len(text) < 100:
                    max_tension = text.split(":", 1)[-1].strip()

        # Rút gọn mức căng chỉ lấy dạng "XX - YY lbs" hoặc "XX lbs"
        if max_tension != "N/A":
            # Biểu thức chính quy tìm các số đi kèm chữ lbs (có thể có khoảng trắng, gạch ngang)
            # VD matches: "20 - 28 lbs", "28lbs", "20-30 LBS"
            match = re.search(r'\d+\s*[-~]*\s*\d*\s*(?:lbs|LBS|Lbs)', max_tension)
            if match:
                # Xóa các khoảng trắng thừa bên trong chuỗi tìm được để chuẩn format
                max_tension = re.sub(r'\s+', ' ', match.group(0))
            else:
                max_tension = "N/A" # Nếu tìm được text nhưng không có format lbs thì coi như N/A

        # --- LẤY CÔNG NGHỆ TỪ BÀI VIẾT (Như ảnh bạn gửi) ---
        technologies_list = []
        
        # Tìm tiêu đề bài viết có chữ "Công nghệ" (thường là thẻ h2, h3, h4)
        tech_headings = soup.find_all(['h2', 'h3', 'h4'], string=re.compile(r"Công nghệ", re.IGNORECASE))
        
        if tech_headings:
            # Lấy khu vực dưới tiêu đề đầu tiên tìm thấy
            heading = tech_headings[0]
            
            # Duyệt qua các phần tử nằm ngay dưới tiêu đề đó
            for sibling in heading.find_next_siblings():
                # Nếu đụng phải một tiêu đề mục khác (VD: "4. Đối tượng phù hợp") thì dừng lại
                if sibling.name in ['h2', 'h3', 'h4']:
                    break
                
                text = sibling.get_text(strip=True)
                # Tìm các dòng có dấu ":" (Ví dụ: "– 2G NAMD FLEX FORCE: Là công nghệ...")
                if ":" in text:
                    # Cắt lấy phần trước dấu hai chấm
                    tech_name = text.split(":")[0]
                    # Xóa dấu gạch ngang (–, -) ở đầu dòng nếu có
                    tech_name = re.sub(r"^[-–\s]+", "", tech_name).strip()
                    
                    # Lọc sương sương: Tên công nghệ thường không quá dài
                    if len(tech_name) > 2 and len(tech_name) < 40:
                        technologies_list.append(tech_name)

        # Gộp danh sách công nghệ lại bằng dấu phẩy
        if technologies_list:
            technologies = ", ".join(technologies_list)
        else:
            # Backup: Nếu bài viết không ghi rõ công nghệ, quay về lấy vật liệu Khung vợt
            tech_tag = soup.find(string=re.compile(r"Khung vợt", re.IGNORECASE))
            if tech_tag and tech_tag.parent and ":" in tech_tag.parent.text:
                technologies = tech_tag.parent.text.split(":", 1)[-1].strip()

        return {
            "Brand": brand,
            "Series": series,
            "Version": version,
            "Max Tension": max_tension,
            "Technologies": technologies,
            "Origin": origin,
            "Price": price
        }
    except Exception as e:
        tqdm.write(f"Lỗi khi crawl link {product_url}: {e}")
        return None

if __name__ == "__main__":
    print("BƯỚC 0: Đang lấy danh sách các hãng vợt cầu lông...")
    brands = get_all_brands()
    
    if not brands:
        print("Không tìm thấy hãng nào. Dừng chương trình.")
        exit()
        
    print(f"Tìm thấy {len(brands)} hãng: {', '.join([b['name'] for b in brands])}\n")
    
    all_data = []
    
    # Lặp qua từng hãng
    for i, brand_info in enumerate(brands, 1):
        target_brand = brand_info['name']
        category_url = brand_info['url']
        
        print(f"--- Đang xử lý hãng {i}/{len(brands)}: {target_brand} ---")
        
        # Bước 1: Lấy link (tqdm đã nằm trong hàm)
        links = get_product_links(category_url, max_pages=999) 
        
        print(f"Tìm thấy {len(links)} sản phẩm của {target_brand}.")
        
        # Bước 2: Bọc tqdm vào vòng lặp crawl chi tiết sản phẩm
        for link in tqdm(links, desc=f"BƯỚC 2: Đang tải chi tiết ({target_brand})", unit="vợt"):
            data = get_product_details(link, target_brand)
            if data:
                all_data.append(data)
            time.sleep(1) # Nghỉ 1s chống spam
            
        print("\n")
        
    print("BƯỚC 3: Lưu dữ liệu ra file CSV...")
    df = pd.DataFrame(all_data)
    
    columns_order = ["Brand", "Series", "Version", "Max Tension", "Technologies", "Origin", "Price"]
    if not df.empty:
        df = df[columns_order]
        filename = f"ProjectTabular/data/data_fbshop.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"HOÀN THÀNH! Đã lưu {len(df)} sản phẩm vào: {filename}")
    else:
        print("Không có dữ liệu nào được crawl.")