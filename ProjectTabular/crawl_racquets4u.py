import os
import subprocess
import time
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

# ĐẶT TEST_MODE = True ĐỂ CHẠY THỬ VÀI SẢN PHẨM RỒI DỪNG LẠI. Đặt False để cào toàn bộ mạng.
TEST_MODE = True

def fetch_html(url):
    """Sử dụng cURL qua subprocess để lấy HTML, vượt qua lớp block của web"""
    # Tuỳ chọn -s (silent) để tắt thanh tiến trình mặc định của cURL
    command = ['curl.exe', '-s', url, '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36']
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        return result.stdout
    except Exception as e:
        print(f"Lỗi cURL khi truy cập {url}: {e}")
        return ""

def get_all_brands(base_url):
    """Lấy danh sách tất cả các hãng vợt cầu lông từ bộ lọc của Racquets4U"""
    html = fetch_html(base_url)
    brands = []
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        # Tìm tất cả thẻ li chứa thương hiệu trong filter
        for a in soup.select('.item a'):
            href = a.get('href', '')
            if 'brands=' in href and 'badminton-racquets' in href:
                label_span = a.select_one('span.label')
                if label_span:
                    name = label_span.text.strip()
                    brands.append({"name": name, "url": href})
                    
    # Loại bỏ trùng lặp nếu có (do filter có thể lặp lại trên mobile/desktop view)
    unique_brands = []
    seen = set()
    for b in brands:
        if b['name'] not in seen:
            unique_brands.append(b)
            seen.add(b['name'])
            
    return unique_brands

def get_product_links(base_url, target_brand_name, max_pages=50):
    """Lấy toàn bộ link sản phẩm Yonex qua các trang (Pagination)"""
    product_links = []
    print(f"BƯỚC 1: Đang quét danh mục hãng {target_brand_name} để lấy link sản phẩm...")
    
    for page in range(1, max_pages + 1):
        # Sửa chỗ nối chuỗi url cho chắc chắn. Nếu href gốc là '?brands=xxx' thì nối thêm '&p=page'
        if '?' in base_url:
            url = f"{base_url}&p={page}"
        else:
            url = f"{base_url}?p={page}"
            
        html = fetch_html(url)
        
        if not html:
            break
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # Sửa lỗi lấy nhầm cả "Related Products" ở sidebar của hãng khác
        # Chỉ lấy sản phẩm nằm trong khối .products.wrapper (danh sách chính)
        main_list = soup.select_one('.products.wrapper')
        if not main_list:
            print(f" -> Đã đến trang cuối cùng. Dừng quét tại trang {page-1}.")
            break
            
        items = main_list.select('.product-item-info')
        
        if not items:
            print(f" -> Đã đến trang cuối cùng. Dừng quét tại trang {page-1}.")
            break # Hết dữ liệu (hết trang)
            
        page_links = []
        for item in items:
            link_tag = item.select_one('a.product-item-link')
            if link_tag and link_tag.get('href'):
                page_links.append(link_tag['href'])
                
        if not page_links:
            break
            
        product_links.extend(page_links)
        print(f" -> Trang {page}: Tìm thấy {len(page_links)} sản phẩm. (Tổng cộng đang có: {len(product_links)})")
        
        # Test mode: Dừng ngay khi lấy được link ở trang đầu tiên
        if TEST_MODE and len(product_links) > 0:
            print(" -> [TEST MODE] Đã thu thập đủ link mẫu, dừng Pagination.")
            # Cắt bớt chỉ lấy 3 link để test tốc độ cào
            product_links = product_links[:3]
            break
            
        # Nghỉ chút tránh server nghi ngờ và chặn rate limit
        time.sleep(1)
        
    return list(dict.fromkeys(product_links)) # Loại bỏ link trùng lặp

def get_product_details(url):
    """Bóc tách chi tiết kỹ thuật từ trang sản phẩm"""
    html = fetch_html(url)
    if not html:
        return None
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # 1. Tên sản phẩm
    title_tag = soup.select_one('h1.page-title span')
    title = title_tag.text.strip() if title_tag else "N/A"
    
    # 2. Giá trị (Lấy phần số từ div.price thứ 2 hoặc thẻ có class chứa giá)
    price_tags = soup.select('.price')
    # Thẻ price đầu tiên thường là ký hiệu đồng tiền Rupee '₹', thẻ thứ 2 mới là số tiền
    price = price_tags[1].text.strip() if len(price_tags) > 1 else "N/A"
    
    # 3. Thông số kỹ thuật chung
    specs = {
        "Brand": "N/A",  # Tạm để N/A, sẽ được ghi đè bằng tên hãng thực tế. Lát loop sẽ insert.
        "Name": title,
        "Price (Rs)": price,
        "Item Code": "N/A",
        "Weight": "N/A",
        "Grip Size": "N/A",
        "Flexibility": "N/A",
        "Balance Point": "N/A",
        "Head Shape": "N/A",
        "Stringing Advice": "N/A",
        "Playing Level": "N/A",
        "Frame": "N/A",
        "Shaft": "N/A",
        "Joint": "N/A",
        "Length": "N/A",
    }
    
    # Bảng `additional-attributes` chứa toàn bộ thông số cực sạch của web này
    table = soup.select_one('table.additional-attributes')
    if table:
        for row in table.select('tr'):
            th = row.select_one('th')
            td = row.select_one('td')
            if th and td:
                key = th.text.strip()
                val = td.text.strip()
                
                # Gán giá trị vào từ điển specs
                if key in specs:
                    specs[key] = val
                else:
                    specs[key] = val # Cột ngoài dự kiến
                    
    return specs

if __name__ == "__main__":
    main_category_url = "https://www.racquets4u.com/badminton/badminton-racquets.html"
    
    print("BƯỚC 0: Đang tự động quét danh sách toàn bộ hãng vợt từ website...")
    brands = get_all_brands(main_category_url)
    
    if not brands:
        print("Không tìm thấy hãng nào. Vui lòng kiểm tra lại mạng hoặc anti-bot.")
        exit()
        
    print(f"Tìm thấy {len(brands)} hãng: {', '.join([b['name'] for b in brands])}\n")
    
    all_data = []
    
    # Lặp qua toàn bộ hãng tìm được
    for i, brand_info in enumerate(brands, 1):
        target_brand = brand_info['name']
        category_url = brand_info['url']
        
        print(f"\n--- Đang xử lý hãng {i}/{len(brands)}: {target_brand} ---")
        
        # Lấy link từng sản phẩm của hãng này
        target_links = get_product_links(category_url, target_brand, max_pages=20)
        
        print(f"Tìm thấy {len(target_links)} sản phẩm của {target_brand}.")
        if not target_links:
            continue
            
        # Bước 2: Quét chi tiết từng link (Bọc tqdm để xem thanh tiến trình)
        for link in tqdm(target_links, desc=f"BƯỚC 2: Cào chi tiết ({target_brand})", unit="vợt"):
            product_data = get_product_details(link)
            if product_data:
                product_data['Product_URL'] = link
                
                # Nếu trang web không có thông tin brand, lúc đó mới dùng tên danh mục (target_brand)
                if product_data.get('Brand', 'N/A') == 'N/A':
                    product_data['Brand'] = target_brand 
                    
                all_data.append(product_data)
            
            # Nhất định phải sleep 1 giây khi dùng curl với web nước ngoài để không bị cấm
            time.sleep(1) 
            
        if TEST_MODE:
            print(f"\n[TEST MODE] Đã chạy xong mẫu cho 1 hãng ({target_brand}). Đang thoát vòng lặp hãng...")
            break
            
    print("\nBƯỚC 3: Dọn dẹp và lưu file CSV...")
    df = pd.DataFrame(all_data)
    
    if not df.empty:
        # Ưu tiên các cột quan trọng nhất lên đầu DataFrame
        important_cols = [
            "Brand", "Name", "Price (Rs)", "SKU", "Unstrung Weight", "Grip Size",
            "Balance", "String Tension", "Playing Level", "Material", "Country of Origin"
        ]
        
        # Các cột còn lại được xếp cuối
        existing_cols = [c for c in important_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in important_cols]
        final_cols = existing_cols + other_cols
        
        df = df[final_cols]
        
        # Đảm bảo thư mục lưu trữ tồn tại
        os.makedirs("ProjectTabular/data", exist_ok=True)
        filename = "ProjectTabular/data/racquets4u_all_brands.csv"
        
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"🎉 HOÀN THÀNH TỐT ĐẸP! Đã lưu {len(df)} cây vợt toàn bộ website vào file {filename}")
    else:
        print("Không có mặt hàng nào được cào.")
