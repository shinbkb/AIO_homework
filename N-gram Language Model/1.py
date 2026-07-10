import re
import math
import random
import os
import sys
from collections import defaultdict, Counter

# Fix lỗi hiển thị tiếng Việt và emoji trên Windows
sys.stdout.reconfigure(encoding='utf-8')
from tqdm import tqdm

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)   # bỏ dấu câu
    text = re.sub(r'\d+', '', text)        # bỏ số
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(filepath, train_ratio=0.8):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    cleaned = preprocess(raw_text)
    tokens = cleaned.split()
    
    split_idx = int(train_ratio * len(tokens))
    train_tokens = tokens[:split_idx]
    test_tokens  = tokens[split_idx:]
    
    print(f"✅ Load dữ liệu thành công | Tổng token: {len(tokens):,}")
    print(f"👉 Train: {len(train_tokens):,} words | Test: {len(test_tokens):,} words\n")
    return train_tokens, test_tokens

class NGramModel:
    def __init__(self, n=2):
        self.n = n
        self.ngram_counts   = defaultdict(Counter)
        self.context_totals = defaultdict(int)
        self.vocab = set()
        self.V = 0

    def train(self, tokens):
        self.vocab = set(tokens)
        self.V = len(self.vocab)
        total_ngrams = len(tokens) - self.n + 1

        print("=" * 60)
        print(f"🚀 TRAINING MÔ HÌNH N-GRAM (N = {self.n})")
        print("=" * 60)

        for i in tqdm(range(total_ngrams), desc=f"Counting {self.n}-grams", ncols=80, colour="green"):
            ngram   = tuple(tokens[i : i + self.n])
            context = ngram[:-1]
            word    = ngram[-1]
            self.ngram_counts[context][word] += 1
            self.context_totals[context]     += 1

        print(f"\n📊 Thống kê:")
        print(f"   - Kích thước Vocabulary: {self.V:,} từ")
        print(f"   - Tổng số context duyệt: {len(self.ngram_counts):,}")
        
        top_ctx = sorted(self.context_totals.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   - Top 3 context phổ biến:")
        for ctx, cnt in top_ctx:
            top_next = self.ngram_counts[ctx].most_common(2)
            next_str = ", ".join([f"'{w}'({c})" for w, c in top_next])
            print(f"       {ctx} → [{next_str}] (tổng: {cnt} lần)")
        print("-" * 60 + "\n")

    def prob(self, context, word):
        """Tính xác suất Laplace Smoothing"""
        context = tuple(context)
        count_w = self.ngram_counts[context].get(word, 0)
        count_c = self.context_totals.get(context, 0)
        return (count_w + 1) / (count_c + self.V)

    def predict_next(self, context, top_k=5):
        """Dự đoán từ tiếp theo"""
        context = tuple(context[-(self.n - 1):])
        candidates = self.ngram_counts.get(context, {})
        
        if not candidates:
            return []
            
        scored = {w: self.prob(context, w) for w in candidates}
        return sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_k]


def compute_perplexity(model, tokens):
    n       = model.n
    log_sum = 0.0
    count   = 0

    for i in range(n - 1, len(tokens)):
        context = tuple(tokens[i - (n - 1) : i])
        word    = tokens[i]
        p       = model.prob(context, word)
        log_sum += math.log(p)
        count   += 1

    return math.exp(-log_sum / count)

def generate_sentence(model, seed_words, max_words=30):
    words = list(seed_words)
    n = model.n

    for _ in range(max_words):
        context = tuple(words[-(n - 1):])
        candidates = model.predict_next(context, top_k=10)

        if not candidates:
            break

        next_words = [w for w, _ in candidates]
        probs      = [p for _, p in candidates]
        total      = sum(probs)
        norm_probs = [p / total for p in probs]

        next_word = random.choices(next_words, weights=norm_probs, k=1)[0]
        words.append(next_word)

    return ' '.join(words)


def main():
    # Lấy thư mục chứa file 1.py hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, "data", "truyen_kieu_data.txt")
    
    try:
        train_tokens, test_tokens = load_data(DATA_PATH)
    except Exception as e:
        print(f"Lỗi đọc file: {e}\nBạn hãy kiểm tra lại đường dẫn.")
        return

    # 2. Khởi tạo và Train mô hình Trigram (N=3)
    model_tri = NGramModel(n=3)
    model_tri.train(train_tokens)

    # 3. Đánh giá bằng Perplexity
    pp_tri = compute_perplexity(model_tri, test_tokens)
    print(f"🎯 Perplexity trên tập TEST (Trigram): {pp_tri:.2f}\n")

    # 4. Sinh văn bản thử nghiệm
    seed = ['trăm', 'năm']
    print("📝 Thử sinh văn bản với từ gợi ý:", seed)
    sentence = generate_sentence(model_tri, seed_words=seed, max_words=30)
    print(f"💬 Kết quả: \"{sentence}\"")

    # 5. So sánh nhanh Bigram và Trigram
    print("\n" + "="*60)
    print("🔍 SO SÁNH NHANH BIGRAM VÀ TRIGRAM")
    model_bi = NGramModel(n=2)
    model_bi.train(train_tokens)
    pp_bi = compute_perplexity(model_bi, test_tokens)
    
    print(f"   => Perplexity N=2 (Bigram) : {pp_bi:.2f}")
    print(f"   => Perplexity N=3 (Trigram): {pp_tri:.2f}")
    if pp_bi < pp_tri:
         print("   💡 Note: Kích thước dữ liệu đang khá nhỏ nên Bigram cho PP tốt hơn!")


if __name__ == "__main__":
    main()
