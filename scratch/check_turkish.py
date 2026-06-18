import os
import re
import tokenize

turkish_words = {
    've', 'bir', 'ile', 'icin', 'olan', 'olarak', 'en', 'bu', 'da', 'de', 'ise',
    'deger', 'sinif', 'sayisi', 'koordinat', 'hastane', 'uzaklik', 'mesafe',
    'nufus', 'siginak', 'bina', 'yikim', 'durum', 'alan', 'cozum', 'sec', 'kat',
    'yil', 'sure', 'oran', 'gosterge', 'agirlik', 'kompozit', 'esitlik',
    'oncelik', 'sentez', 'isi', 'konfor', 'yesil', 'acik', 'gecirimsiz',
    'sicaklik', 'maruziyet', 'risk', 'skor', 'uretir', 'okul', 'yol',
    'en_kisa', 'yakinlik', 'arasilik', 'ozdeger', 'kritiklik', 'cekim',
    'kumulatif', 'firsat', 'erisebilirlik', 'otokorelasyon', 'esik',
    'komsu', 'ag', 'komsuluk', 'eklenti', 'veri', 'analiz', 'motoru',
    'ekosistem', 'cekirdek', 'mekansal', 'istatistik', 'direnclilik',
    'bagimsiz', 'calistirma', 'hizli', 'test', 'merkezi', 'yonetim', 'tanimi'
}

# Add words with Turkish characters too (although we already searched for Turkish characters, just in case)
turkish_char_words = {
    'için', 'değer', 'sınıf', 'sayısı', 'sığınak', 'yıkım', 'yıl', 'süre', 'ağırlık',
    'öncelik', 'ısı', 'yeşil', 'açık', 'sıcaklık', 'üretir', 'hücre', 'yakınlık',
    'arasılık', 'özdeğer', 'çekim', 'kümülatif', 'fırsat', 'erişilebilirlik',
    'eşik', 'komşu', 'ağ', 'komşuluk', 'çekirdek', 'mekansal', 'bağımsız',
    'çalıştırma', 'hızlı', 'yönetim', 'tanımı'
}

all_turkish_words = turkish_words.union(turkish_char_words)

def clean_text(text):
    # remove punctuation and keep words
    return re.findall(r'\b\w+\b', text.lower())

def check_file(filepath):
    findings = []
    with open(filepath, 'rb') as f:
        try:
            tokens = list(tokenize.tokenize(f.readline))
        except tokenize.TokenError:
            return []
        
        for tok in tokens:
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                content = tok.string
                words = clean_text(content)
                for w in words:
                    if w in all_turkish_words:
                        findings.append((tok.start[0], tok.type, w, content.strip()))
    return findings

def main():
    root_dirs = ['src', 'tests']
    total_findings = 0
    for root_dir in root_dirs:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py'):
                    path = os.path.join(root, file)
                    findings = check_file(path)
                    if findings:
                        print(f"\nFile: {path}")
                        for line_num, tok_type, word, snippet in findings[:10]:
                            type_str = "COMMENT" if tok_type == tokenize.COMMENT else "STRING"
                            print(f"  Line {line_num} ({type_str}) matched word '{word}': {snippet[:80]}")
                        if len(findings) > 10:
                            print(f"  ... and {len(findings) - 10} more findings")
                        total_findings += len(findings)
    print(f"\nTotal matches found: {total_findings}")

if __name__ == '__main__':
    main()
