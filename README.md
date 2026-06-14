# PlanX SDK (Software Development Kit)

[![PyPI version](https://img.shields.io/pypi/v/planx-sdk.svg)](https://pypi.org/project/planx-sdk/)
[![Python version support](https://img.shields.io/pypi/pyversions/planx-sdk.svg)](https://pypi.org/project/planx-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Code Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PlanX QGIS eklenti ekosisteminin çekirdek mekansal analiz, istatistik, ağ analizleri ve kentsel dirençlilik hesaplama motorlarını barındıran, QGIS arayüzünden bağımsız, saf Python, NumPy ve SciPy tabanlı resmi kütüphanesidir.

Bu kütüphane sayesinde, QGIS eklentilerindeki karmaşık algoritmaları:
1. **Headless & Bağımsız Çalıştırma:** QGIS'i başlatmadan (örneğin Jupyter Notebooks, bağımsız komut dosyaları, sunucular veya web uygulamalarında) çalıştırabilir,
2. **Hızlı Test Edebilme:** `pytest` ile hızlı ve headless bir şekilde tüm analitik modülleri test edebilir,
3. **Merkezi Yönetim:** Tek bir merkezden güncelleyerek (PyPI üzerinden `pip install planx-sdk`) tüm eklentilerinizde ve araç setlerinizde tutarlı bir şekilde kullanabilirsiniz.

---

## 📂 Proje Yapısı (Directory Structure)

```text
planx_sdk/
  ├── .github/workflows/          # GitHub Actions (PyPI otomatik yayınlama entegrasyonu)
  ├── pyproject.toml              # Modern paket tanımı, bağımlılıklar ve metadata (PEP 517/621)
  ├── README.md                   # Bu dökümantasyon dosyası
  ├── LICENSE                     # Lisans dosyası
  ├── .gitignore                  # Önbellek ve build dosyalarını yoksayan Git kuralları
  ├── src/                        # Kaynak kodlar (Standard src-layout)
  │   └── planx/                  # Ana paket dizini
  │       ├── __init__.py         # Versiyon ve paket tanımı
  │       ├── spatial/            # Çekirdek mekansal ağ analizi ve space syntax motoru
  │       │   ├── __init__.py
  │       │   ├── paths.py        # En kısa yol algoritmaları (Dijkstra, SciPy entegrasyonu)
  │       │   ├── centrality.py   # Yakınlık, Arasılık (Betweenness), Özdeğer (Eigenvector) ve Kritiklik
  │       │   └── accessibility.py# Çekim (Hansen) ve Kümülatif Fırsatlar erişilebilirlik modelleri
  │       ├── geostats/           # Mekansal istatistik ve otokorelasyon motorları
  │       │   ├── __init__.py
  │       │   └── stats_engines.py# Getis-Ord Gi*, Local/Global Moran's I, OLS, GWR, SDE, k-means
  │       ├── suitability/        # Raster tabanlı MCDA (Multi-Criteria Decision Analysis) motoru
  │       │   ├── __init__.py
  │       │   ├── mcda.py         # Normalizasyon metotları (Sigmoid, Gaussian, Min-Max) ve WLC
  │       │   ├── facility.py     # Greedy MCLP (Maximal Covering Location Problem) tesis yerleşimi
  │       │   └── weights.py      # Karar matrisi ağırlıklandırma metotları (AHP, Entropy, CRITIC, PCA)
  │       └── resilience/         # Kentsel dirençlilik, afet ve risk simülasyon motorları
  │           ├── __init__.py
  │           ├── seismic.py      # Monte Carlo sismik yapısal hasar ve enkaz yayılım simülasyonu
  │           ├── flood.py        # DEM tabanlı plüvyal (yüzey suyu) taşkın duyarlılık analizi
  │           ├── social.py       # Sosyal kırılganlık endeksi (SVI) tarama ve analizi
  │           └── heat.py         # Kentsel ısı konforu riski ve yeşil alan açığı tarama modeli
  └── tests/                      # Birim testler (Unit Tests)
```

---

## 💡 Temel Özellikler ve Kullanım Örnekleri

### 1. Kentsel Erişilebilirlik Analizleri (`planx.spatial`)
Köken noktalarından (örn: konutlar) varış noktalarına (örn: hastaneler) olan mesafeleri kullanarak çekim veya kümülatif fırsat modelleriyle erişilebilirliği hesaplar.

```python
import numpy as np
from planx.spatial import gravity_accessibility

# Mesafe Matrisi (O x D): 2 köken noktasının 3 varış noktasına uzaklıkları (m)
dists = np.array([
    [150.0, 300.0, 900.0],
    [500.0, 100.0, 1200.0]
])
# Varış noktalarının kapasite/cazibe ağırlıkları (örn. hastane yatak sayısı)
weights = np.array([50.0, 100.0, 250.0])

# Üstel (exponential) azalım fonksiyonu ile çekim tabanlı erişilebilirlik (Hansen Index)
accessibility = gravity_accessibility(
    dists, weights, decay_method="exponential", beta=0.002, cutoff=1000.0
)
print("Erişilebilirlik Skorları:", accessibility)
```

### 2. Tesis Konumu Optimizasyonu (`planx.suitability`)
Belirli sayıda acil toplanma alanı veya sığınağı, maksimum kapsama mesafesini ve nüfusu gözeterek en optimum şekilde yerleştirmek için **MCLP** çözümünü uygular.

```python
import numpy as np
from planx.suitability import greedy_mclp

candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]) # Sığınak aday koordinatları
demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])   # Bina koordinatları
populations = np.array([100.0, 250.0, 500.0])                 # Bina nüfusları

# K=2 sığınak seç, maksimum yürüme mesafesi 5.0 birim olsun
selected, added_pop, cum_pop = greedy_mclp(candidates, demands, populations, max_distance=5.0, k=2)
print("Seçilen Tesis İndisleri:", selected) # [1, 0]
```

### 3. Sismik Hasar ve Enkaz Yayılımı (`planx.resilience`)
Monte Carlo simülasyonu yardımıyla, bir bölgedeki binaların inşa yılı, kat sayısı ve deprem büyüklüğüne göre yıkılma olasılıklarını stokastik olarak simüle eder. Oluşacak enkazın yatay yayılma yarıçapını ve hafriyat hacmini hesaplar.

```python
import numpy as np
from planx.resilience import simulate_seismic_debris

areas = np.array([120.0, 200.0, 80.0])  # Bina taban alanları (m2)
floors = np.array([4.0, 8.0, 2.0])     # Kat sayıları
years = np.array([1990, 2005, 2021])    # İnşa yılları

# 7.2 Mw deprem senaryosu
probs, collapsed, radii, volumes = simulate_seismic_debris(
    areas, floors, years, magnitude=7.2, seed=42
)
print("Bina Yıkım Durumları (0: Ayakta, 1: Yıkık):", collapsed)
print("Enkaz Yarıçapları (m):", radii)
```

### 4. Sosyal Kırılganlık Endeksi (SVI) (`planx.resilience`)
Nüfus yoğunluğu, yaşlılık oranı, çocuk nüfus oranı, düşük gelir seviyesi ve engellilik durumu gibi çeşitli demografik göstergeleri normalize ederek mekansal birimler için sosyal kırılganlık skorunu oluşturur.

```python
import numpy as np
from planx.resilience import social_vulnerability_index

# Gösterge verileri (örneğin 3 farklı mahalle için)
indicators = {
    "elderly": np.array([10.0, 50.0, 100.0]),
    "low_income": np.array([200.0, 100.0, 50.0])
}
# Gösterge ağırlıkları
weights = {
    "elderly": 0.5,
    "low_income": 0.5
}

scores, classes = social_vulnerability_index(indicators, weights)
print("Sosyal Kırılganlık Skorları:", scores)
print("Sınıflar:", classes) # ['Moderate', 'Moderate', 'Moderate']
```

### 5. Kentsel Isı Konforu Riski (`planx.resilience`)
Betonlaşma/geçirimsiz yüzey oranı, bina yoğunluğu, yeşil alan açığı, serinleme alanlarına olan yürüme mesafesi ve hassas nüfus noktalarını birleştirerek 0-100 aralığında bir kentsel sıcaklık maruziyet/risk skoru üretir.

```python
import numpy as np
from planx.resilience import urban_heat_comfort_risk

# 2x2 grid hücreleri için veriler
impervious = np.array([[0.8, 0.2], [0.5, 0.1]]) # Geçirimsiz yüzey oranı [0-1]
buildings = np.array([[0.6, 0.1], [0.4, 0.05]]) # Bina yoğunluğu oranı [0-1]
green = np.array([[0.1, 0.8], [0.3, 0.9]])      # Yeşil alan oranı [0-1]
cooling_dists = np.array([[300.0, 50.0], [200.0, 20.0]]) # Serinleme alanına olan mesafe (m)
vuln_assets = np.array([[2, 0], [1, 0]])        # Hassas tesis sayısı (okul, hastane vb.)

scores, classes = urban_heat_comfort_risk(
    impervious, buildings, green, cooling_dists, vuln_assets, cooling_distance=400.0
)
print("Isı Konfor Riski Skorları:\n", scores)
```

### 6. Karar Verme ve Ağırlıklandırma Metotları (`planx.suitability`)
Çok Kriterli Karar Verme (MCDA) süreçleri için AHP, Entropy, CRITIC ve PCA ağırlık hesaplama yöntemlerini barındırır. Ayrıca coğrafi katmanları (gridleri) karar matrisine dönüştüren yardımcı araçlar sunar.

```python
import numpy as np
from planx.suitability import ahp_weights, entropy_weights

# 1. AHP (Analitik Hiyerarşi Süreci) ile tutarlılık kontrolü ve ağırlık hesaplama
# 3x3 karşılaştırma matrisi
matrix = np.array([
    [1.0, 2.0, 3.0],
    [0.5, 1.0, 2.0],
    [0.33, 0.5, 1.0]
])
weights, cr = ahp_weights(matrix)
print("AHP Ağırlıkları:", weights)
print("Tutarlılık Oranı (CR):", cr)

# 2. Entropy Ağırlıklandırma Metodu
# 4 alternatif (örn. mahalle/hücre), 3 kriter içeren karar matrisi
decision_matrix = np.array([
    [10.0, 100.0, 0.1],
    [20.0, 50.0, 0.2],
    [15.0, 80.0, 0.15],
    [30.0, 20.0, 0.3]
])
ent_weights = entropy_weights(decision_matrix)
print("Entropy Ağırlıkları:", ent_weights)
```

---

## 🛠️ Kurulum ve Geliştirme (Installation & Development)

### 1. Standart Kurulum
Kütüphaneyi doğrudan PyPI üzerinden kurabilirsiniz:
```bash
pip install planx-sdk
```

### 2. Geliştirici Modunda Kurulum (Editable Install)
SDK kodlarında yaptığınız değişikliklerin anında çalışma ortamınıza yansıması için:
```bash
git clone https://github.com/YusufEminoglu/planx_sdk.git
cd planx_sdk
pip install -e .[dev]
```

---

## 🧪 Testler ve Kod Standartları

Kütüphanedeki tüm modüllerin matematiksel doğruluğu `pytest` birim testleri ile güvence altına alınmıştır. Testleri koşturmak için:
```bash
pytest
```

Kod kalitesi ve standartları `ruff` ve `black` araçları ile denetlenmektedir. Commit öncesi kodunuzu formatlamak ve denetlemek için:
```bash
black .
ruff check .
```

---

## 📝 Lisans (License)

Bu proje **MIT Lisansı** ile lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına göz atabilirsiniz.
