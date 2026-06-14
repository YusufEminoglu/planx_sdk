# PlanX SDK (Software Development Kit)

PlanX QGIS eklenti ekosisteminin çekirdek mekansal analiz, istatistik ve yerleşilebilirlik (MCDA) hesaplama motorlarını barındıran, QGIS arayüzünden bağımsız, saf Python ve NumPy tabanlı resmi kütüphanesidir.

Bu kütüphane sayesinde, QGIS eklentilerinizdeki karmaşık algoritmaları:
1. QGIS'i başlatmadan (örneğin Jupyter Notebooks, bağımsız komut dosyaları veya sunucularda) çalıştırabilir,
2. `pytest` ile hızlı ve headless bir şekilde test edebilir,
3. Tek bir merkezden güncelleyerek tüm eklentilerinizde tutarlı bir şekilde kullanabilirsiniz.

---

## 📂 Proje Yapısı (Directory Structure)

```text
planx_sdk/
  ├── pyproject.toml              # Modern paket tanımı, bağımlılıklar ve metadata (PEP 517/621)
  ├── README.md                   # Bu dökümantasyon dosyası
  ├── LICENSE                     # Lisans dosyası
  ├── .gitignore                  # Python önbellek ve build klasörlerini yoksayan git kuralları
  ├── src/                        # Kaynak kodlar (Standard src-layout)
  │   └── planx/                  # Ana paket dizini
  │       ├── __init__.py         # Versiyon ve paket tanımı
  │       ├── spatial/            # Çekirdek mekansal ağ analizi ve space syntax motoru
  │       │   ├── __init__.py
  │       │   ├── paths.py        # En kısa yol algoritmaları (Dijkstra, SciPy entegrasyonu)
  │       │   └── centrality.py   # Yakınlık, Brandes Arasılık (Betweenness) ve Özdeğer (Eigenvector)
  │       ├── geostats/           # Mekansal istatistik ve otokorelasyon motorları
  │       │   ├── __init__.py
  │       │   └── stats_engines.py# Getis-Ord Gi*, Local/Global Moran's I, OLS, GWR, SDE, k-means
  │       └── suitability/        # Raster tabanlı MCDA (Multi-Criteria Decision Analysis) motoru
  │           ├── __init__.py
  │           └── mcda.py         # Normalizasyon metotları (Sigmoid, Gaussian, Min-Max) ve WLC
  └── tests/                      # Birim testler (Unit Tests)
      ├── __init__.py
      ├── test_spatial.py         # Network algoritmaları testleri
      ├── test_geostats.py        # İstatistik testleri
      └── test_suitability.py     # MCDA testleri
```

---

## 🛠️ Kurulum (Installation)

### 1. Geliştirici Modunda Kurulum (Editable Install - Geliştirme İçin)
SDK kodlarında yaptığınız değişikliklerin anında QGIS veya test ortamınızda yansıması için kütüphaneyi **geliştirici (editable) modda** kurmalısınız.

QGIS'in veya geliştirme yaptığınız IDE'nin (PyCharm/VS Code) aktif Python ortamında terminali açarak şu komutu çalıştırın:
```bash
pip install -e C:\Users\YE\PyCharmMiscProject\planx_sdk
```
*(Ya da `planx_sdk` klasörünün içindeyken: `pip install -e .`)*

### 2. Geliştirici Bağımlılıklarının Kurulumu
Birim testlerini koşmak, kod formatlamak veya lint araçlarını kullanmak için geliştirici paketlerini kurabilirsiniz:
```bash
pip install -e .[dev]
```

---

## 💡 QGIS Eklentilerinde Kullanım Stratejileri

QGIS eklentilerinizin bu paket ile haberleşmesi için iki temel yöntem mevcuttur:

### Yöntem A: Yerel Geliştirici Ortamı (Development)
Geliştirme makinenizde, QGIS'in kullandığı Python interpreter'ına (`OSGeo4W` veya `conda` ortamı) SDK'yı yukarıdaki gibi `pip install -e` ile kurun. 
Eklenti kodunuzda doğrudan standart importları yapabilirsiniz:
```python
from planx.spatial import brandes_betweenness
from planx.geostats import calculate_getis_ord
```

### Yöntem B: Dağıtım / Yayınlama Süreci (Vendoring / Bundling)
QGIS Hub'da yayınlarken, eklenti zip paketinin kendi kendine yetmesi gerekir (kullanıcıdan harici `pip install` yapması beklenemez).
Bu sorunu çözmek için, `Build-PluginZip.ps1` veya release betiğinizin içine SDK'yı eklentinin kendi `libs` klasörüne yükleyen bir adım ekleyebilirsiniz:

```powershell
# Eklentinin zip paketleme aşamasında target klasör olarak eklenti altındaki 'libs' belirlenir:
pip install C:\Users\YE\PyCharmMiscProject\planx_sdk -t C:\Users\YE\PyCharmMiscProject\qgis_plugins\<eklenti_klasoru>\libs --upgrade
```

Eklentinizin giriş noktasında (`__init__.py` veya ana modülün en üstünde) bu `libs` klasörünü Python arama yoluna (`sys.path`) eklemeniz yeterlidir:
```python
import os
import sys

# Eklenti altındaki 'libs' klasörünü sys.path'e en yüksek öncelikle ekle
libs_path = os.path.join(os.path.dirname(__file__), 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

# Artık SDK sorunsuzca import edilebilir
from planx.spatial import many_to_many
```

---

## 🧪 Testleri Koşturma (Running Tests)

SDK'nın doğruluğunu test etmek için `pytest` kullanabilirsiniz. Proje kök dizininde (`C:\Users\YE\PyCharmMiscProject\planx_sdk`) şu komutu çalıştırmanız yeterlidir:

```bash
pytest
```

---

## 📝 Lisans (License)

Bu proje **MIT Lisansı** ile lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına göz atabilirsiniz.
