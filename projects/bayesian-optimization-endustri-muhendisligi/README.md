# Endüstri Mühendisliği için Bayes Optimizasyonu

Bu depo, Gaussian Process Regression (GPR) ile Gaussian Process tabanlı Bayes optimizasyonunu endüstri mühendisliği bağlamında öğretmek için hazırlanmıştır.

Ana amaç, `sklearn.gaussian_process.GaussianProcessRegressor` sınıfının tek başına bir optimizasyon algoritması olmadığını; Bayes optimizasyonunda bir **surrogate model** olarak nasıl kullanıldığını açık biçimde göstermektir.

## İçerik

- Gaussian Process Regression temelleri
- GPR ile Bayes optimizasyonu arasındaki fark
- Expected Improvement, Probability of Improvement ve Lower Confidence Bound
- Sürekli karar değişkenleri için sıfırdan yazılmış GP tabanlı Bayes optimizasyonu
- Ayrık karar değişkenleri için aday-küme tabanlı GP Bayes optimizasyonu
- Vardiya/personel kapasitesi için simülasyon tabanlı optimizasyon
- İmalat proses parametreleri için sentetik black-box optimizasyon örneği
- Endüstri mühendisliğinde hangi problemlerde kullanılmalı, hangi problemlerde kullanılmamalı

## Temel ayrım

`GaussianProcessRegressor` bir regresyon modelidir. Gözlenen verilere göre tahmin ortalaması ve tahmin belirsizliği üretir.

Bayes optimizasyonu ise bu belirsizliği bir **edinim fonksiyonu** aracılığıyla kullanarak sıradaki pahalı değerlendirme noktasını seçen iteratif bir optimizasyon yaklaşımıdır.

Kısaca:

```text
Gaussian Process Regression
        |
        v
Surrogate model: mu(x), sigma(x)
        |
        v
Acquisition function
        |
        v
Sıradaki x seçilir
        |
        v
Gerçek/simülasyon amaç fonksiyonu çalıştırılır
        |
        v
Veri güncellenir ve döngü devam eder
```

## Depo yapısı

```text
.
├── README.md
├── LICENSE.md
├── requirements.txt
├── src/
│   ├── gaussian_bo.py
│   ├── discrete_bo.py
│   └── uretim_simulasyonu.py
├── notebooks/
│   ├── 00_gaussian_process_regression_temelleri.ipynb
│   ├── 01_bayesian_optimizasyon_temelleri.ipynb
│   ├── 02_vardiya_personel_optimizasyonu.ipynb
│   └── 03_imalat_proses_parametre_optimizasyonu.ipynb
└── docs/
    ├── kod_denetim_notlari.md
    └── endustri_muhendisligi_uygulama_rehberi.md
```

## Kurulum

Python 3.10 veya daha yeni bir sürüm önerilir.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell için:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Google Colab üzerinde depo klonlandıktan sonra notebook dosyaları doğrudan çalıştırılabilir.

## Neden `scikit-learn` ile sıfırdan bir Bayes optimizasyonu döngüsü var?

`scikit-learn`, `GaussianProcessRegressor` ve kernel bileşenlerini sağlar; ancak genel amaçlı bir `BayesianOptimization` sınıfı sağlamaz. Bu depoda algoritmanın mekanizmasını görünür tutmak için acquisition function ve sequential sampling döngüsü doğrudan yazılmıştır.

Daha yüksek seviyeli alternatiflerden biri `scikit-optimize` paketindeki `gp_minimize` fonksiyonudur. Bu paket `scikit-learn` paketinin bir modülü değildir; ayrı bir projedir.

## Endüstri mühendisliği açısından ne zaman uygundur?

Bayes optimizasyonu özellikle şu durumda anlamlıdır:

- Bir çözümü değerlendirmek pahalıysa
- Amaç fonksiyonunun kapalı formu yoksa
- Değerlendirme simülasyon, fiziksel deney veya karmaşık yazılım üzerinden yapılıyorsa
- Türev bilgisi yoksa veya güvenilir değilse
- Karar değişkeni sayısı düşük ya da orta düzeydeyse
- Her yeni deneyden mümkün olduğunca fazla bilgi elde etmek isteniyorsa

Örnekler:

- CNC, kaynak, enjeksiyon, fırın veya ısıl işlem parametre ayarı
- Simülasyon tabanlı personel ve kapasite planlama
- Bakım periyodu ve bakım eşiklerinin ayarlanması
- Stok politikası parametrelerinin simülasyonla ayarlanması
- Üretim hattı simülasyonu parametrelerinin kalibrasyonu
- Enerji tüketimi ile kalite arasındaki proses dengesi
- Pahalı çözüm algoritmalarının hiperparametrelerinin ayarlanması

## Ne zaman ilk tercih değildir?

Aşağıdaki problemlerde klasik Yöneylem Araştırması yöntemleri çoğu zaman daha doğrudan ve güçlüdür:

- Doğrusal programlama
- MILP/MINLP biçiminde açıkça modellenebilen üretim planlama problemleri
- CP-SAT ile doğal biçimde ifade edilen çizelgeleme problemleri
- Çok büyük kombinatoryal uzaylarda tam veya güçlü özel amaçlı çözücülerin bulunduğu problemler

Bayes optimizasyonu bu tür modellerin yerine otomatik olarak geçmez. Ancak simülasyon, fiziksel deney veya çözücü parametresi gibi pahalı bir dış değerlendirme katmanı varsa yardımcı bir üst-seviye optimizasyon yöntemi olabilir.

## Lisans

Bu depo ticari olmayan kullanım amacıyla yayımlanacak şekilde hazırlanmıştır. Ayrıntılar için `LICENSE.md` dosyasına bakın.
