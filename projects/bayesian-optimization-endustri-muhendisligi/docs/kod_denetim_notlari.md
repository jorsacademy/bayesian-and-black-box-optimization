# Kod Denetim Notları

Bu belge, daha önce yazılmış Gaussian Process / Bayes optimizasyonu kodlarının teknik denetimini özetler.

## 1. Doğru olan ana fikir

Eski kodun temel mimarisi doğrudur:

1. `GaussianProcessRegressor` surrogate model olarak kullanılır.
2. Gözlenen `X, y` verileriyle GP yeniden eğitilir.
3. GP'den `mu(x)` ve `sigma(x)` elde edilir.
4. Expected Improvement, Probability of Improvement veya güven sınırı türünde bir edinim fonksiyonu hesaplanır.
5. Edinim fonksiyonu optimize edilerek sıradaki pahalı deney noktası seçilir.
6. Gerçek amaç fonksiyonu bu noktada değerlendirilir.
7. Veri kümesi güncellenir ve süreç tekrarlanır.

Bu yapı Bayes optimizasyonunun standart sequential surrogate optimization mantığıyla uyumludur.

## 2. Eski özel `BayesianOptimization` sınıfındaki önemli sorunlar

### 2.1. `bounds.shape[0]` kullanımı

Eski kodda:

```python
self.bounds = np.array(bounds)
self.n_dimensions = bounds.shape[0]
```

`bounds` bir Python listesi verilirse `shape` özelliği bulunmayabilir.

Düzeltilmiş yaklaşım:

```python
self.bounds = np.asarray(bounds, dtype=float)
self.n_dimensions = self.bounds.shape[0]
```

### 2.2. LCB/UCB işaret hatası

Eski kod minimizasyon problemi için:

```python
return mu - kappa * sigma
```

hesabını yapıyor, fakat daha sonra edinim fonksiyonunu **maksimize** ediyordu. `mu - kappa*sigma` Lower Confidence Bound'dur ve minimizasyon için **minimize edilmesi** gerekir.

Düzeltilmiş depoda tüm edinim ölçütleri "büyük skor daha iyi" biçimine çevrilmiştir:

```python
lcb = mu - kappa * sigma
score = -lcb
```

Böylece ortak maximization mantığı tutarlı hale gelir.

### 2.3. `nu=2.5` açıklaması

Eski yorumda Matérn `nu=2.5` için "once differentiable" denmişti. Bu doğru değildir.

Scikit-learn dokümantasyonundaki standart yorum:

- `nu=1.5`: yaklaşık bir kez türevlenebilir fonksiyonlar
- `nu=2.5`: yaklaşık iki kez türevlenebilir fonksiyonlar

### 2.4. Amaç fonksiyonunun dizi döndürmesi

Eski 1 boyutlu örnekte `x` bir NumPy dizisi olduğu için amaç fonksiyonu bazen şekli `(1,)` olan bir dizi döndürüyordu. Bu durum `y_observed` boyutlarının tutarsızlaşmasına yol açabilir.

Yeni kod, her amaç fonksiyonu çıktısını tek bir `float` olarak doğrular.

### 2.5. Tekrarlanabilirlik

Eski kodda:

- başlangıç noktaları,
- acquisition optimizer başlangıçları,
- gürültülü amaç fonksiyonu

için ortak bir `random_state` yönetimi yoktu.

Yeni kod `numpy.random.default_rng(random_state)` kullanır ve GP modeline de aynı rastgelelik kontrolü verilir.

### 2.6. Acquisition optimization'ın kırılganlığı

Eski sürüm yalnızca 10 rastgele L-BFGS-B başlangıcı kullanıyordu.

Yeni sürüm:

1. Binlerce rastgele acquisition adayı tarar.
2. En iyi adaylardan birkaçını yerel optimizasyona başlangıç olarak verir.
3. En iyi sonucu seçer.

Bu yapı düşük boyutlu problemlerde daha dayanıklıdır.

### 2.7. Karar değişkeni ölçekleri

Örneğin devir sayısı 1000–5000, kesme derinliği 0.1–3.0 gibi çok farklı ölçeklerdeyse GP length-scale optimizasyonu zorlaşabilir.

Yeni sürekli optimizer, GP girişlerini dahili olarak `[0, 1]` aralığına ölçekler.

## 3. Eski vardiya/personel örneğindeki sorunlar

### 3.1. Sadece bekleme süresini minimize etmek

İlk personel örneğinde yalnızca toplam bekleme süresi minimize edildiği için optimizer'ın her vardiyada üst sınıra, yani `10, 10, 10` çözümüne gitmesi beklenen bir sonuçtur.

Bu bir Bayes optimizasyonu başarısından çok, amaç fonksiyonunun doğal sonucudur.

Gerçekçi personel planlamasında personel maliyeti veya personel bütçesi gibi karşı ağırlık gerekir.

### 3.2. Birimlerin doğrudan toplanması

Daha sonraki örnekte:

```text
bekleme süresi + personel maliyeti + ceza
```

doğrudan toplandı. Dakika ile para birimini doğrudan toplamak fiziksel/ekonomik açıdan tutarlı değildir.

Yeni örnekte bekleme süresi önce:

```text
bekleme dakikası x bekleme dakika maliyeti
```

ile parasal maliyete dönüştürülür.

### 3.3. `np.random.seed()` kullanımı

Simülasyon fonksiyonunun her çağrısında `np.random.seed()` çağırmak tekrarlanabilirliği zayıflatır ve deney tasarımını kontrol etmeyi zorlaştırır.

Yeni örnekte açık seed listeleri ve `default_rng` kullanılır.

### 3.4. Ortak rastgele sayılar

Yeni personel örneğinde her aday politika aynı replikasyon seed'leriyle değerlendirilir. Bu, common random numbers yaklaşımının basit bir uygulamasıdır.

Amaç, iki personel politikasını farklı rastgele senaryolar yerine aynı talep/hizmet senaryolarında karşılaştırarak varyansı azaltmaktır.

### 3.5. Vardiya başlangıç saatleri gerçekte optimize edilmiyordu

Eski geliştirilmiş kodda `shift_starts = [0, 240, 480]` değişkeni tanımlanmış olsa da:

- optimizasyon karar değişkenlerine eklenmemişti,
- simülasyon mantığında aktif kullanılmıyordu.

Dolayısıyla "vardiya başlangıç saatleri de optimize edildi" ifadesi doğru değildi.

Yeni notebook bu konuda açık davranır: temel personel örneğinde vardiya başlangıç saatleri sabittir ve optimize edilmez.

Vardiya başlangıçlarını optimize etmek için vardiya örtüşmelerini, gün içi zaman eksenini ve personel kapasitesinin zamana bağlı değişimini temsil eden farklı bir simülasyon modeli gerekir.

## 4. `scikit-optimize` ile yazılan eski `gp_minimize` örnekleri

Temel kullanım fikri doğrudur:

```python
from skopt import gp_minimize
```

`gp_minimize`, Gaussian Process tabanlı Bayes optimizasyonu uygular.

Ancak iki nokta önemlidir:

1. `scikit-optimize`, scikit-learn'in alt modülü değildir; ayrı bir pakettir.
2. Personel sayısı gibi ayrık değişkenlerde arama uzayını açık `Integer` boyutlarıyla tanımlamak daha okunaklıdır.

Örnek:

```python
from skopt.space import Integer

boyutlar = [
    Integer(1, 10, name="vardiya_1"),
    Integer(1, 10, name="vardiya_2"),
    Integer(1, 10, name="vardiya_3"),
]
```

Bu repository'nin ana uygulamaları eğitim amacıyla doğrudan `GaussianProcessRegressor` üzerinden kurulmuştur. Böylece GP ile Bayes optimizasyonu arasındaki ilişki saklanmaz.
