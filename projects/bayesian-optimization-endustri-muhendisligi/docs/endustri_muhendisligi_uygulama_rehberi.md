# Endüstri Mühendisliğinde Bayes Optimizasyonu Uygulama Rehberi

## Temel karar sorusu

Bir problemi gördüğünüzde önce şu soruyu sorun:

> Bir aday çözümün performansını hesaplamak ucuz mu, pahalı mı?

Eğer performansı tek bir formülle, LP/MILP modeliyle veya hızlı bir algoritmayla kolayca hesaplayabiliyorsanız Bayes optimizasyonu çoğu zaman ilk tercih değildir.

Eğer her aday çözüm için:

- uzun bir simülasyon,
- fiziksel üretim deneyi,
- pahalı mühendislik analizi,
- saatler süren makine öğrenmesi eğitimi,
- yüksek maliyetli kalite testi

gerekiyorsa Bayes optimizasyonu daha anlamlı hale gelir.

## 1. İmalat proses parametreleri

Örnek karar değişkenleri:

- CNC devir sayısı
- ilerleme hızı
- kesme derinliği
- kaynak akımı
- kaynak hızı
- fırın sıcaklığı
- bekletme süresi
- enjeksiyon basıncı
- kalıp sıcaklığı
- soğutma süresi

Amaçlar:

- yüzey pürüzlülüğünü azaltmak
- çevrim süresini azaltmak
- enerji kullanımını azaltmak
- fire oranını düşürmek
- kalite-maliyet dengesini iyileştirmek

Bayes optimizasyonunun avantajı, tam faktöriyel deney tasarımına göre daha az pahalı deneyle iyi bölgeleri bulmaya çalışmasıdır.

## 2. Simülasyon tabanlı üretim sistemi optimizasyonu

Karar değişkenleri:

- istasyon başına operatör sayısı
- makine sayısı
- buffer kapasitesi
- batch büyüklüğü
- taşıma aracı sayısı
- bakım eşikleri
- dispatching rule parametreleri

Amaçlar:

- throughput artırmak
- WIP azaltmak
- ortalama akış süresini azaltmak
- bekleme süresini azaltmak
- toplam işletme maliyetini azaltmak

Bu tür problemlerde amaç fonksiyonu çoğu zaman Arena, AnyLogic, SimPy veya özel bir ayrık olay simülasyonu modelinden gelir.

## 3. Bakım optimizasyonu

Karar değişkenleri:

- periyodik bakım aralığı
- condition monitoring alarm eşiği
- yedek parça seviyesi
- planlı duruş süresi
- bakım ekibi kapasitesi

Amaç:

```text
beklenen arıza maliyeti
+ planlı bakım maliyeti
+ üretim kaybı
+ stok maliyeti
```

Bu problemde güvenilirlik modeli veya simülasyon pahalıysa BO uygundur.

## 4. Stok politikalarının ayarlanması

Örnek karar değişkenleri:

- yeniden sipariş noktası `r`
- sipariş miktarı `Q`
- base-stock seviyesi `S`
- güvenlik stoğu katsayısı

Talep ve lead time stokastik olduğunda, bir politikanın performansı Monte Carlo simülasyonuyla tahmin edilebilir.

Amaç:

```text
elde bulundurma maliyeti
+ sipariş maliyeti
+ stockout/backorder maliyeti
```

Burada BO, politika parametrelerini ayarlamak için kullanılabilir.

## 5. Çizelgelemede nerede kullanılır?

Klasik job-shop veya flow-shop problemini doğrudan BO ile çözmek genellikle doğru başlangıç değildir.

Daha uygun kullanım:

- genetik algoritmanın mutasyon oranını ayarlamak
- tabu search tenure değerini ayarlamak
- dispatching rule ağırlıklarını ayarlamak
- simülasyon tabanlı çizelgeleme politikasının parametrelerini ayarlamak

Yani BO çoğu zaman kombinatoryal çizelgeleme probleminin kendisini değil, çözüm yöntemini veya politika parametrelerini optimize eder.

## 6. Kalite ve deney tasarımı

Her fiziksel deney pahalıysa BO, sequential design of experiments yaklaşımı olarak düşünülebilir.

Örneğin:

```text
sıcaklık -> kalite
basınç -> kalite
hız -> kalite
malzeme oranı -> kalite
```

İlk birkaç deneyden sonra GP bir response surface oluşturur. Edinim fonksiyonu sıradaki deneyin nerede yapılacağına karar verir.

## 7. Enerji verimliliği

Karar değişkenleri:

- makine hızları
- HVAC set-point
- fırın profili
- kompresör basıncı
- üretim hızı

Amaç:

- enerji tüketimini azaltmak
- kaliteyi korumak
- throughput kısıtlarını sağlamak

Pahalı dijital ikiz veya fiziksel test varsa BO iyi bir adaydır.

## 8. BO ile klasik Yöneylem Araştırması ilişkisi

Bayes optimizasyonu, LP/MILP/CP-SAT/Gurobi/Pyomo gibi yöntemlerin genel bir alternatifi değildir.

Örnek ayrım:

| Problem | İlk tercih |
|---|---|
| Deterministik ürün karması LP | LP |
| Kapasiteli üretim planlama MILP | MILP |
| Küçük/orta çizelgeleme CP modeli | CP-SAT |
| Her çözümü 20 dakika süren üretim simülasyonu | Bayes optimizasyonu düşünülebilir |
| Fiziksel CNC deneyi başına yüksek maliyet | Bayes optimizasyonu güçlü aday |
| ML model hiperparametre ayarı | Bayes optimizasyonu güçlü aday |

## 9. Boyut problemi

Klasik GP tabanlı BO düşük ve orta boyutlarda güçlüdür. Karar değişkeni sayısı büyüdükçe:

- acquisition optimization zorlaşır,
- GP kernel öğrenimi zorlaşır,
- gereken deney sayısı artar.

Çok yüksek boyutlarda TPE, random forest surrogate, evolutionary optimization veya problem-özel yöntemler daha uygun olabilir.

## 10. Uygulama kontrol listesi

Bir projede BO kullanmadan önce:

1. Amaç fonksiyonunu net tanımlayın.
2. Karar değişkenlerinin tipini belirleyin: sürekli, tamsayı, kategorik.
3. Her değerlendirmenin maliyetini hesaplayın.
4. Gürültünün kaynağını belirleyin.
5. Kısıtları nasıl ele alacağınızı tanımlayın.
6. Amaç bileşenlerini aynı birime getirin veya açık ağırlıklandırma kullanın.
7. Küçük bir örnekte exhaustive/random search ile doğrulama yapın.
8. Seed ve replikasyon yapısını kaydedin.
9. Bulunan çözümü ek bağımsız simülasyon/deney replikasyonlarıyla doğrulayın.
10. Bayes optimizasyonunun global optimum garantisi vermediğini raporlayın.
