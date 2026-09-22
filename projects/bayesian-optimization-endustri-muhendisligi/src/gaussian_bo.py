from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern


@dataclass
class OptimizasyonSonucu:
    """Bayes optimizasyonunun temel sonucunu saklar."""

    en_iyi_x: np.ndarray
    en_iyi_y: float
    X_gozlenen: np.ndarray
    y_gozlenen: np.ndarray
    gecmis: list[dict]


class GaussianProcessBayesOptimizer:
    """
    Sürekli ve sınırlı karar değişkenleri için Gaussian Process tabanlı
    Bayes optimizasyonu.

    Bu sınıf minimizasyon yapar. GaussianProcessRegressor yalnızca surrogate
    modeldir; Bayes optimizasyonu, acquisition function ve iteratif örnekleme
    mantığı bu sınıfta kurulmaktadır.

    Parametreler
    ------------
    amac_fonksiyonu:
        Tek boyutlu NumPy dizisi alıp tek bir sayısal değer döndüren fonksiyon.
    sinirlar:
        Her karar değişkeni için [alt_sinir, ust_sinir] çiftleri.
    baslangic_noktasi_sayisi:
        GP kurulmadan önce rastgele değerlendirilecek nokta sayısı.
    edinim_fonksiyonu:
        "ei", "pi" veya "lcb".
    xi:
        EI ve PI için keşif katsayısı.
    kappa:
        LCB için belirsizlik katsayısı.
    alpha:
        GP kovaryans matrisinin köşegenine eklenen değer. Sayısal kararlılık
        veya bilinen gözlem gürültüsü için kullanılabilir.
    random_state:
        Tekrarlanabilirlik için rastgelelik tohumu.
    kernel:
        İstenirse özel scikit-learn GP kerneli.
    """

    def __init__(
        self,
        amac_fonksiyonu: Callable[[np.ndarray], float],
        sinirlar: Iterable[Iterable[float]],
        baslangic_noktasi_sayisi: int = 6,
        edinim_fonksiyonu: str = "ei",
        xi: float = 0.01,
        kappa: float = 2.0,
        alpha: float = 1e-8,
        random_state: int | None = 42,
        kernel: Kernel | None = None,
        gp_optimizer_tekrari: int = 3,
    ) -> None:
        self.amac_fonksiyonu = amac_fonksiyonu
        self.sinirlar = np.asarray(sinirlar, dtype=float)

        if self.sinirlar.ndim != 2 or self.sinirlar.shape[1] != 2:
            raise ValueError("sinirlar (n_boyut, 2) biçiminde olmalıdır.")
        if np.any(self.sinirlar[:, 0] >= self.sinirlar[:, 1]):
            raise ValueError("Her alt sınır üst sınırdan küçük olmalıdır.")

        self.boyut_sayisi = self.sinirlar.shape[0]
        self.baslangic_noktasi_sayisi = int(baslangic_noktasi_sayisi)
        if self.baslangic_noktasi_sayisi < 2:
            raise ValueError("baslangic_noktasi_sayisi en az 2 olmalıdır.")

        self.edinim_fonksiyonu = edinim_fonksiyonu.lower()
        if self.edinim_fonksiyonu not in {"ei", "pi", "lcb"}:
            raise ValueError("edinim_fonksiyonu 'ei', 'pi' veya 'lcb' olmalıdır.")

        self.xi = float(xi)
        self.kappa = float(kappa)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if kernel is None:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.full(self.boyut_sayisi, 0.2),
                length_scale_bounds=(1e-3, 1e3),
                nu=2.5,
            )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=int(gp_optimizer_tekrari),
            random_state=random_state,
        )

        self.X_gozlenen: np.ndarray | None = None
        self.y_gozlenen: np.ndarray | None = None
        self.gecmis: list[dict] = []

    def _birim_araliga_donustur(self, X: np.ndarray) -> np.ndarray:
        """Karar değişkenlerini [0, 1] aralığına ölçekler."""
        X = np.asarray(X, dtype=float)
        alt = self.sinirlar[:, 0]
        ust = self.sinirlar[:, 1]
        return (X - alt) / (ust - alt)

    def _orijinal_araliga_donustur(self, U: np.ndarray) -> np.ndarray:
        """[0, 1] uzayındaki noktaları gerçek karar değişkeni uzayına taşır."""
        U = np.asarray(U, dtype=float)
        alt = self.sinirlar[:, 0]
        ust = self.sinirlar[:, 1]
        return alt + U * (ust - alt)

    @staticmethod
    def _tek_sayiya_donustur(deger: float | np.ndarray) -> float:
        """Amaç fonksiyonunun gerçekten tek bir sonlu sayı döndürmesini doğrular."""
        dizi = np.asarray(deger, dtype=float)
        if dizi.size != 1:
            raise ValueError("Amaç fonksiyonu tek bir sayısal değer döndürmelidir.")
        sonuc = float(dizi.reshape(-1)[0])
        if not np.isfinite(sonuc):
            raise ValueError("Amaç fonksiyonu sonlu bir değer döndürmelidir.")
        return sonuc

    def _edinim_skorlari(self, U: np.ndarray) -> np.ndarray:
        """
        Verilen birim-uzay noktaları için edinim skorunu hesaplar.

        Tüm skorlar 'büyük olan daha iyi' biçiminde tanımlanmıştır. Bu sayede
        EI ve PI doğrudan maksimize edilir. LCB ise önce minimizasyon ölçütü
        olarak hesaplanır, sonra işareti çevrilerek maksimize edilecek skora
        dönüştürülür.
        """
        if self.y_gozlenen is None:
            raise RuntimeError("Önce en az bir gözlem yapılmalıdır.")

        U = np.atleast_2d(np.asarray(U, dtype=float))
        mu, sigma = self.gp.predict(U, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        en_iyi_y = float(np.min(self.y_gozlenen))

        if self.edinim_fonksiyonu == "ei":
            iyilesme = en_iyi_y - mu - self.xi
            z = iyilesme / sigma
            ei = iyilesme * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0.0)

        if self.edinim_fonksiyonu == "pi":
            z = (en_iyi_y - mu - self.xi) / sigma
            return norm.cdf(z)

        lcb = mu - self.kappa * sigma
        return -lcb

    def _sonraki_noktayi_sec(
        self,
        aday_sayisi: int = 5000,
        yerel_baslangic_sayisi: int = 12,
    ) -> np.ndarray:
        """
        Edinim fonksiyonunu önce rastgele adaylarla tarar, sonra iyi adayları
        L-BFGS-B ile yerel olarak iyileştirir.
        """
        adaylar = self.rng.random((int(aday_sayisi), self.boyut_sayisi))
        skorlar = self._edinim_skorlari(adaylar)

        kac_baslangic = min(int(yerel_baslangic_sayisi), len(adaylar))
        iyi_indeksler = np.argsort(skorlar)[-kac_baslangic:]

        en_iyi_u = adaylar[iyi_indeksler[-1]].copy()
        en_iyi_skor = float(skorlar[iyi_indeksler[-1]])
        birim_sinirlar = [(0.0, 1.0)] * self.boyut_sayisi

        for indeks in iyi_indeksler:
            u0 = adaylar[indeks]

            sonuc = minimize(
                lambda u: -float(
                    self._edinim_skorlari(np.asarray(u).reshape(1, -1))[0]
                ),
                u0,
                bounds=birim_sinirlar,
                method="L-BFGS-B",
            )

            if sonuc.success:
                skor = -float(sonuc.fun)
                if skor > en_iyi_skor:
                    en_iyi_skor = skor
                    en_iyi_u = np.clip(sonuc.x, 0.0, 1.0)

        # Aynı noktayı yeniden seçme riskini azaltır.
        if self.X_gozlenen is not None:
            gozlenen_u = self._birim_araliga_donustur(self.X_gozlenen)
            uzakliklar = np.linalg.norm(gozlenen_u - en_iyi_u, axis=1)

            if np.min(uzakliklar) < 1e-8:
                sirali = np.argsort(skorlar)[::-1]
                for indeks in sirali:
                    aday = adaylar[indeks]
                    aday_uzakligi = np.min(
                        np.linalg.norm(gozlenen_u - aday, axis=1)
                    )
                    if aday_uzakligi >= 1e-8:
                        en_iyi_u = aday
                        break

        return self._orijinal_araliga_donustur(en_iyi_u)

    def optimize_et(
        self,
        iterasyon_sayisi: int = 20,
        ayrintili: bool = False,
    ) -> OptimizasyonSonucu:
        """Bayes optimizasyonu döngüsünü çalıştırır."""
        if self.X_gozlenen is None:
            baslangic_u = self.rng.random(
                (self.baslangic_noktasi_sayisi, self.boyut_sayisi)
            )
            self.X_gozlenen = self._orijinal_araliga_donustur(baslangic_u)
            self.y_gozlenen = np.array(
                [
                    self._tek_sayiya_donustur(self.amac_fonksiyonu(x.copy()))
                    for x in self.X_gozlenen
                ],
                dtype=float,
            )

        for i in range(int(iterasyon_sayisi)):
            self.gp.fit(
                self._birim_araliga_donustur(self.X_gozlenen),
                self.y_gozlenen,
            )

            sonraki_x = self._sonraki_noktayi_sec()
            sonraki_y = self._tek_sayiya_donustur(
                self.amac_fonksiyonu(sonraki_x.copy())
            )

            self.X_gozlenen = np.vstack([self.X_gozlenen, sonraki_x])
            self.y_gozlenen = np.append(self.y_gozlenen, sonraki_y)

            en_iyi_indeks = int(np.argmin(self.y_gozlenen))
            kayit = {
                "iterasyon": i + 1,
                "x": sonraki_x.copy(),
                "y": float(sonraki_y),
                "en_iyi_x": self.X_gozlenen[en_iyi_indeks].copy(),
                "en_iyi_y": float(self.y_gozlenen[en_iyi_indeks]),
            }
            self.gecmis.append(kayit)

            if ayrintili:
                print(
                    f"İterasyon {i + 1:02d} | "
                    f"y = {sonraki_y:.6f} | "
                    f"en iyi = {kayit['en_iyi_y']:.6f}"
                )

        en_iyi_indeks = int(np.argmin(self.y_gozlenen))

        return OptimizasyonSonucu(
            en_iyi_x=self.X_gozlenen[en_iyi_indeks].copy(),
            en_iyi_y=float(self.y_gozlenen[en_iyi_indeks]),
            X_gozlenen=self.X_gozlenen.copy(),
            y_gozlenen=self.y_gozlenen.copy(),
            gecmis=list(self.gecmis),
        )
