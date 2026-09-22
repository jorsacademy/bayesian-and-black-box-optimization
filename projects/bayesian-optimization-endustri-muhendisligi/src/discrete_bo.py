from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern


@dataclass
class AyrikOptimizasyonSonucu:
    """Ayrık Bayes optimizasyonunun temel sonucunu saklar."""

    en_iyi_x: np.ndarray
    en_iyi_y: float
    gozlenen_indeksler: list[int]
    y_gozlenen: np.ndarray
    gecmis: list[dict]


class AyrikGaussianProcessBayesOptimizer:
    """
    Sonlu bir aday kümesi üzerinde Gaussian Process tabanlı Bayes optimizasyonu.

    Endüstri mühendisliğinde personel sayısı, batch büyüklüğü veya sınırlı
    parametre kombinasyonları gibi ayrık kararlar için öğretici bir yapıdır.

    Bu yaklaşım her olası çözümü değerlendirmez. GP surrogate modeli, henüz
    denenmemiş adayların Expected Improvement değerini hesaplar ve sıradaki
    pahalı değerlendirmeyi seçer.
    """

    def __init__(
        self,
        amac_fonksiyonu: Callable[[np.ndarray], float],
        aday_noktalar: np.ndarray,
        baslangic_noktasi_sayisi: int = 8,
        xi: float = 0.01,
        alpha: float = 1e-8,
        random_state: int | None = 42,
        kernel: Kernel | None = None,
        gp_optimizer_tekrari: int = 2,
    ) -> None:
        self.amac_fonksiyonu = amac_fonksiyonu
        self.aday_noktalar = np.asarray(aday_noktalar, dtype=float)

        if self.aday_noktalar.ndim != 2:
            raise ValueError("aday_noktalar iki boyutlu bir dizi olmalıdır.")

        self.aday_sayisi, self.boyut_sayisi = self.aday_noktalar.shape
        self.baslangic_noktasi_sayisi = int(baslangic_noktasi_sayisi)

        if not 2 <= self.baslangic_noktasi_sayisi < self.aday_sayisi:
            raise ValueError(
                "baslangic_noktasi_sayisi en az 2 ve aday sayısından küçük olmalıdır."
            )

        self.xi = float(xi)
        self.rng = np.random.default_rng(random_state)

        self.alt = self.aday_noktalar.min(axis=0)
        self.ust = self.aday_noktalar.max(axis=0)
        self.aralik = np.where(self.ust > self.alt, self.ust - self.alt, 1.0)

        if kernel is None:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.full(self.boyut_sayisi, 0.3),
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

        self.gozlenen_indeksler: list[int] = []
        self.y_gozlenen: list[float] = []
        self.gecmis: list[dict] = []

    def _olcekle(self, X: np.ndarray) -> np.ndarray:
        """Aday noktaları [0, 1] aralığına ölçekler."""
        return (np.asarray(X, dtype=float) - self.alt) / self.aralik

    @staticmethod
    def _tek_sayiya_donustur(deger: float | np.ndarray) -> float:
        dizi = np.asarray(deger, dtype=float)
        if dizi.size != 1:
            raise ValueError("Amaç fonksiyonu tek bir sayısal değer döndürmelidir.")
        sonuc = float(dizi.reshape(-1)[0])
        if not np.isfinite(sonuc):
            raise ValueError("Amaç fonksiyonu sonlu bir değer döndürmelidir.")
        return sonuc

    def _expected_improvement(self, adaylar: np.ndarray) -> np.ndarray:
        """Minimizasyon için Expected Improvement değerlerini hesaplar."""
        mu, sigma = self.gp.predict(self._olcekle(adaylar), return_std=True)
        sigma = np.maximum(sigma, 1e-12)

        en_iyi_y = float(np.min(self.y_gozlenen))
        iyilesme = en_iyi_y - mu - self.xi
        z = iyilesme / sigma

        ei = iyilesme * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)

    def optimize_et(
        self,
        iterasyon_sayisi: int = 25,
        ayrintili: bool = False,
    ) -> AyrikOptimizasyonSonucu:
        """Ayrık Bayes optimizasyonu döngüsünü çalıştırır."""
        if not self.gozlenen_indeksler:
            baslangic = self.rng.choice(
                self.aday_sayisi,
                size=self.baslangic_noktasi_sayisi,
                replace=False,
            )

            for indeks in baslangic:
                indeks = int(indeks)
                y = self._tek_sayiya_donustur(
                    self.amac_fonksiyonu(self.aday_noktalar[indeks].copy())
                )
                self.gozlenen_indeksler.append(indeks)
                self.y_gozlenen.append(y)

        for i in range(int(iterasyon_sayisi)):
            X_gozlenen = self.aday_noktalar[self.gozlenen_indeksler]
            y_gozlenen = np.asarray(self.y_gozlenen, dtype=float)

            self.gp.fit(self._olcekle(X_gozlenen), y_gozlenen)

            denenmemis_maskesi = np.ones(self.aday_sayisi, dtype=bool)
            denenmemis_maskesi[self.gozlenen_indeksler] = False
            denenmemis_indeksler = np.flatnonzero(denenmemis_maskesi)

            if len(denenmemis_indeksler) == 0:
                break

            denenmemis_adaylar = self.aday_noktalar[denenmemis_indeksler]
            ei = self._expected_improvement(denenmemis_adaylar)

            secilen_yerel_indeks = int(np.argmax(ei))
            sonraki_indeks = int(denenmemis_indeksler[secilen_yerel_indeks])
            sonraki_x = self.aday_noktalar[sonraki_indeks].copy()
            sonraki_y = self._tek_sayiya_donustur(
                self.amac_fonksiyonu(sonraki_x)
            )

            self.gozlenen_indeksler.append(sonraki_indeks)
            self.y_gozlenen.append(sonraki_y)

            en_iyi_yerel_indeks = int(np.argmin(self.y_gozlenen))
            en_iyi_aday_indeksi = self.gozlenen_indeksler[en_iyi_yerel_indeks]

            kayit = {
                "iterasyon": i + 1,
                "x": sonraki_x.copy(),
                "y": float(sonraki_y),
                "en_iyi_x": self.aday_noktalar[en_iyi_aday_indeksi].copy(),
                "en_iyi_y": float(self.y_gozlenen[en_iyi_yerel_indeks]),
            }
            self.gecmis.append(kayit)

            if ayrintili:
                print(
                    f"İterasyon {i + 1:02d} | "
                    f"x = {sonraki_x.astype(int).tolist()} | "
                    f"y = {sonraki_y:.3f} | "
                    f"en iyi = {kayit['en_iyi_y']:.3f}"
                )

        en_iyi_yerel_indeks = int(np.argmin(self.y_gozlenen))
        en_iyi_aday_indeksi = self.gozlenen_indeksler[en_iyi_yerel_indeks]

        return AyrikOptimizasyonSonucu(
            en_iyi_x=self.aday_noktalar[en_iyi_aday_indeksi].copy(),
            en_iyi_y=float(self.y_gozlenen[en_iyi_yerel_indeks]),
            gozlenen_indeksler=list(self.gozlenen_indeksler),
            y_gozlenen=np.asarray(self.y_gozlenen, dtype=float),
            gecmis=list(self.gecmis),
        )
