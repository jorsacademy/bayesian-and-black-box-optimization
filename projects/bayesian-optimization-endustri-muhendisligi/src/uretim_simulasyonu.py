from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable

import numpy as np


@dataclass
class SimulasyonOzeti:
    """Bir personel politikasının tekrarlı simülasyon özetini saklar."""

    toplam_maliyet: float
    personel_maliyeti: float
    bekleme_maliyeti: float
    hizmet_seviyesi_cezasi: float
    ortalama_bekleme_dakika: float
    vardiya_bazli_ortalama_bekleme: np.ndarray
    ortalama_musteri_sayisi: float


def tek_gun_simule_et(
    personel: Iterable[int],
    seed: int,
    vardiya_suresi_dakika: int = 240,
    gelis_hizlari: tuple[float, float, float] = (1.0, 1.6, 1.2),
    ortalama_hizmet_suresi: float = 5.0,
    hizmet_suresi_std: float = 1.2,
) -> dict:
    """
    Üç vardiyalı basitleştirilmiş kuyruk sisteminin bir günlük koşumunu yapar.

    Her vardiya ayrı bir kuyruk olarak ele alınır. Bu nedenle vardiya başlangıç
    saatleri bu temel modelde karar değişkeni değildir. Amaç, personel sayısı ile
    bekleme performansı arasındaki dengeyi incelemektir.
    """
    personel = np.asarray(personel, dtype=int)

    if personel.shape != (3,):
        raise ValueError("personel üç elemanlı olmalıdır.")
    if np.any(personel < 1):
        raise ValueError("Her vardiyada en az bir personel olmalıdır.")
    if len(gelis_hizlari) != 3:
        raise ValueError("gelis_hizlari üç vardiya için üç değer içermelidir.")

    rng = np.random.default_rng(seed)

    toplam_bekleme = 0.0
    toplam_musteri = 0
    vardiya_ortalamalari: list[float] = []

    for vardiya, personel_sayisi in enumerate(personel):
        dakika_bazli_gelis = rng.poisson(
            lam=gelis_hizlari[vardiya],
            size=vardiya_suresi_dakika,
        )
        musteri_sayisi = int(dakika_bazli_gelis.sum())

        hizmet_sureleri = np.clip(
            rng.normal(
                loc=ortalama_hizmet_suresi,
                scale=hizmet_suresi_std,
                size=musteri_sayisi,
            ),
            1.0,
            12.0,
        )

        # Min-heap, her sunucunun bir sonraki boşalma zamanını tutar.
        sunucu_bosalma = [0.0] * int(personel_sayisi)
        heapq.heapify(sunucu_bosalma)

        vardiya_bekleme = 0.0
        musteri_indeksi = 0

        for dakika, gelen_sayi in enumerate(dakika_bazli_gelis):
            for _ in range(int(gelen_sayi)):
                en_erken_bosalma = heapq.heappop(sunucu_bosalma)
                hizmete_baslama = max(float(dakika), en_erken_bosalma)
                bekleme = hizmete_baslama - float(dakika)

                vardiya_bekleme += bekleme
                yeni_bosalma = (
                    hizmete_baslama + float(hizmet_sureleri[musteri_indeksi])
                )
                heapq.heappush(sunucu_bosalma, yeni_bosalma)
                musteri_indeksi += 1

        toplam_bekleme += vardiya_bekleme
        toplam_musteri += musteri_sayisi

        vardiya_ortalamalari.append(
            vardiya_bekleme / musteri_sayisi if musteri_sayisi > 0 else 0.0
        )

    return {
        "toplam_bekleme": float(toplam_bekleme),
        "toplam_musteri": int(toplam_musteri),
        "vardiya_ortalama_bekleme": np.asarray(
            vardiya_ortalamalari, dtype=float
        ),
    }


def politika_degerlendir(
    personel: Iterable[int],
    tekrar_seedleri: Iterable[int],
    saatlik_personel_maliyeti: float = 180.0,
    bekleme_dakika_maliyeti: float = 2.0,
    hedef_vardiya_bekleme_dakika: float = 1.5,
    ceza_katsayisi: float = 1500.0,
    vardiya_suresi_dakika: int = 240,
    gelis_hizlari: tuple[float, float, float] = (1.0, 1.6, 1.2),
) -> SimulasyonOzeti:
    """
    Bir personel politikasını ortak rastgele sayılarla birden fazla kez değerlendirir.

    Aynı tekrar_seedleri tüm aday politikalarda kullanılır. Bu teknik,
    simülasyon optimizasyonunda 'common random numbers' yaklaşımının basit bir
    uygulamasıdır ve aday politikalar arasındaki karşılaştırma gürültüsünü
    azaltmaya yardımcı olur.
    """
    personel = np.asarray(personel, dtype=int)
    seedler = [int(s) for s in tekrar_seedleri]

    if len(seedler) == 0:
        raise ValueError("En az bir tekrar tohumu verilmelidir.")

    kosumlar = [
        tek_gun_simule_et(
            personel=personel,
            seed=seed,
            vardiya_suresi_dakika=vardiya_suresi_dakika,
            gelis_hizlari=gelis_hizlari,
        )
        for seed in seedler
    ]

    toplam_beklemeler = np.array(
        [k["toplam_bekleme"] for k in kosumlar], dtype=float
    )
    toplam_musteriler = np.array(
        [k["toplam_musteri"] for k in kosumlar], dtype=float
    )
    vardiya_beklemeleri = np.vstack(
        [k["vardiya_ortalama_bekleme"] for k in kosumlar]
    )

    vardiya_saat = vardiya_suresi_dakika / 60.0
    personel_maliyeti = (
        float(personel.sum()) * vardiya_saat * saatlik_personel_maliyeti
    )

    bekleme_maliyeti = (
        float(toplam_beklemeler.mean()) * bekleme_dakika_maliyeti
    )

    vardiya_bazli_ortalama = vardiya_beklemeleri.mean(axis=0)
    hedef_asimi = max(
        0.0,
        float(vardiya_bazli_ortalama.max()) - hedef_vardiya_bekleme_dakika,
    )
    hizmet_seviyesi_cezasi = ceza_katsayisi * hedef_asimi**2

    toplam_maliyet = (
        personel_maliyeti + bekleme_maliyeti + hizmet_seviyesi_cezasi
    )

    toplam_musteri = float(toplam_musteriler.sum())
    agirlikli_ortalama_bekleme = (
        float(toplam_beklemeler.sum()) / toplam_musteri
        if toplam_musteri > 0
        else 0.0
    )

    return SimulasyonOzeti(
        toplam_maliyet=float(toplam_maliyet),
        personel_maliyeti=float(personel_maliyeti),
        bekleme_maliyeti=float(bekleme_maliyeti),
        hizmet_seviyesi_cezasi=float(hizmet_seviyesi_cezasi),
        ortalama_bekleme_dakika=float(agirlikli_ortalama_bekleme),
        vardiya_bazli_ortalama_bekleme=vardiya_bazli_ortalama,
        ortalama_musteri_sayisi=float(toplam_musteriler.mean()),
    )
