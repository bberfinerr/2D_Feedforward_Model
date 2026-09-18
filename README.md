# 2D Conditional Diffusion Model

*[English](#english) | [Türkçe](#türkçe)*

---

## English

A PyTorch implementation of a **conditional diffusion model (DDPM)** trained on a synthetic 2D dataset.

> Built during my internship to learn how diffusion models work, with guidance from my mentor and assigned tasks.

### Dataset
5 Gaussian clusters in 2D space, each labeled `flag 0` or `flag 1`.

### Pipeline
```
Forward Diffusion  →  Denoising Network  →  Reverse Diffusion
 (add noise)         (predict noise,        (sample new data
                    conditioned on t + flag)   from pure noise)
```

The model learns to reverse a gradual noising process and, once trained, generates new 2D samples conditioned on a chosen flag.

### Getting Started
**Requirements:** `torch`, `numpy`, `matplotlib`, `tqdm`

1. Open `2D_Feedforward_Model.ipynb` in Google Colab or Jupyter.
2. Run all cells.

---

## Türkçe

Sentetik bir 2D veri seti üzerinde eğitilmiş, PyTorch ile yazılmış bir **koşullu diffusion model (DDPM)** implementasyonu.

> Stajım sırasında diffusion modellerini öğrenmek amacıyla, mentorumun rehberliği ve verilen ödevler kapsamında geliştirilmiştir.

### Veri Seti
2 boyutlu uzayda, `flag 0` veya `flag 1` etiketli 5 Gauss kümesi.

### Akış
```
Forward Diffusion   →   Denoising Network   →   Reverse Diffusion
(gürültü ekleme)      (t + flag koşuluyla          (saf gürültüden
                       gürültü tahmini)             yeni veri üretme)
```

Model, kademeli gürültü ekleme sürecini tersine çevirmeyi öğreniyor ve eğitim sonunda seçilen flag koşuluna göre yeni 2D örnekler üretebiliyor.

### Çalıştırma
**Gereksinimler:** `torch`, `numpy`, `matplotlib`, `tqdm`

1. `2D_Feedforward_Model.ipynb` dosyasını Colab veya Jupyter'de açın.
2. Tüm hücreleri çalıştırın.
