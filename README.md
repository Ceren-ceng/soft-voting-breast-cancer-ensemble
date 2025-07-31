# 🧠 Meme Kanseri Tespiti – Soft Voting ile Ensemble Öğrenme

## 📌 Proje Tanımı

Bu proje, meme kanseri teşhisinde doğruluk oranını artırmak amacıyla **Soft Voting** (Yumuşak Oylama) yönteminin uygulandığı bir topluluk öğrenmesi (Ensemble Learning) sistemidir.  
Birden fazla sınıflandırma algoritmasının birleşimi ile daha istikrarlı, dengeli ve doğru tahminler elde edilmesi hedeflenmiştir.

---

## 🔍 Kullanılan Veri Seti: `BreastCancer.csv`

- 569 örnek kayıt
- 30 sayısal özellik (mean, worst, se değerleri gibi)
- Hedef sütun: `diagnosis`
  - `M` (Malignant – Kötü huylu): 1
  - `B` (Benign – İyi huylu): 0

### 🧼 Veri Ön İşleme Adımları

1. `id` ve `Unnamed: 32` sütunları silindi (anlamsız/boş).
2. `diagnosis` sütunu sayısallaştırıldı: M→1, B→0
3. Eğitim/Test setlerine ayrıldı (%80 eğitim, %20 test).
4. Özellikler `StandardScaler` ile normalize edildi (ölçeklendirme).

---

## 🔬 Kullanılan Ensemble Yöntemi: **Soft Voting Classifier**

### 🤝 Ensemble Nedir?

Birden fazla makine öğrenmesi modelinin çıktısını birleştirerek daha güçlü sonuçlar elde etmeye yönelik tekniktir.

### 🧮 Voting Türleri:

| Tür         | Açıklama                                               |
|-------------|--------------------------------------------------------|
| **Hard Voting** | Her modelin yaptığı sınıf tahmini (etiket) alınır. Çoğunluk oyu kazanır. |
| **Soft Voting** | Her modelin sınıflara dair tahmin ettiği olasılıklar ortalanır. En yüksek ortalama seçilir. |

:

## 🧠 Soft Voting Nedir? Nasıl Çalışır?
Soft Voting, Ensemble Learning (topluluk öğrenmesi) yöntemlerinden biridir ve birden fazla makine öğrenmesi modelinin birlikte çalışarak tahmin yaptığı bir tekniktir.
Ancak Hard Voting’den farklı olarak sadece “hangi sınıfı seçtin?” diye sormaz, “bu sınıfa ne kadar güveniyorsun?” sorusunu da dikkate alır.

🎯 Çalışma Mantığı:
Her model, tahmin ettiği her sınıf için bir olasılık değeri üretir (predict_proba()).

Örnek: Logistic Regression → [Kanserli: %85, Temiz: %15]

Tüm modellerin verdiği olasılıklar toplanır ve ortalaması alınır.

Örnek:
Model 1: [0.85, 0.15]
Model 2: [0.90, 0.10]
Model 3: [0.80, 0.20]
Ortalama: [0.85, 0.15]

En yüksek ortalamaya sahip sınıf final karar olarak seçilir.

## 📌 Neden Soft Voting Kullanılır?
Çünkü sadece çoğunluk oyuna değil, modelin eminlik derecesine de bakar.

Kararsız modellerin zayıf tahminlerini baskılamaz, onları da dikkate alır ama az ağırlıklı olur.

Özellikle modeller farklı yapıda ama olasılık çıktısı üretebiliyorsa, bu çeşitlilik başarıyı ciddi oranda artırır.

## 🔍 Hard Voting ile Kıyas:
Özellik	Hard Voting	Soft Voting
Karar Verme	En çok oyu alan sınıf	En yüksek ortalama olasılığa sahip sınıf
Olasılık Gerekli mi	Hayır (predict)	Evet (predict_proba)
Hassaslık	Düşük (sadece etiket)	Yüksek (eminlik dahil)
Denge ve Kararlılık	Orta	Daha iyi dengeleme sağlar
Kullanım Durumu	Basit modeller, hızlı uygulama	Olasılık çıktısı veren modeller, daha yüksek doğruluk

##📌 Özetle
Soft Voting, "model çoğunlukla ne diyor?" yerine,
"her model ne kadar emin? ve ortalama güven kime daha yüksek?" diye sorar.
Bu yüzden çoğu durumda daha akıllı, dengeli ve gerçek hayata uygun kararlar üretir.


## 📌 Bu projede **Soft Voting** seçilmiştir çünkü:
- Olasılık temelli daha hassas karar verir.
- Modellerin belirsizliğini dikkate alır.
- Genelde daha istikrarlı ve dengeli sonuçlar sunar.

---

## 🧠 Kullanılan 3 Modelin Açıklaması

### 1. Logistic Regression
- Doğrusal sınıflandırıcıdır.
- Olasılık hesaplar.
- Basit, yorumlanabilir, genellikle güçlü bir temel modeldir.

### 2. Decision Tree
- Ağaç yapısı ile karar verir.
- Veri ayrımı kurallar üzerinden yapılır.
- Fazla derin olursa overfitting riski vardır.

### 3. K-Nearest Neighbors (KNN)
- Sınıflandırma, en yakın **k komşunun** sınıfına göre yapılır.
- Normalize veri gerektirir.
- Sezgisel ama yavaş olabilir.

---

## 📈 Model Performans Karşılaştırması

| Model                | Doğruluk | Kesinlik | Duyarlılık | F1 Skoru |
|---------------------|----------|-----------|-------------|-----------|
| Logistic Regression | 0.9737   | 0.9762    | 0.9535      | 0.9647    |
| Decision Tree       | 0.9474   | 0.9302    | 0.9302      | 0.9302    |
| KNN                 | 0.9474   | 0.9302    | 0.9302      | 0.9302    |
| **Soft Voting**     | **0.9737** | **0.9762** | **0.9535**  | **0.9647** |

### 🎯 Yorum:
- **Soft Voting**, en yüksek doğruluk oranını Logistic Regression ile birlikte yakaladı.
- Karışıklık matrisinde TP (gerçek pozitif) oranı yüksek, FP (yanlış pozitif) düşüktür.
- Diğer modellerin hataları telafi edilmiş, sınıflar daha net ayrılmıştır.

---

## 📊 Confusion Matrix Analizi

| Model         | TP | TN | FP | FN |
|---------------|----|----|----|----|
| Logistic Reg. | 41 | 70 | 1  | 2  |
| Decision Tree | 40 | 68 | 3  | 3  |
| KNN           | 40 | 68 | 3  | 3  |
| Soft Voting   | 41 | 70 | 1  | 2  |

- TP ve TN sayıları yüksek → Doğru tahmin oranı güçlü
- FP ve FN sayıları düşük → Yanlış teşhis oranı az

---

## ⚙️ Proje Dosya Yapısı

📦 soft-voting-breast-cancer-ensemble
┣ 📄 BreastCancer.csv → Temizlenmiş veri seti
┣ 📄 softvoiting.ipynb → Model eğitimi ve analiz
┣ 📄 README.md → Bu belge





---

## 🔑 Anahtar Kelimeler

`Soft Voting` · `Ensemble Learning` · `Breast Cancer Detection` · `Machine Learning` ·  
`Logistic Regression` · `Decision Tree` · `KNN` · `Scikit-learn` · `Confusion Matrix` · `Precision Recall F1`

---

## 👩‍💻 Geliştirici

**Ceren Mencütekin**

Her türlü katkı ve öneriye açığız!  
🎁 Pull request göndermekten çekinme.  
🤝 Yeni modeller, ROC eğrileri, grid search gibi katkılarla bu projeyi büyütebilirsin.




