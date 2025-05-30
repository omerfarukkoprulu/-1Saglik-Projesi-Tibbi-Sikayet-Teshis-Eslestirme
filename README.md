# -1Saglik-Projesi-Tibbi-Sikayet-Teshis-Eslestirme

# Yapay Zeka Dersi - Ödev 2: Eğitilen Modellerle Metin Benzerliği Hesaplama ve Değerlendirme

Bu depo, Yapay Zeka dersi kapsamında gerçekleştirilen ikinci ödevin kodlarını ve raporunu içermektedir. Ödevin temel amacı, doğal dil işleme (NLP) tekniklerini kullanarak metinler arası benzerlik hesaplamaları yapmak ve farklı modellerin (TF-IDF ve Word2Vec) başarısını karşılaştırmalı olarak değerlendirmektir.

## Proje Amacı

Bu projenin ana hedefleri şunlardır:

* Sağlık alanındaki tıbbi terimlerden oluşan bir metin korpusunu ön işleme tabi tutmak (tokenizasyon, stop-word temizliği, lemmatizasyon, stemming).
* Ön işlenmiş veri üzerinde TF-IDF (Term Frequency-Inverse Document Frequency) ve Word2Vec modellerini eğitmek.
* Eğitilen modelleri kullanarak metinler arası anlamsal ve bağlamsal benzerlikleri hesaplamak.
* Farklı Word2Vec model yapılandırmalarının (CBOW/Skip-gram, window boyutu, vektör boyutu) benzerlik sonuçları üzerindeki etkilerini değerlendirmek.
* TF-IDF ve Word2Vec modellerinin karşılaştırmalı başarısını analiz etmek.

## İçerik

Bu depo aşağıdaki dosyaları ve dizinleri içermektedir:

* `dogal_dil_isleme.ipynb`: Metin ön işleme adımlarını (tokenizasyon, stop-word temizliği, lemmatizasyon, stemming) gerçekleştiren Jupyter Notebook dosyası. Bu adımda `cleaned_corpus.csv`, `lemmatized_corpus.csv` ve `stemmed_corpus.csv` dosyaları oluşturulur.
* `tfidf_lemmatized.ipynb`: Lemmatize edilmiş veri üzerinde TF-IDF modelini eğiten ve metin benzerliği hesaplayan Jupyter Notebook dosyası.
* `tfidf_stemmed.ipynb`: Stemmed edilmiş veri üzerinde TF-IDF modelini eğiten ve metin benzerliği hesaplayan Jupyter Notebook dosyası.
* `word2vec1.ipynb`: Hem lemmatize edilmiş hem de stemmed edilmiş veri üzerinde çeşitli Word2Vec modellerini (CBOW/Skip-gram, farklı window ve vektör boyutları) eğiten ve benzer kelimeleri bulan Jupyter Notebook dosyası.
* `metin_benzerlik.ipynb`: Eğitilmiş Word2Vec modelleri arasında Jaccard benzerlik skorlarını hesaplayarak model tutarlılığını değerlendiren Jupyter Notebook dosyası.
* `zipf.ipynb`: Orijinal (temizlenmiş) metin üzerinde Zipf yasası analizini gerçekleştiren Jupyter Notebook dosyası.
* `Zipf_Analizi (lemmatized).ipynb`: Lemmatize edilmiş metin üzerinde Zipf yasası analizini gerçekleştiren Jupyter Notebook dosyası.
* `Zipf_Analizi (stemmed).ipynb`: Stemmed edilmiş metin üzerinde Zipf yasası analizini gerçekleştiren Jupyter Notebook dosyası.
* `yz_final_2.odev (1).pdf`: Ödevin orijinal PDF yönergeleri.
* `Rapor.pdf` (veya `Rapor.md`): Ödevin gerektirdiği tüm analizleri, sonuçları ve değerlendirmeleri içeren detaylı rapor dosyası. (Bu dosya raporun kendisini içerecektir.)

## Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

### Ön Gereksinimler

* Python 3.x
* `pip` paket yöneticisi

### Bağımlılıkların Yüklenmesi

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
Not: requirements.txt dosyanız yoksa, Jupyter Notebook dosyalarında kullanılan başlıca kütüphaneleri (nltk, pandas, numpy, scikit-learn, gensim, matplotlib) tek tek yüklemeniz gerekebilir.

Bash

pip install nltk pandas numpy scikit-learn gensim matplotlib
NLTK Verilerinin İndirilmesi
Bazı NLTK kaynakları (stopwords, punkt, wordnet) otomatik olarak indirilirken, bazen manuel indirme gerekebilir. Jupyter Notebook'ları ilk çalıştırdığınızda otomatik indirme gerçekleşecektir.

Python

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Adımlar
Veri Ön İşleme:

dogal_dil_isleme.ipynb dosyasını çalıştırarak veri setini temizleyin, tokenizasyon, lemmatizasyon ve stemming işlemlerini yapın. Bu adım sonucunda gerekli csv dosyaları (cleaned_corpus.csv, lemmatized_corpus.csv, stemmed_corpus.csv) oluşacaktır.
Zipf Yasası Analizi (İsteğe Bağlı):

zipf.ipynb, Zipf_Analizi (lemmatized).ipynb ve Zipf_Analizi (stemmed).ipynb dosyalarını çalıştırarak veri setinin kelime dağılımını inceleyin.
TF-IDF Modellerini Eğitme ve Değerlendirme:

tfidf_lemmatized.ipynb ve tfidf_stemmed.ipynb dosyalarını çalıştırarak TF-IDF modellerini eğitin ve belirli bir kelimenin benzerlerini bulun.
Word2Vec Modellerini Eğitme ve Değerlendirme:

word2vec1.ipynb dosyasını çalıştırarak farklı Word2Vec modellerini eğitin ve belirli bir kelimenin en benzer kelimelerini listeleyin. Bu adım, eğitilmiş Word2Vec modellerini .model uzantılı dosyalar olarak kaydedecektir.
Model Karşılaştırma:

metin_benzerlik.ipynb dosyasını çalıştırarak Word2Vec modelleri arasındaki Jaccard benzerliğini hesaplayın ve sonuçları inceleyin.
Rapor
Projenin tüm detayları, metodolojisi, sonuçları ve değerlendirmeleri Rapor.pdf (veya Rapor.md) dosyasında bulunabilir. Bu rapor, ödev yönergelerinde belirtilen tüm bölümleri içermektedir.

Sonuçlar (Özet)
TF-IDF: Kelime sıklığı ve doküman nadirliğine dayalı olarak metin benzerliğini ölçer. Direkt kelime ilişkilerini yakalar.
Word2Vec: Kelimelerin bağlamsal ilişkilerini öğrenerek anlamsal olarak daha zengin benzerlikler sunar. Özellikle tıbbi terimler gibi alanlarda daha anlamsal sonuçlar vermiştir.
Model Tutarlılığı: Farklı Word2Vec model yapılandırmaları (CBOW/Skip-gram, window/vektör boyutları) arasında yüksek Jaccard benzerlik skorları gözlemlenmiştir, bu da modellerin benzer anlamsal uzaylar öğrendiğini göstermektedir.
Ön İşlemenin Etkisi: Lemmatizasyon ve stemming, kelimelerin kök hallerine indirgenerek veri setini düzenlemiş ve anlamsal benzerlik sonuçları üzerinde tutarlı bir etki yaratmıştır.

