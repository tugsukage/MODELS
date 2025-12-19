#  IMDb Sentiment Analysis

Энэхүү repository нь **IMDb кино тоймны өгөгдөл** дээр sentiment analysis (эерэг / сөрөг) хийх зорилгоор **өөр өөр embedding аргууд** болон **ангилагч загваруудын гүйцэтгэлийг харьцуулан судалсан туршилтын ажлыг** агуулна. Уламжлалт machine learning аргууд болон орчин үеийн transformer-д суурилсан embedding-үүдийг системтэйгээр харьцуулж, тэдгээрийн давуу ба сул талыг туршилтаар үнэлсэн болно.

---

##  Abstract

Энэхүү судалгаагаар IMDb кино тоймны өгөгдөл дээр sentiment analysis хийх зорилгоор **TF, IDF, TF-IDF, Word2Vec (CBOW, Skip-gram), BERT** embedding-үүдийг **Logistic Regression, Random Forest, AdaBoost, LSTM** зэрэг ангилагч загваруудтай хослуулан ашиглав. Загваруудыг **Stratified K-Fold Cross Validation** аргаар үнэлж, **Accuracy, Precision, Recall, Macro-F1** үзүүлэлтүүдийг ашиглан харьцуулсан.

Үр дүнгээс харахад TF-IDF embedding нь classical ML загваруудтай хослоход тогтвортой өндөр гүйцэтгэл үзүүлсэн бол, BERT embedding нь контекстийн мэдээллийг илүү сайн тусгасан ч тооцооллын зардал өндөр, зарим тохиолдолд overfitting илэрсэн.

---

##  Introduction

Sentiment analysis нь текстэн өгөгдлөөс хэрэглэгчийн санал бодол, хандлагыг автоматаар тодорхойлох **Natural Language Processing (NLP)**-ийн чухал салбар юм. Онлайн орчинд хуримтлагдаж буй кино, бүтээгдэхүүн, үйлчилгээний тоймуудыг автоматаар ангилах хэрэгцээ улам бүр нэмэгдэж байна.

IMDb кино тоймны dataset нь sentiment analysis загваруудыг туршихад өргөн хэрэглэгддэг **стандарт benchmark өгөгдөл** бөгөөд энэхүү ажлын зорилго нь уг dataset дээр **өөр өөр embedding + classifier хослолуудын гүйцэтгэлийг бодит туршилтаар харьцуулах** явдал юм.

---

##  Dataset Description

**Stanford Large Movie Review Dataset (IMDb)**

* Нийт тойм: **50,000**
* Train set: **25,000**
* Test set: **25,000**
* Ангилал: Positive / Negative (50% – 50%)
* Хэл: Англи
* Онцлог: Бодит хэрэглэгчдийн бичсэн, урт нь харилцан адилгүй тоймууд

 Dataset татах линк:
[https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)

---

##  Dataset дээр хийгдсэн task-ууд

Энэхүү ажлын хүрээнд dataset дээр дараах **гол task-уудыг 3 удаагийн томоохон туршилтаар** хийсэн:

1. **Text preprocessing ба feature engineering**
2. **Embedding аргуудын харьцуулалт**
3. **Classical ML vs Embedding-based model** харьцуулалт
4. **Cross-validation бүхий гүйцэтгэлийн үнэлгээ**

---

##  Preprocessing

Өгөгдөлд дараах preprocessing алхмуудыг хэрэгжүүлсэн:

* Text lowercase болгох
* Tokenization
* Stopword removal (зарим туршилтад)
* TF / IDF / TF-IDF vectorization
* Word2Vec-д sequence үүсгэх
* BERT embedding-д padding & truncation ашиглах
* Word2Vec embedding-д өгүүлбэрийн representation-ийг **үгүүдийн embedding-ийн дундаж** утгаар тооцох

---

##  Embedding аргуудын тайлбар

### TF (Term Frequency)

Үгийн тухайн баримт бичиг дэх давтамжид суурилсан энгийн статистик арга.

### IDF (Inverse Document Frequency)

Үг тухайн corpus-д хэр ховор байгааг илэрхийлнэ:

IDF(t) = log(N / df(t))

### TF-IDF

TF ба IDF-ийн үржвэр бөгөөд үгийн ач холбогдлыг илүү сайн тусгадаг.

### Word2Vec (CBOW, Skip-gram)

Нейрон сүлжээгээр үг хоорондын семантик хамаарлыг сурдаг embedding арга.

* CBOW: Context → Word
* Skip-gram: Word → Context

### BERT Embedding

Transformer архитектурт суурилсан, **контекстэд мэдрэмтгий** representation үүсгэдэг pretrained загварууд.

---

##  Ашигласан ангилагч загварууд

### Logistic Regression

Анх **David Cox (1958)** танилцуулсан, магадлалд суурилсан шугаман ангилагч.

* Зорилго: Binary classification
* Loss: Log-loss (Cross-entropy)
* Давуу тал: Хурдан, тайлбарлахад хялбар

### Random Forest

Decision Tree-үүдийн ансамбль загвар.

### AdaBoost

Сул загваруудыг дараалан сайжруулж сургадаг boosting арга.

### LSTM

Sequence өгөгдөл дээр урт хугацааны хамаарлыг сурах чадвартай RNN.

---

##  Туршилтын тохиргоо

* Cross-validation: **Stratified 5-Fold**
* Үнэлгээний үзүүлэлтүүд:

  * Accuracy
  * Precision
  * Recall
  * Macro-F1 score

### Hyperparameter тохируулга:

* Logistic Regression: C
* Random Forest: n_estimators, max_depth
* AdaBoost: n_estimators
* LSTM: embedding size, hidden units, epochs

---

##  Туршилтын орчин

* Platform: **Google Colab**
* CPU: Intel Xeon
* GPU: (зарим BERT туршилтад)
* RAM: ~12GB
* OS: Linux
* Туршилтын давтамж: **3 үндсэн туршилт + cross-validation**

---

##  Үр дүн ба харьцуулалт


| Embedding / Feature                    | LogReg Acc | LogReg F1 | Random Forest | AdaBoost | LSTM (Test F1) |
|---------------------------------------|------------|-----------|---------------|----------|---------------|
| TF-IDF (unigram + bigram)             | **0.8908** | **0.8915** | 0.8215 | 0.7046 | 0.0410 |
| TF (term frequency)                   | 0.7246 | 0.7294 | 0.8204 | 0.7144 | – |
| IDF-only (binary × IDF)               | 0.8734 | 0.8725 | **0.8328** | 0.7112 | – |
| Word2Vec CBOW (IMDB)                  | 0.8476 | 0.8474 | 0.7590 | 0.7368 | – |
| Word2Vec Skip-gram (IMDB)             | 0.8598 | 0.8589 | – | – | – |
| Word2Vec (GoogleNews, pretrained)     | 0.8483 | 0.8467 | – | – | 0.7252 |
| BERT embeddings (MiniLM-L6-v2)         | 0.8192 | 0.8184 | 0.7661 | **0.7569** | **0.8368** |
| Word2Vec (trainable, LSTM)            | – | – | – | – | 0.8093 |




##  Дүгнэлт

Embedding арга болон classifier-ийн сонголт нь sentiment analysis-ийн гүйцэтгэлд шууд нөлөөлдөг. Практик хэрэглээнд **TF-IDF + Logistic Regression** нь хурд, гүйцэтгэлийн тэнцвэрийг сайн хангаж байсан бол, BERT нь илүү нарийвчлалтай боловч нөөц их шаардсан.

---

##  Ирээдүйн ажил

* BERT-ийг full fine-tuning хийх
* Inference хурд ба model size-ийн харьцуулалт
* Class imbalance бүхий өгөгдөл дээр турших

---

##  Лавлагаа

IMDb sentiment analysis-тэй холбоотой **10+ судалгааны ажил** ашигласан (README төгсгөлд бүрэн жагсаалт хавсаргасан).

[1] Pouransari, H., & Ghili, S. (2014).  
Deep Learning for Sentiment Analysis of Movie Reviews.  
Stanford University (CS224d).  
PDF: https://cs224d.stanford.edu/reports/PouransariHadi.pdf

[2] Meng, X., & Wang, Y. (2023).  
Sentiment Analysis with Adaptive Multi-Head Attention in Transformer.  
arXiv preprint arXiv:2310.14505.  
PDF: https://arxiv.org/abs/2310.14505

[3] Nkhata, T., et al. (2025).  
Fine-tuning BERT with BiLSTM for Movie Review Sentiment.  
arXiv preprint arXiv:2502.20682.  
PDF: https://arxiv.org/abs/2502.20682

[4] Timmaraju, A. (2015).  
Recursive and Recurrent Neural Networks for Sentiment Analysis.  
Stanford University (CS224d).  
PDF: https://cs224d.stanford.edu/reports/TimmarajuAditya.pdf

[5] International Journal of Advanced Computer Science and Applications (IJACSA). (2022).  
Sentiment Analysis of Online Movie Reviews Using Machine Learning Models.  
PDF: https://thesai.org/Downloads/Volume13No9/Paper_73-Sentiment_Analysis_of_Online_Movie_Reviews.pdf

[6] ResearchGate. (2023).  
Sentiment Analysis on IMDB Review Dataset.  
PDF: https://www.researchgate.net/publication/377024012

[7] Derbentsev, V., et al. (2023).  
Comparative Study of Deep Learning Models for Sentiment Analysis.  
CEUR Workshop Proceedings, Vol. 3465.  
PDF: https://ceur-ws.org/Vol-3465/paper18.pdf

[8] Lu, Y., et al. (2025).  
Sentiment Analysis of IMDB Movie Reviews Based on LSTM.  
Journal of Applied Engineering and Technology.  
PDF: https://ojs.apspublisher.com/index.php/jaet/article/view/429

[9] Madasu, A., & Sivasankar, E. (2019).  
Survey of Feature Extraction Techniques for Sentiment Analysis.  
arXiv preprint arXiv:1906.01573.  
PDF: https://arxiv.org/abs/1906.01573

[10] Alaparthi, N., & Mishra, M. (2020).  
BERT-based Sentiment Analysis on IMDB Dataset.  
arXiv preprint arXiv:2007.01127.  
PDF: https://arxiv.org/abs/2007.01127
cs224d.stanford.edu
