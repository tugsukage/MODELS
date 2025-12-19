# IMDb Sentiment Analysis  
**Comparative Study of Embeddings and Classifiers**

## 1. Overview

Энэхүү repository нь **IMDb кино тоймны өгөгдөл** дээр sentiment analysis (эерэг / сөрөг) хийх зорилгоор  
**өөр өөр embedding аргууд** болон **ангилагч (classifier) загваруудын гүйцэтгэлийг системтэйгээр харьцуулсан судалгааны ажлыг** агуулна.

Судалгаанд:
- Уламжлалт **feature-based machine learning**
- Орчин үеийн **neural embedding болон deep learning**

аргуудыг бодит туршилтаар үнэлж, **ямар нөхцөлд аль арга илүү тохиромжтой вэ** гэдгийг тогтоохыг зорьсон.

---

## 2. Abstract

Энэхүү судалгаагаар IMDb кино тоймны өгөгдөл дээр sentiment analysis хийх зорилгоор  
**TF, IDF, TF-IDF, Word2Vec (CBOW, Skip-gram), BERT** embedding-үүдийг  
**Logistic Regression, Random Forest, AdaBoost, LSTM** ангилагч загваруудтай хослуулан ашиглав.

Загваруудыг **Stratified 5-Fold Cross Validation** ашиглан сургаж,  
**Accuracy, Precision, Recall, Macro-F1** үзүүлэлтүүдээр үнэлсэн.

Үр дүнгээс харахад:
- **TF-IDF + Logistic Regression** нь хамгийн тогтвортой, өндөр гүйцэтгэлтэй
- **LSTM + BERT chunked embeddings** нь deep learning аргуудаас хамгийн сайн үр дүн үзүүлсэн
- BERT embedding нь контекстийг сайн тусгасан ч classical ML-тэй хослоход үр ашиг буурсан

---

## 3. Introduction

Sentiment analysis нь хэрэглэгчийн текстэн өгөгдлөөс сэтгэл хандлагыг автоматаар тодорхойлох  
**Natural Language Processing (NLP)**-ийн суурь асуудлын нэг юм.

IMDb кино тоймны dataset нь:
- Balanced
- Том хэмжээтэй
- Бодит хэрэглэгчийн текст агуулсан

гэдгээрээ sentiment analysis судалгаанд **стандарт benchmark** болж ашиглагддаг.

Энэхүү ажлын зорилго нь:
> **Embedding + classifier хослолуудын онолын ялгаа бодит гүйцэтгэлд хэрхэн нөлөөлдгийг туршилтаар нотлох** явдал юм.

---

## 4. Dataset Description

**Stanford Large Movie Review Dataset (IMDb)**

- Total samples: **50,000**
- Train set: **25,000**
- Test set: **25,000**
- Classes: Positive / Negative (50% – 50%)
- Language: English

Dataset link:  
https://ai.stanford.edu/~amaas/data/sentiment/

---

## 5. Methodology

### 5.1 Preprocessing

Дараах preprocessing алхмуудыг хэрэгжүүлсэн:

- Text normalization (lowercasing)
- Tokenization
- Stopword removal (сонгомол)
- TF / IDF / TF-IDF vectorization
- Word2Vec sequence үүсгэх
- BERT embedding-д padding ба truncation
- Sentence representation-ийг:
  - Classical ML-д: vector
  - LSTM-д: sequence хэлбэрээр ашигласан

---

## 6. Embedding Methods

### TF / IDF / TF-IDF
Статистикт суурилсан, sparse representation үүсгэдэг бөгөөд  
**Logistic Regression** зэрэг шугаман загваруудад маш тохиромжтой.

### Word2Vec (CBOW, Skip-gram)
Үг хоорондын семантик хамаарлыг нейрон сүлжээгээр сурч,  
dense embedding үүсгэдэг.

### BERT Embeddings
Transformer архитектурт суурилсан,  
**context-aware** representation үүсгэдэг pretrained загвар.

---

## 7. Classifiers (Theoretical Background)

### Logistic Regression
- Шугаман, магадлалд суурилсан ангилагч
- Sparse, өндөр хэмжээст текст өгөгдөлд онцгой тохиромжтой
- Туршилтаар **TF-IDF-тэй хамгийн сайн гүйцэтгэл** үзүүлсэн

### Random Forest
- Decision Tree ансамбль
- Non-linear хамаарлыг барих чадвартай
- IDF embedding-тэй үед харьцангуй сайн ажилласан

### AdaBoost
- Boosting-д суурилсан загвар
- Noise-д мэдрэмтгий
- BERT embedding-тэй үед classical аргуудаас арай илүү үр дүнтэй

### LSTM
- Sequence өгөгдлийн дарааллыг хадгална
- Word order, context-ийг ашиглана
- **LSTM + BERT chunked embeddings** → хамгийн сайн deep learning үр дүн

---

## 8. Experimental Setup

- Validation: **Stratified 5-Fold**
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - Macro-F1
- Platform: Google Colab
- Hardware: Intel Xeon CPU, GPU (сонгомол)

---

## 9. Results

| Embedding / Feature | LogReg Acc | LogReg F1 | Random Forest | AdaBoost | LSTM (Test F1) |
|--------------------|------------|-----------|---------------|----------|---------------|
| TF-IDF (uni+bi) | **0.8908** | **0.8915** | 0.8215 | 0.7046 | 0.0410 |
| IDF-only | 0.8734 | 0.8725 | **0.8328** | 0.7112 | – |
| Word2Vec CBOW | 0.8476 | 0.8474 | 0.7590 | 0.7368 | – |
| BERT embeddings | 0.8192 | 0.8184 | 0.7661 | **0.7569** | **0.8368** |

---

## 10. Discussion

- Classical ML + TF-IDF нь **хурд, тогтвортой байдал, гүйцэтгэлийн хувьд хамгийн практик**
- BERT embedding нь context сайн тусгасан ч classical ML-д хэт өндөр dimensional болж үр ашиг буурсан
- Deep learning нь sequence ойлгох давуу талтай боловч нөөц их шаарддаг

---

## 11. Conclusion

Embedding болон classifier-ийн сонголт нь sentiment analysis-ийн гүйцэтгэлд **шийдвэрлэх нөлөөтэй**.  
Нэг загвар бүх нөхцөлд давуу байх боломжгүй тул асуудлын шинж чанарт тулгуурлан сонгох шаардлагатай.

---

## 12. Future Work

- Full BERT fine-tuning
- Model size vs performance trade-off
- Multilingual sentiment analysis

---

## 13. References

(Дээрх лавлагаанууд хэвээр ашиглагдана)
