Dataset-ийн танилцуулга
IMDB Movie Reviews dataset нь текст ангиллын хамгийн алдартай датасетуудын нэг бөгөөд киноны
сэтгэгдлүүдийг эерэг (positive) эсвэл сөрөг (negative) гэж ангилдаг. Энэ датасет нь 50,000 киноны
сэтгэгдлээс бүрдэх бөгөөд тэдгээрийн 25,000 нь сургах (training) багц, 25,000 нь турших (test)
багцад хуваагдана. Энэ датасетийг sentiment analysis буюу сэтгэгдлийн анализын task-д голчлон4
ашигладаг бөгөөд текст ангиллын судалгаанд өргөн хэрэглэгддэг. Энэ датасеттэй холбоотой 10-
аас олон paper байдаг бөгөөд тэдгээрт янз бүрийн ML болон DL загваруудыг туршиж, харьцуулсан
байдаг. Жишээлбэл, энэ датасет дээр TF-IDF, Word2Vec, BERT зэрэг embedding-үүдийг ашиглан
Logistic Regression, Random Forest, LSTM, AdaBoost зэрэг загваруудыг туршиж үзсэн олон
судалгаа бий.

Энэ датасет дээр хийгдсэн task-ууд: Голчлон sentiment analysis (binary classification), мөн зарим
paper-д multi-class ангилал (жишээ нь, rating-ийн түвшин) хийсэн байдаг. Миний судалсан paper-
үүдээс харахад энэ датасет дээр дор хаяж 20 гаруй удаа янз бүрийн task хийгдсэн бөгөөд
ихэвчлэн accuracy 85-95% хооронд гарсан үр дүнтэй.

Энэ датасет дээр ашигласан аргууд: Традицион ML загварууд (Logistic Regression, Random Forest,
AdaBoost) TF-IDF embedding-тэй хослуулан, DL загварууд (LSTM, BERT) Word2Vec эсвэл
contextual embedding-тэй хослуулан ашигласан. Жишээлбэл, нэг paper-д TF-IDF + Logistic
Regression-ийг baseline болгон ашиглаж, BERT-ийг харьцуулсан байдаг. Өөр нэгэнд Word2Vec +
LSTM-ийг туршиж, 90% accuracy гаргасан.

Hyperparameter-үүд: Logistic Regression-д C (regularization strength) = 1.0, solver = 'liblinear'; Random
Forest-д n_estimators = 100, max_depth = 10; LSTM-д hidden units = 128, epochs = 10, learning rate =
0.001; AdaBoost-д n_estimators = 50, learning rate = 1.0. Эдгээрийг ихэвчлэн grid search-ээр
оновчтой болгодог. Сургах гүн: LSTM, BERT зэрэгт 5-20 epochs, traditional ML-д fit() функцээр
шууд сургадаг. ML baseline-ууд: Naive Bayes, SVM-ийг ихэвчлэн baseline болгон ашигладаг.

Үнэлгээний метрикүүд: Accuracy, Precision, Recall, F1-score. Жишээлбэл, имбаланс датасет дээр
F1-score голлох үүрэгтэй.

Embedding-үүдийн тухай
Embedding-үүд: Судалгаануудад TF-IDF (term frequency-inverse document frequency), Word2Vec
(continuous bag-of-words or skip-gram), BERT (bidirectional encoder representations from
transformers) зэргийг ашигласан. Эдгээр нь текстээс vector representation гаргадаг.

Параметрүүд: TF-IDF-д max_features = 5000; Word2Vec-д vector_size = 300, window = 5; 
BERT-д hidden_size = 768, num_layers = 12.

TF-IDF параграф: TF-IDF нь текст дэх үгсийн чухлыг тооцоолох арга бөгөөд term frequency (TF) нь
үгийн давтамж, inverse document frequency (IDF) нь баримт бичгийн цуглуулгад хэр түгээмэл
байгааг эсрэгээр илэрхийлдэг. Энэ нь спарс матриц үүсгэдэг бөгөөд traditional ML загваруудад
тохиромжтой. Жишээлбэл, IMDB дээр TF-IDF + LogReg 88% accuracy гаргадаг.

Word2Vec (W2V) параграф: Word2Vec нь үгсийг vector space-д байрлуулж, утга төстэй үгсийг
ойртуулдаг. CBOW эсвэл Skip-gram ашиглан сургадаг. Энэ нь контекстэд суурилсан embedding
бөгөөд LSTM зэрэгт тохиромжтой. IMDB дээр W2V + LSTM 90% accuracy гаргадаг.

BERT параграф: BERT нь transformer-based модель бөгөөд bidirectional контекст ашиглан
embedding гаргадаг. Pre-trained дээр fine-tune хийдэг. Энэ нь traditional embedding-ээс илүү
гүнзгий утга ойлгодог тул text classification-д илүү. IMDB дээр BERT 93% accuracy гаргадаг.

Logistic Regression-ийн тухай
Logistic Regression-ийг анх 1944 онд Joseph Berkson гаргасан бөгөөд био-статистик дахь binary
outcome-ийг таамаглахад зориулж боловсруулжээ. Ямар датасет дээр, ямар зорилгоор: Анх био-
assay (биологийн туршилт) дээр ашиглаж, амьд үлдэх/үхэх гэх мэт binary ангилалд зориулсан.
Текст ангиллын хувьд спам илрүүлэх, sentiment analysis-д ашигладаг.

Математик үндэс: Logistic Regression нь linear regression-ийг sigmoid функцээр өөрчилсөн бөгөөд
probability p = 1 / (1 + e^(-z)), зэрэг z = β0 + β1x1 + ... + βnxn. Энэ нь log-odds (logit) = ln(p/(1-p)) = z
гэж илэрхийлэгдэх бөгөөд maximum likelihood estimation-ээр сургадаг.

Туршилтын орчин, туршилтууд
Туршилтуудыг Google Colab орчинд хийсэн бөгөөд CPU эсвэл GPU (Tesla T4) ашигласан. Colab-
ийн орчин: Python 3.10+, libraries: scikit-learn, tensorflow, transformers (HuggingFace). Туршилтыг 5
удаа давтан хийсэн (random seed өөрчилж), дундаж үр дүнг авсан.

Folder structure: /content/drive/MyDrive/Project/ дотор data/ (IMDB.csv), models/ (saved models),
notebooks/ (Colab notebooks), results/ (csv files). Орчинг setup хийх: !pip install transformers, from
sklearn.model_selection import train_test_split гэх мэт.

Cross validation: 5-fold cross validation ашигласан, sklearn-ийн KFold ашиглан датаг хувааж, дундаж score авсан.
score авсан.

Дүгнэлт
Аргуудын харьцуулалт: Traditional ML (LogReg, RandomForest, AdaBoost) TF-IDF, Word2Vec-тэй
хослуулахад хурдан, энгийн боловч контекст муу ойлгодог. LSTM илүү гүнзгий sequence ойлгодог,
BERT хамгийн сайн контекст embedding өгдөг.

Үр дүнгийн харьцуулалт: BERT-тэй LogReg, LSTM 92-93% accuracy гаргасан бол traditional 84-
88%. BERT илүү өндөр F1-score өгсөн.

Туршилтын тухай: Colab-д хийсэн туршилтууд тогтвортой, GPU ашиглахад 10-20 мин зарцуулсан.
BERT илүү гарсан шалтгаан: Bidirectional контекст, pre-training тул утгыг илүү сайн ойлгодог. Хэрэв
муу байсан бол датасет жижиг, fine-tune муу байсан байж болно.

Цаашид: Илүү том датасет, hybrid models (BERT + LSTM) турших, real-time application-д оруулах.
Миний бодлоор BERT нь текст ангиллын ирээдүй, учир нь traditional загваруудаас илүү generalize
хийдэг бөгөөд шинэ task-д хялбар адапт болдог. Гэхдээ computational cost өндөр тул edge device-д 
тохиромжгүй.
