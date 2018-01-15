# speech_emotion_recognition

Wymagania:
* python 3.5
* pip3

Polecane:
* virtualenv

W celu pobrania wszystkich zależności należy włączyć i wpisać komendę (najlepiej w środowisku wirtualnym)
* pip3 install -r requirements.txt

Aby rozpocząć testowanie używając algorytmu KNN należy wpisać jedną z poniższych komand:
```
$ python main.py KNN
$ python main.py KNN "name of database" "password to database" (Aby zapisać cześć informacji w bazie danych i przyspieszyć działanie programu)

Aby rozpocząć testowanie używając algorytmu HMM należy wpisać jedną z poniższych komand:
```
$ python main.py HMM
$ python main.py HMM "name of database" "password to database" (Aby zapisać cześć informacji w bazie danych i przyspieszyć działanie programu)

```

Example output:
```
tested emotion: anger	 , anger: 0	 , boredom: 0	,  happiness: 4	, sadness: 0	, guessed:36	, tested: 40	, trained: 37

tested emotion: boredom	 , anger: 2	 , boredom: 0	,  happiness: 5	, sadness: 8	, guessed:11	, tested: 26	, trained: 37

tested emotion: happiness, anger: 19 , boredom: 0	,  happiness: 0	, sadness: 0	, guessed:4	    , tested: 23	, trained: 38

tested emotion: sadness	 , anger: 0	 , boredom: 1	,  happiness: 0	, sadness: 0	, guessed:23	, tested: 24	, trained: 27
```

* tested emotion - testowana emocja
* trained - liczba plików które zostały użyte do wytrenowania tested emotion
* tested - liczba testów tej emocji
* guessed - ile testów przeszło poprawnie (algorytm poprawnie obliczył jaką emocję ten plik reprezentuje)
* kolumny z emocjami - jakie emocje wyliczył algorytm dla każdego z testowanych plików (na przykład:
w wierszu 1, "happiness: 4" oznacza, że algorytm dla 4 plików reprezentujących anger obliczył happiness.

