Celem pracy było opracowanie i zaimplementowanie algorytmu, który wspomógłby diagnostykę mózgowego porażenia dziecięcego (MPD) u niemowląt w oparciu o metodykę Prechtl'a. Algorytm wiąże się z tematem estymacji pozy. W projekcie wykorzystano zagadnienie uczenia maszynowego, a dokładniej regresyjne konwolucyjne sieci neuronowe. Skupiono się na znalezienu czterech dystalnych punktów kończyn niemowlęcia. Wykorzystany zbiór danych składał się z filmu ukazującego niemowlę w pozycji leżącej na plecach, nieskrępowanej, a nagranie obejmowało ujęcia z góry. Wynikiem działania algorytmu były wyznaczone na zdjęciach niemowląt punkty kluczowe, a do walidacji skuteczności rezultatów przedstawiono wyniki odległości euklidesowych między punktami wyznaczonymi przez stworzony model, a punktami wzorcowymi modelu OpenPose.

Zdjęcie poniżej przedstawia wywołanie czterech przykładowych klatek, wraz z wynikowymi punktami predycji
zaimplementowanego modelu oraz z punktami zbioru walidacyjnego - OpenPose.

![predykcja](https://user-images.githubusercontent.com/58743872/129804204-b382cd0c-54cc-4612-b56d-ffa256c7ee43.png)
# Human-Pose-Estimation
