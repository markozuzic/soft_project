# soft_project
Projekat je radjen u Python-u uz koriscenje OpenCV-a i Dliba-a, tako da je to potrebno. Neophodno je i skinuti neki prediktor
(projekat je radjen upotrebom shape_predictor_68_face_landmarks.dat), cija ce se putanja prosledjivati kao argument prilikom pokretanja
programa. Ubacen je i obucavajuci skup, u fajlu test2.csv, koju ce program sam pozvati. Za proveru rada sa videom pokrenuti video.py

Fajlovi:
video.py - Kao argument proslediti file path shape predictor-a
Program prilikom pokretanja otvara prozor na kojem vidimo sadrzaj web camere, pronalazi lica i uokviruje ih ispisujuci starosnu grupu.
Moguce starosne grupe: child, adult, senior. Za tacnije rezultate pribliziti se kameri i umiriti.

dateset_creator.py - Kao argumente proslediti file path foldera sa slikama, naziv grupe(u nasem slucaju jedno od: child, adult, senior) i
fajl path prediktora
Program ucitava slike iz prosledjenog fajla, racuna karakteristicne vrednosti i prosledjuje ih, zajedno sa nazivom grupe, u .csv fajl.
Napravljeni fajl se moze korisiti kao obucavajuci skup.

tester.py - Kao argument proslediti file path foldera sa slikama i file path prediktora
Program je koriscen za testiranje algoritma. Za prosledjene slike ispisuje kojoj starosnoj grupi pripadaju.
