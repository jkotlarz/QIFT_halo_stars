QIFT – 6 PODEJŚĆ Z NOWYMI LATENTAMI CHEMICZNYMI (kroki 03 i 04)
================================================================

Kubity
------
  wersje „7q” (podejścia 1a, 1b, 2): q0 = w_atm_1, q1 = w_atm_2 (atmosfera – bez zmian),
                                     q2..q6 = z_Fe, z_alpha, z_C, z_N, z_act
  wersje „5q” (podejścia 3a, 3b, 4): q0..q4 = z_Fe, z_alpha, z_C, z_N, z_act
Podejścia, referencje, miary, ocena i porównania – jak w wersji z 4/2 kubitami
(README_6_podejsc.txt). Nazwy katalogów wyników mają końcówkę _7q / _5q, więc nie
nadpisują poprzednich wyników.

Wymagania wejściowe
-------------------
  - astro_reports/qiftdata.csv z NOWEGO kroku 02 (02_latent_signatures.py w wersji z 7 latentami);
    skrypt 03 przerwie pracę z komunikatem, jeśli brak kolumn z_C, z_N, z_act.
  - gwiazdy_qift_kinematyka.csv (może być wersja uzupełniona przez 02a).
  - nie jest potrzebna topologia z oryginalnego potoku (dla 5 i 7 kubitów jej nie ma).

Uruchamianie
------------
  python 03_hilbert_topology_<podejście>.py
  python 04_qift_eigenstates_<podejście>.py
Przybliżone czasy (test na zwykłym CPU):
  03: ~7 min (7q; 21 sprzężeń RZZ, 3 starty), ~1,5 min (5q)
  04: kilka minut dla 1a/3a; dla 1b, 2, 3b, 4 dłużej, bo topologia jest uczona od nowa
      w każdym z 15 foldów (czasy z testu – w pliku czasy_testu.txt)
Pamięć: < 1 GB (miary T² i Q liczone przez jądro wierności, bez jawnych macierzy 2·4^n).

Zmiany względem wersji 4q/2q
----------------------------
  - nowe latenty z_C, z_N, z_act czytane z qiftdata.csv (nie muszą być w REF_CSV),
  - T² i Q przez jądro K_ij = |⟨ψ_i|ψ_j⟩|² – wyniki identyczne z jawnymi odchyleniami
    Hilberta–Schmidta (sprawdzone do 1e-14), ale pamięć O(N²) zamiast O(N·4^n),
  - mody odchyleń: K_MODES = 4 (7q) i 3 (5q),
  - nowa metoda klasyczna dla jednej klasy: LOF (wykrywa anomalie lokalne / relacyjne),
  - porównania parowane: QIFT T² także vs Isolation Forest, LOF i −z_Fe,
  - główna decyzja jednoklasowa na wykresach i w analizie splątania: T² > próg (było: a0),
  - splątanie (tylko 7q): A = q0,q1 (atmosfera), B = q2..q6 (pięć latentów chemicznych);
    entropia gwiazdy maks. 2 bity, przetasowanie całego bloku chemii.

Uwaga do interpretacji
----------------------
Przerwa masowa dla 7 kubitów ma wiele lokalnych maksimów (różne starty dają różne kąty
przy podobnej przerwie) – stabilność kątów między foldami jest w raporcie_glownym.txt.
