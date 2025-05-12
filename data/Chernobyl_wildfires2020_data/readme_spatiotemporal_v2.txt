V souboru jsou SRS matice pro 3 ruzne velikosti castic (mean diameter 0.4 8 a 16 um) a 7 ruznych vysek uniku pro okoli Cernobylu 11 x 11 (diskretizace 0.5, uprostred je NPP Cernobyl).

y_orig, y_unc: mereni a neurcitost mereni (kde je k dispozici), je potreba brat kazdou 5. hodnotu, tzn. 1:5:4290

koefBq: koeficient pro prevod na Bq vynasobenim vysledneho odhadu

Ms: tenzor se SRS maticemi, 6 rozmeru
lat lon frakce hladina mereni cas
- frakce 1 az 3 pro mean diameter 0.4 8 a 16 um
- hladina 1 az 7 pro 0-100-500-1000-1500-2000-2500-3000 m
- 858 mereni (doporucuju vyhodit ty, kde ind_cez==1, tzn. z exclusion zone)
- 30 dni (duben)

latitudes a longitudes: stredy pixelu domeny, [latitudes(1) longitudes(1)] odpovida Ms(11,1,...) a je to levy dolni pixel domeny. Je to stejne jako u v1, Matlabovsky imagesc to vytiskne spravne zobrazene SRS citlivosti sum(Ms,[3 4 5 6]), tzn
1,1 1,2 1,3 ... 1,11
2,1 2,2 2,3 ... 2,11
....................
11,1 11,2 .... 11,11

