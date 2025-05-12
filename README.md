# Bakalářská práce

Bayesovský odhad časoprostorové emise cesia-137 při požárech v okolí Černobylu -- Michal Uliáš

## Adresáře

Repozitář je rozdělen na jednotlivé adresáře:
- repozitáře `01` až `05` obsahují jednotlivé modely a jejich následné testy na datech `dataCustom.mat`;
- repozitář `06` obsahuje test na syntetických datech;
- repozitář `07` představuje spuštění modelů na reálných datech;
- repozitář `data` obsahuje využitá data:
    - Mapa pro vykreslení z https://www.naturalearthdata.com/downloads/ - public domain;
    - Data z radiační měrné sítě vycházející z článku https://pubs.acs.org/doi/10.1021/acs.est.1c03314 - poskytnuta vedoucím práce: doc. Ing. Ondřej Tichý, Ph.D.;
    - Odhady požárů (Kovalets_prior) z článku https://www.sciencedirect.com/science/article/pii/S1352231022003703 .

## Další soubory

Kód všech odvozených modelů: `ModelsCollection.py`

Data: pro testování funkčnosti: `dataCustom.mat` (generován z adresáře 06) 

Licenční soubor: `license.txt` - zdrojový kód je pod licencí MIT - dostupné také na: https://github.com/UliasMichal/BayesLVLTRegressionForSpatialTemporalEmissionChernobyl2020

## Jak spustit

Kód byl vyvinut na verzi Pythonu: `3.11.5 (main, Sep 11 2023, 13:54:46) [GCC 11.2.0]`.

- Potřebné Python balíčky jsou uvedeny v `requirements.txt`.
- Přepnout se do kořenu implementace (neboli do stejné složky, která obsahuje tento soubor `README.md`).
    - Doporučení: vytvořit si venv (= virtuální prostředí) a následně jej aktivovat:
        - `python -m venv venv`
        - `source venv/bin/activate`
- Instalace balíčků například příkazem: `pip install -r requirements.txt`.
- Poté je možné spustit prostředí Jupyter Lab pomocí `jupyter lab` (popřípadě doinstaloval `jupyter notebook` a využít ten).
- Jednotlivé Jupyter Notebooky (`.ipynb`) lze následně projít a spustit.
