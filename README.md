# HieroglyphsApp

Webapp bygget ud fra Hieroglyf-app-prototypen.

## Opdatering til en ny udgave af prototypen

1. Gem prototypens HTML (artefaktet) som en fil.
2. Skriv hvad der er nyt øverst i `changelog.json` (nyeste først, `version` er datoen). Første gang appen åbnes efter opdateringen, vises de nye punkter i et vindue. Uden et nyt punkt opdateres appen uden vindue.
3. Kør `python3 tools/build.py sti/til/prototype.html`. Det bygger `index.html` (lokale skrifttyper, fuldskærm, automatisk opdatering) og skifter cachenavnet i `sw.js`.
4. Commit og merge til `main`. GitHub Pages udgiver siden, og appen på telefonen henter selv den nye udgave, næste gang den åbnes.
