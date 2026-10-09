# Hotel Booking Demand - ML Project

## Kom igång

1. Skapa och aktivera virtuell miljö:
   `python -m venv .venv`
   `.venv\Scripts\activate` (Windows) eller `source .venv/bin/activate` (Mac/Linux)

2. Installera beroenden:
   `pip install -r requirements.txt`

3. Verifiera setupen med testerna:
   `pytest -v`


## Uppdelning i träning, validering och test

Uppdelningen görs i `src/data_split.py` och bygger på tid, inte slump, eftersom modellen ska förutsäga avbokningar för framtida bokningar.

**Princip.** Prognosen görs vid bokningstillfället. Vid en prognostidpunkt får modellen bara lära sig av bokningar vars utfall redan var känt. Om en bokning avbokas eller blir en no-show är känt senast vid ankomsten, så träningsdata består av bokningar med ankomst före prognostidpunkten. Utvärderingsdata består av bokningar som görs efter den.

| Delmängd | Innehåll | Antal | Andel avbokade |
|---|---|---|---|
| `train_val` | Ankomst före 2016-07-01 | 40 418 | 30,8 % |
| `val` | Bokningar gjorda 2016-07-01 – 2016-09-30 | 11 643 | 35,3 % |
| `train_full` | Ankomst före 2016-10-01 | 55 397 | 32,1 % |
| `test` | Bokningar gjorda 2016-10-01 – 2016-12-31 | 14 879 | 39,6 % |

Modeller och hyperparametrar väljs genom att träna på `train_val` och utvärdera på `val`. De valda modellerna tränas sedan om på `train_full`, som även innehåller ankomsterna juli–september 2016, och utvärderas en gång på `test`.

**Val av perioder.**
- Träningsdata för validering omfattar ett helt år av ankomster, så att alla månader finns med.
- Bokningar gjorda från 2017 används inte. Datan slutar med ankomster i augusti 2017, så bokningar med lång ledtid saknas för den perioden och skulle ge ett för lätt test.
- Identiska rader har samma bokningsdatum och hamnar därför alltid i samma delmängd.

**Skillnader mellan delmängderna.** Andelen avbokade ökar från 31 % i träningsdata till 40 % i test. Testbokningarna har längre ledtid (median 70 dagar mot 39) och fler ej återbetalningsbara bokningar (14 % mot 12 %). Valideringsperioden har nästan inga ej återbetalningsbara bokningar (2 %), så effekten av `deposit_type` kan inte bedömas på valideringsdata. 