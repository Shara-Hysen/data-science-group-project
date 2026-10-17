# Hotel Booking Demand - ML Project

## Kom igång

1. Skapa och aktivera virtuell miljö:
   `python -m venv .venv`
   `.venv\Scripts\activate` (Windows) eller `source .venv/bin/activate` (Mac/Linux)

2. Installera beroenden:
   `pip install -r requirements.txt`

3. Verifiera setupen med testerna:
   `pytest -v`

4. Hämta och städa data:
    `python -m src.data_cleaning`


## Avvikelser från projektplanen

**Tidsuppdelning**
- Uppdelningen görs efter bokningsdatum i stället för ankomstdatum, efter feedback. Träningsdata vid en prognostidpunkt består av bokningar med ankomst före den, eftersom utfallet är känt senast vid ankomsten. Utvärderingsdata består av bokningar gjorda efter prognostidpunkten.
- Perioderna har därför ändrats. Planen hade test på ankomster januari–augusti 2017. Nu valideras modellerna på bokningar gjorda juli–september 2016 och testas på bokningar gjorda oktober–december 2016. Bokningar gjorda från 2017 används inte, eftersom bokningar med lång ledtid saknas för den perioden.
- Bokningar gjorda före juli 2015 tas bort i städningen. De är ofullständiga, eftersom bara bokningar med lång ledtid finns med, och innehåller en avvikande topp hösten 2014.

**Depositionstyp**
- Ej återbetalningsbara bokningar (Non Refund) behålls, eftersom de är en del av hotellens bokningar, trots att de beter sig som gruppblock från resebyråer.
- `deposit_type` ingår i huvudmodellen, och modellerna körs även utan den som jämförelse. Planen hade den omvända ordningen.

**Baseline**
- Vanligaste klassen används inte som baseline. Den förutsäger att ingen bokning avbokas, vilket ger 63 % rätt men hittar ingen avbokning, och den kan inte rangordna bokningar.
- Lead time-regeln används som rangordning (de 20 % bokningar med längst ledtid flaggas) i stället för en fast gräns på 90 dagar. Då flaggar baseline och modeller exakt lika många bokningar, och ingen gräns behöver väljas utifrån EDA på hela datan.

**Dubbletter**
- Ingen separat analys utan dubbletter görs. Identiska rader har samma bokningsdatum och hamnar därför alltid i samma delmängd, så risken att samma bokning finns i både träning och test är redan hanterad.

**Modeller**
- Beslutsträd prövades inte av tidsskäl.


## Initiala beslut baserade på observationer i EDA (`01_eda.ipynb`)

### Städning av data (`src/data_cleaning.py`)

- Saknade värden i kolumnerna `children`, `company`, `agent` sätts till noll och `country` till "Unknown".
- Orimliga värden som noll gäster, negativt eller extremt högt värde för `adr` och Undefined i segment/kanal tas bort, totalt 187 rader
- Skapar `arrival_date` och härleder `booking_date` utifrån `arrival_date` och `lead_time`.
- Filtrerar bort bokningar gjorda före 2015-07-01. Observationer före detta datum är ofullständiga (endast långa ledtider finns representerade) och innehåller skeva blockbokningar från 2014.
- Ej återbetalningsbara bokningar (Non Refund) och övriga identiska rader behålls, eftersom de troligen är verkliga bokningar. Identiska rader hamnar alltid i samma delmängd vid uppdelningen.

### Uppdelning i träning, validering och test (`src/data_split.py`)

Uppdelningen bygger på tid, inte slump, eftersom modellen ska förutsäga avbokningar för framtida bokningar.

**Princip.** Prognosen görs vid bokningstillfället. Vid en prognostidpunkt får modellen bara lära sig av bokningar vars utfall redan var känt. Om en bokning avbokas eller blir en no-show är känt senast vid ankomsten, så träningsdata består av bokningar med ankomst före prognostidpunkten. Utvärderingsdata består av bokningar som görs efter den.

| Delmängd | Innehåll | Antal | Andel avbokade |
|---|---|---|---|
| `train_val` | Ankomst före 2016-07-01 | 40 414 | 30,8 % |
| `val` | Bokningar gjorda 2016-07-01 – 2016-09-30 | 11 643 | 35,3 % |
| `train_full` | Ankomst före 2016-10-01 | 55 393 | 32,1 % |
| `test` | Bokningar gjorda 2016-10-01 – 2016-12-31 | 14 879 | 39,6 % |

Modeller och hyperparametrar väljs genom att träna på `train_val` och utvärdera på `val`. De valda modellerna tränas sedan om på `train_full`, som även innehåller ankomsterna juli–september 2016, och utvärderas en gång på `test`.

**Val av perioder.**
- Träningsdata för validering omfattar ett helt år av ankomster, så att alla månader finns med.
- Bokningar gjorda från 2017 används inte. Datan slutar med ankomster i augusti 2017, så bokningar med lång ledtid saknas för den perioden och skulle ge ett för lätt test.
- Identiska rader har samma bokningsdatum och hamnar därför alltid i samma delmängd.

**Skillnader mellan delmängderna.** Andelen avbokade ökar från 31 % i träningsdata till 40 % i test. Testbokningarna har längre ledtid (median 70 dagar mot 39) och fler ej återbetalningsbara bokningar (14 % mot 12 %). Valideringsperioden har nästan inga ej återbetalningsbara bokningar (2 %), så effekten av `deposit_type` kan inte bedömas på valideringsdata. 

### Preprocessing (`src/preprocessing.py`)

**De features som bedöms vara kända vid bokningstillfället och som kan användas för att träna modellen är:**

- `lead_time`
- `previous_cancellations`
- `previous_bookings_not_canceled`
- `hotel`
- `arrival_date_month`
- `market_segment`
- `distribution_channel`
- `customer_type`
- `is_repeated_guest`
- `agent`
- `company`
- `reserved_room_type`

**Feature som behöver kontrollanalys:**

`deposit_type` visar ett extremt mönster för `Non Refund`: 99,36 % avbokas och 92,88 % av raderna är identiska. Samma agenter har 14–51 % avbokningar för sina övriga bokningar, så mönstret beror på depositionstypen. Variabeln är beräknad från betalningar fram till ankomst eller avbokning och var därför inte nödvändigtvis känd vid bokningstillfället.  
Den ingår i huvudmodellen, och modellerna körs även utan den som jämförelse (`include_deposit_type=False`).

**Följande features används inte, eftersom de beskriver utfallet eller kan ha ändrats efter bokningen (se Antonio m.fl., 2019, och EDA:n):**

- `reservation_status` och `reservation_status_date`
- `assigned_room_type`
- `booking_changes`
- `country`
- `adults`, `children` och `babies`
- `stays_in_weekend_nights` och `stays_in_week_nights`
- `meal`
- `adr`
- `days_in_waiting_list`
- `total_of_special_requests`
- `required_car_parking_spaces`

**Bearbetning:**
- `company` görs om till 0/1 (bokad via företag eller inte).
- `agent` behandlas som kategori. Agenter med färre än 50 bokningar i träningsdatan, och agenter som inte finns i träningsdatan, slås ihop till en gemensam kategori. Det minskar antalet agentkategorier från 210 till 72, medan cirka 5 % av bokningarna hamnar i den gemensamma kategorin.
- Numeriska variabler standardiseras och kategoriska one-hot-kodas. Preprocessorn anpassas endast på träningsdata.


## Modeller

Vi använder en enkel regelbaserad baseline som jämförelse mot våra maskininlärningsmodeller.  
Syftet är att undersöka om ML-modellerna kan identifiera fler avbokningar än en enkel regel.

1. Baseline - regelbaserad modell
2. Logistic Regression
3. Random Forest

Alla modeller jämförs vid samma antal flaggade bokningar: de 20 % med högst riskpoäng. För dem beräknas precision (hur stor andel av de flaggade som avbokades) och recall (hur stor andel av alla avbokningar som flaggades). Modeller och inställningar väljs på valideringsdata, och de valda modellerna utvärderas en gång på testdata.

### Baseline – Regelbaserad modell

Baselinen använder `lead_time`, alltså antalet dagar mellan bokning och planerad ankomst, som riskmått. Vi antar att bokningar med längre `lead_time` har högre risk att avbokas.  
Baselinen tränas inte och använder inte information om faktiska avbokningar när den väljer vilka bokningar som ska flaggas.

### Logistic Regression

Logistisk regression beräknar sannolikheten för avbokning som en viktad summa av variablerna. Modellen är enkel och lätt att tolka: varje variabel får en koefficient som visar om den ökar eller minskar risken för avbokning, vilket gör det möjligt för hotellet att förstå varför en bokning flaggas.

- Använder samma preprocessing som Random Forest (`src/preprocessing.py`), där numeriska variabler standardiseras och kategoriska one-hot-kodas.
- Tränas med scikit-learns `LogisticRegression` med standardinställningar och `max_iter=1000`.
- Bokningarna rangordnas efter den förutsagda sannolikheten, och de 20 % med högst sannolikhet flaggas.
- Körs både med och utan `deposit_type`.

En begränsning är att modellen bara fångar samband där risken ökar eller minskar jämnt med en variabel, och den hittar inte samspel mellan variabler om de inte läggs in manuellt.

### Random Forest

Random Forest består av många beslutsträd, där varje träd tränas på ett slumpmässigt urval av bokningar och variabler. Sannolikheten för avbokning är medelvärdet av trädens förutsägelser. Modellen kan fånga samband som inte är linjära och samspel mellan variabler, till exempel att lång ledtid betyder olika mycket för olika marknadssegment.

- Använder samma preprocessing som logistisk regression.
- Tränas med `n_estimators=100` och `max_depth=15`. `min_samples_leaf` valdes till [10] genom att jämföra flera värden på valideringsdatan (se `notebooks/random_forest.ipynb`).
- Bokningarna rangordnas efter den förutsagda sannolikheten, och de 20 % med högst sannolikhet flaggas.
- Körs både med och utan `deposit_type`.

Modellen är svårare att tolka än logistisk regression. Feature importance visar vilka variabler träden använder mest, men måttet räknas på träningsdatan och kan överskatta variabler med många möjliga värden, som `lead_time`. One-hot-kodade variabler som `agent` delas dessutom upp på många kolumner, vilket kan göra att de ser mindre viktiga ut än de är.

