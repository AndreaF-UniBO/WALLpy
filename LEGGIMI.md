🇮🇹 [Leggi in italiano](LEGGIMI.md) | 🇬🇧 [Read in English](README.md)

# PyWALL v13

PyWALL è un'applicazione desktop che assiste la segmentazione di immagini di paramenti murari e la produzione di elaborati grafici archeologici. La versione 13 è l'attuale release alpha pubblica.

[Sito del progetto](https://andreaf-unibo.github.io/) · [Repository del codice sorgente](https://github.com/AndreaF-UniBO/PyWALL) · [Release v0.13.0](https://github.com/AndreaF-UniBO/PyWALL/releases/tag/v0.13.0)

L'applicazione offre due procedure di segmentazione indipendenti:

- un metodo di base autosufficiente K-Means, fondato su descrittori cromatici e tessiturali;
- la generazione opzionale di maschere automatiche tramite l'implementazione ufficiale Meta di Segment Anything Model 2 (SAM 2).

La precedente pipeline Deep Learning personalizzata non fa parte di PyWALL v13.

## Stato

PyWALL v13 è un software di ricerca in versione alpha (`0.13.0`). I risultati devono essere controllati da un archeologo e non devono essere considerati interpretazioni validate in assenza di una valutazione esperta.

![Interfaccia desktop di PyWALL v13](docs/assets/pywall-v13-interface.png)

## Funzionalità principali

- Caricamento di fotografie di paramenti murari in formato JPEG e PNG.
- Segmentazione di un'immagine tramite K-Means o Meta SAM 2.
- Rifinitura della maschera binaria della muratura tramite controlli morfologici e cromatici.
- Correzione manuale del risultato mediante uno strumento gomma.
- Ispezione dei contorni e delle maschere binarie.
- Esportazione delle maschere in formato PNG e dei contorni vettorializzati in formato DXF.

## Requisiti

- Windows 10 o Windows 11 per gli script di configurazione PowerShell forniti.
- Python 3.11 a 64 bit.
- Almeno 8 GB di spazio libero durante la configurazione. Dopo l'installazione, il pacchetto locale verificato occupa circa 5,2 GB.
- Per SAM 2 è consigliata una GPU NVIDIA compatibile con CUDA. L'inferenza su CPU è possibile, ma può essere molto lenta.

Meta consiglia WSL per SAM 2 su Windows. Questo pacchetto fornisce anche una configurazione per Windows nativo destinata alla validazione locale; la compatibilità deve essere confermata sul computer di destinazione.

## Installazione

Apri PowerShell in questa cartella ed esegui:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\download_sam2_checkpoint.ps1
.\scripts\test_local.ps1
.\scripts\run_pywall.ps1
```

Gli script creano un ambiente isolato `.venv`, installano l'applicazione principale, PyTorch e il pacchetto ufficiale Meta SAM 2, scaricano il checkpoint ufficiale `sam2.1_hiera_base_plus.pt`, ne verificano il checksum SHA-256 ed eseguono gli smoke test.

Usa `setup_windows.ps1 -Cpu` soltanto quando CUDA non è disponibile.

Ulteriori istruzioni dettagliate in italiano sono disponibili in [README_PyWALL_v13.md](README_PyWALL_v13.md).

## Installazione manuale

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$env:SAM2_BUILD_CUDA = "0"
.\.venv\Scripts\python.exe -m pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

Scarica quindi il checkpoint ufficiale Meta seguendo le [istruzioni per il checkpoint](checkpoints/README.md).

## Avvio dell'applicazione

```powershell
.\scripts\run_pywall.ps1
```

oppure:

```powershell
.\.venv\Scripts\python.exe PyWALL_v13.py
```

## Procedura minima

1. Seleziona **Carica Immagine** e scegli uno dei file pubblicati nella cartella `samples/`, oppure un'altra immagine JPEG o PNG che sei autorizzato a elaborare.
2. Seleziona **K-Means** per la procedura di base oppure **SAM 2** per il modello ufficiale Meta.
3. Regola i controlli dei filtri e, quando necessario, correggi manualmente la maschera.
4. Esamina i contorni e la maschera binaria.
5. Esporta una maschera PNG o i contorni DXF.

## Input e output

Gli input sono normali immagini raster supportate da Pillow, principalmente in formato JPEG e PNG. Internamente le immagini vengono convertite in RGB.

La maschera binaria interna utilizza il valore `255` per gli elementi murari e `0` per la malta. PyWALL può salvare la maschera come PNG ed esportare i contorni individuati in formato DXF utilizzando la scala inserita nell'interfaccia.

## Immagini di esempio

Il repository include tre fotografie scattate da Andrea Fiorini ed espressamente autorizzate alla distribuzione come input di prova per PyWALL: `071.png`, `080.png` e `input_04.png`. I metadati incorporati sono stati rimossi prima della pubblicazione senza modificare i pixel. Altre tre immagini usate per la validazione locale restano escluse perché la loro pubblicazione non è stata autorizzata. Consulta [samples/README.md](samples/README.md) per le specifiche condizioni di copyright.

## Gestione del modello SAM 2

PyWALL utilizza esclusivamente il backend ufficiale [`facebookresearch/sam2`](https://github.com/facebookresearch/sam2). Non utilizza Ultralytics e non scarica mai i pesi senza un'azione esplicita dell'utente. Il checkpoint ufficiale viene conservato localmente nella cartella `checkpoints/` e rimane escluso da Git.

## Limitazioni note

- L'installazione di SAM 2 su Windows nativo è fornita secondo il principio del massimo impegno; Meta consiglia WSL.
- Il primo caricamento del modello può richiedere quantità considerevoli di RAM e memoria GPU.
- Le maschere automatiche di SAM 2 identificano regioni visive; l'euristica di fusione basata sull'area non classifica le relazioni archeologiche.
- La qualità della segmentazione dipende dall'illuminazione, dalla scala dell'immagine, dalle condizioni della superficie e dalla scelta dei parametri.
- Devono essere considerati testati soltanto gli ambienti effettivamente indicati nel rapporto di prova.

## Struttura del progetto

```text
PyWALL_v13.py             Applicazione desktop Tkinter principale
sam2_segmentation.py      Adattatore ufficiale Meta SAM 2
checkpoints/              Checkpoint Meta locale e informazioni di verifica
samples/                  Tre esempi autorizzati e relative condizioni di copyright
scripts/                  Script Windows di configurazione, download, test e avvio
tests/                    Test unitari, della GUI e smoke test opzionali del modello
docs/                     Note sulla verifica locale
pyproject.toml            Metadati del pacchetto e delle dipendenze
```

## Pubblicazione associata

La genesi di PyWALL, l'architettura della versione precedente, il metodo di validazione e l'applicazione archeologica sul campo sono descritti in:

> Fiorini, A. (2026). *Munsell Soil Finder e PyWALL: strumenti per la documentazione cromatica e il rilievo dei paramenti murari in archeologia*. **Archeologia e Calcolatori**, 37.1, 255–276. [Pagina dell'articolo e download del PDF](https://www.archcalc.cnr.it/journal/articles/1498) · [DOI: 10.19282/ac.37.1.2026.13](https://doi.org/10.19282/ac.37.1.2026.13)

I software basati sull'intelligenza artificiale evolvono con una rapidità insolita rispetto ai tempi della pubblicazione scientifica; un articolo può quindi descrivere un'implementazione già in parte superata al momento dell'uscita. Questo paper è stato pubblicato nell'agosto 2026, ma documenta una versione di PyWALL sviluppata nel 2025, prima dell'integrazione di SAM 2. Descrive pertanto K-Means e una DynUNet pre-addestrata ottenuta tramite TopoMortar. L'attuale PyWALL v13 mantiene K-Means, elimina il precedente percorso DynUNet/TopoMortar e utilizza invece il backend ufficiale Meta SAM 2. Il paper resta il riferimento scientifico per la genesi del progetto, il disegno della validazione e il test archeologico sul campo; questo repository è la fonte autorevole per l'installazione e le funzionalità correnti.

## Citazione e paternità

PyWALL è opera di Andrea Fiorini, Dipartimento di Storia Culture Civiltà, Alma Mater Studiorum – Università di Bologna. Consulta [CITATION.cff](CITATION.cff) per i metadati di citazione in formato leggibile dalle macchine.

## Licenza

Il codice sorgente di PyWALL è distribuito secondo la [Licenza Apache 2.0](LICENSE). Le fotografie nella cartella `samples/` non sono coperte da questa licenza e rimangono copyright di Andrea Fiorini secondo le condizioni indicate in [samples/README.md](samples/README.md). I pacchetti di terze parti e il modello Meta SAM 2 rimangono soggetti alle rispettive condizioni stabilite dai fornitori originali; consulta [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) e [NOTICE](NOTICE).
