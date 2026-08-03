# PyWALL v13 — installazione e prova locale

PyWALL v13 è una versione alpha pubblica dell'applicazione desktop per la segmentazione assistita di immagini murarie e la produzione di maschere PNG e contorni DXF.

[Sito del progetto](https://andreaf-unibo.github.io/) · [Repository](https://github.com/AndreaF-UniBO/PyWALL) · [Release v0.13.0](https://github.com/AndreaF-UniBO/PyWALL/releases/tag/v0.13.0)

## Cosa contiene

- segmentazione K-Means;
- segmentazione con il solo backend ufficiale Meta SAM 2;
- strumenti di correzione e filtraggio;
- visualizzazione della maschera e dei bordi;
- esportazione PNG e DXF;
- supporto per immagini JPEG e PNG fornite localmente dall'utente.

La precedente funzione **Segmentazione DL** non è presente.

## Preparazione automatica su Windows

Apri PowerShell nella cartella `C:\WALLpy\PyWALL_v13` ed esegui:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\download_sam2_checkpoint.ps1
.\scripts\test_local.ps1
```

Lo script di configurazione usa Python 3.11, crea `.venv`, installa le dipendenze di base, PyTorch con supporto CUDA 12.1 e il pacchetto ufficiale Meta SAM 2 fissato a una revisione precisa.

Per una macchina senza GPU NVIDIA:

```powershell
.\scripts\setup_windows.ps1 -Cpu
```

## Avvio

```powershell
.\scripts\run_pywall.ps1
```

Per provare subito il flusso di base:

1. premi **Carica Immagine**;
2. apri una tua immagine JPEG o PNG che puoi legittimamente elaborare, oppure un'immagine presente nella cartella locale `samples`;
3. premi **K-Means**;
4. controlla filtri, maschera, bordi ed esportazione.

Per SAM 2:

1. verifica che `checkpoints\sam2.1_hiera_base_plus.pt` sia presente;
2. carica un'immagine da `samples`;
3. premi **SAM 2**;
4. attendi il primo caricamento del modello;
5. controlla visivamente il risultato prima di esportarlo.

## Test con inferenza SAM 2 reale

Il test ordinario non carica il modello. Dopo avere scaricato il checkpoint, puoi eseguire anche un'inferenza reale ridotta su un'immagine sintetica generata dal test:

```powershell
.\scripts\test_local.ps1 -Sam2
```

## Note importanti

- Meta raccomanda WSL/Ubuntu per SAM 2 su Windows. La configurazione nativa fornita qui deve quindi essere verificata su questo computer.
- I pesi vengono scaricati esclusivamente dallo script esplicito e verificati con SHA-256.
- `071.png`, `080.png` e `input_04.png` sono fotografie di Andrea Fiorini autorizzate alla pubblicazione come input di prova; i metadati incorporati sono stati rimossi. Le altre tre immagini della validazione locale restano escluse. Consulta `samples/README.md` per i termini separati dal codice Apache-2.0.
- Gli output automatici sono supporti operativi e richiedono sempre controllo archeologico esperto.

Vedi anche le [istruzioni per il checkpoint](checkpoints/README.md), il [rapporto dei test locali](docs/local-testing.md) e la [licenza](LICENSE).
