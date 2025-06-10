# Example Data

## Raw sequences

### COX1 gene of mammals

File: `mammals-cox1.fa`

Search "COX1[Gene] AND mammals[Organism] AND 500:1500[Sequence Length]" on NCBI. <https://www.ncbi.nlm.nih.gov/nuccore/?term=COX1%5BGene%5D+AND+mammals%5BOrganism%5D+AND+500%3A1500%5BSequence+Length%5D>

Got 11976 sequences (on 2025/6/10).

Pick 1-40, 41-80, 81-120 sequences, separately.

### SARS COV2 sequences

Sequence (random picking 2000 sequences) are downloaded from NCBI. <https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?SeqType_s=Nucleotide&VirusLineage_ss=taxid:2697049>

Pick 1-40, 41-80, 81-120 sequences (of first fa), separately.

### CYTB gene sequences

Search "CYTB[Gene] AND 500:1500[Sequence Length]" on NCBI. <https://www.ncbi.nlm.nih.gov/nuccore/?term=CYTB%5BGene%5D+AND+500%3A1500%5BSequence+Length%5D>

Got 456968 sequences (on 2025/6/10).

Pick 1-40, 41-80, 81-120 sequences, separately.

## Multiple alignment

Run following command to multiple alignment the sequences.

```sh
mufft xxx.fa > xxx.aligned.fa
```

Fusang requires multiple alignment file as input.
