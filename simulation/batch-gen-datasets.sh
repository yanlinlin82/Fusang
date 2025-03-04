#!/bin/bash

# ref. Supplementary Table 3 and Supplementary Table 5 in the paper
# only 1/1000 of training sets' data amount

bash gen-dataset.sh out/S1U 42  600 5 '[5, 40]'  200  200 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.03, 0.3]' 0.01 0.25 10
bash gen-dataset.sh out/S2U 42 6000 5 '[5, 40]' 1000 1000 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.03, 0.3]' 0.01 0.25 50
bash gen-dataset.sh out/S1G 42  600 5 '[5, 40]'  200  200 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.03, 0.3]' 0.01 0.25 10
bash gen-dataset.sh out/S2G 42 6000 5 '[5, 40]' 1000 1000 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.03, 0.3]' 0.01 0.25 50

bash gen-dataset.sh out/C1U 42  600 5 '[5, 40]'  200  200 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.03, 0.2]' 0.01 0.1  10
bash gen-dataset.sh out/C2U 42 6000 5 '[5, 40]' 1000 1000 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.03, 0.2]' 0.01 0.1  50
bash gen-dataset.sh out/C1G 42  600 5 '[5, 40]'  200  200 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.03, 0.2]' 0.01 0.1  10
bash gen-dataset.sh out/C2G 42 6000 5 '[5, 40]' 1000 1000 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.03, 0.2]' 0.01 0.1  50

bash gen-dataset.sh out/N1U 42  600 5 '[5, 40]'  200  200 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.15, 0.3]' 0.1  0.25 10
bash gen-dataset.sh out/N2U 42 6000 5 '[5, 40]' 1000 1000 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.15, 0.3]' 0.1  0.25 50
bash gen-dataset.sh out/N1G 42  600 5 '[5, 40]'  200  200 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.15, 0.3]' 0.1  0.25 10
bash gen-dataset.sh out/N2G 42 6000 5 '[5, 40]' 1000 1000 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.15, 0.3]' 0.1  0.25 50
