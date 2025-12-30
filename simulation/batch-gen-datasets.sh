#!/bin/bash

script_dir=$(dirname $(readlink -f $0))

# ref. Supplementary Table 3 and Supplementary Table 5 in the paper

bash $script_dir/gen-dataset.sh $script_dir/out/S1U 42  600000 5 '[5, 40]'  200  200 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.03, 0.3]' 0.01 0.25 10
#bash $script_dir/gen-dataset.sh $script_dir/out/S2U 42 6000000 5 '[5, 40]' 1000 1000 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.03, 0.3]' 0.01 0.25 50
#bash $script_dir/gen-dataset.sh $script_dir/out/S1G 42  600000 5 '[5, 40]'  200  200 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.03, 0.3]' 0.01 0.25 10
#bash $script_dir/gen-dataset.sh $script_dir/out/S2G 42 6000000 5 '[5, 40]' 1000 1000 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.03, 0.3]' 0.01 0.25 50

#bash $script_$script_dir/dir/gen-dataset.sh $script_dir/out/C1U 42  600 5 '[5, 40]'  200  200 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.03, 0.2]' 0.01 0.1  10
#bash $script_dir/gen-dataset.sh $script_dir/out/C2U 42 6000000 5 '[5, 40]' 1000 1000 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.03, 0.2]' 0.01 0.1  50
#bash $script_dir/gen-dataset.sh $script_dir/out/C1G 42  600000 5 '[5, 40]'  200  200 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.03, 0.2]' 0.01 0.1  10
#bash $script_dir/gen-dataset.sh $script_dir/out/C2G 42 6000000 5 '[5, 40]' 1000 1000 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.03, 0.2]' 0.01 0.1  50

#bash $script_$script_dir/dir/gen-dataset.sh $script_dir/out/N1U 42  600 5 '[5, 40]'  200  200 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.15, 0.3]' 0.1  0.25 10
#bash $script_dir/gen-dataset.sh $script_dir/out/N2U 42 6000000 5 '[5, 40]' 1000 1000 20 '[0, 0.001, 0.3]' '[0, 0.001, 0.3]' '[0.15, 0.3]' 0.1  0.25 50
#bash $script_dir/gen-dataset.sh $script_dir/out/N1G 42  600000 5 '[5, 40]'  200  200 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.15, 0.3]' 0.1  0.25 10
#bash $script_dir/gen-dataset.sh $script_dir/out/N2G 42 6000000 5 '[5, 40]' 1000 1000 20 '[1, 0.5,   0.3]' '[1, 0.5,   0.3]' '[0.15, 0.3]' 0.1  0.25 50
