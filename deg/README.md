# DEG
Differentially Expressed Genes

## Overview
Use the DEGA package to compute nice volcano plot of DEG. DEGA is a very cool tool, lot of stuff tou can add if you want (https://github.com/LucaMenestrina/DEGA)

## Usage
python3 deg.py -gc counts.csv -gp pheno.csv -o /tmp/deg465 -l strain
  
  * -gc : gene count, csv file
  * -gp : gene pheno, csv file
  * -o : output folder
  * -l : name of the column in pheno file to use as label
