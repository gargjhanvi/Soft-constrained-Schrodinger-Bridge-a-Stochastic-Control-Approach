#! /bin/sh

s="sb"
b=1
#s="s1"
#b=0.990099
#s="s2"
#b=0.9803922
#s="s3"
#b=0.952381
Rscript cauchy_sim.R 200 20    $b ${s}_20.txt
Rscript cauchy_sim.R 200 50    $b ${s}_50.txt
Rscript cauchy_sim.R 200 100   $b ${s}_100.txt
Rscript cauchy_sim.R 200 200   $b ${s}_200.txt
Rscript cauchy_sim.R 200 500   $b ${s}_500.txt
Rscript cauchy_sim.R 200 1000  $b ${s}_1000.txt
Rscript qqplot.R $b ${s}_1000 

