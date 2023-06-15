#!/bin/sh
readN=SRR835433

refer="../datas/GCF_000001405.39_GRCh38.p13_genomic.fna"
read="../datas/${readN}.fasta"

sam="../results2/${readN}_1cpu_8warp_small.sam"
log="../results2/${readN}_1cpu_8warp_small.log"

./bwa-mem2 mem -t 20 ${refer} $read > ${sam} 2> ${log}
