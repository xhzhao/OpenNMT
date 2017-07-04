#cat gemm_all.log | awk '{print $6,$7,$8 $9 $10 $11 $12 $13 $14 $15 $16 $17 $18}' &> temp.log
cat gemm_all.log | awk '{print $6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18}' &> temp.log
sort temp.log | uniq > knl_gemm_type_10_50.log
