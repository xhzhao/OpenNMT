#grep "transa=110, transb=110, m=1000, n=64, k=2000, lda=1000, alpha=1.0000, ldb=2000, beta=0.0000, ldc=1000" gemm_all.log | awk '{sum+=$4} END {print sum/40}'


cat knl_gemm_type_10_50.log | while read line
do
   #echo "${line}"
   grep "${line}" gemm_all.log | awk '{sum+=$4} END {print sum/40}'
done

