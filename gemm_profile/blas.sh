
#grep sgemm one.log | awk '{sum+=$4} END {print sum}'
#grep sgemv one.log | awk '{sum+=$4} END {print sum}'
#grep saxpy one.log | awk '{sum+=$4} END {print sum}'

logName=$1

#grep iteration blas_10_50.log | awk '{sum+=$4} END {print sum/40}'
#grep sgemm blas_10_50.log | awk '{sum+=$4} END {print sum/40}'
#grep sgemv blas_10_50.log | awk '{sum+=$4} END {print sum/40}'
#grep saxpy blas_10_50.log | awk '{sum+=$4} END {print sum/40}'

grep "iteration =" $logName | awk '{sum+=$4} END {print sum/40}'
grep sgemm $logName | awk '{sum+=$4} END {print sum/40}'
grep sgemv $logName | awk '{sum+=$4} END {print sum/40}'
grep saxpy $logName | awk '{sum+=$4} END {print sum/40}'

grep sgemm $logName &> gemm_all.log
