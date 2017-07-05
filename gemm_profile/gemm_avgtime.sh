
logName=$1
echo $logName

cat $logName | while read line
do
   iter=$(grep "${line}" gemm_all.log | wc -l )
   grep "${line}" gemm_all.log | awk '{sum+=$4} END {print sum/'$iter'}'
done

