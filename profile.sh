logName=$1
#LSTMSize=$2
echo $logName

bash getType.sh $logName

cat type.log | while read line
do
  #get the iteration number
  #echo $line
  iter=$(grep "${line}" $logName | wc -l)
  #echo $iter
  #grep "${line}"      $logName | awk '{sum+=$6}END {print "ori_time    = " sum/'$iter'}'
  #grep "${line}"      $logName | awk '{sum+=$9}END {print "opt_time    = " sum/'$iter'}'
  ori=$(grep "${line}"      $logName | awk '{sum+=$6}END {print sum/'$iter'}')
  opt=$(grep "${line}"      $logName | awk '{sum+=$9}END {print sum/'$iter'}')

  speedup=$(grep "${line}"      $logName | awk '{ori+=$6;opt+=$9}END {print ori/opt }')
  echo $line "ori_time=" $ori "opt_time=" $opt "ori/opt=" $speedup "count=" $iter
done
