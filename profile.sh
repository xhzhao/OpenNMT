logName=$1
echo $logName

#get the iteration number
iter=$(grep "iteration = " $logName | wc -l)
echo $iter

grep iteration     $logName | awk '{sum+=$4}END {print "Iteration time    = " sum/'$iter'}'


grep Linear_F      $logName | awk '{sum+=$3}END {print "Linear_F    = " sum/'$iter'}'
grep Linear_B1     $logName | awk '{sum+=$3}END {print "Linear_B1   = " sum/'$iter'}'
grep Linear_B2     $logName | awk '{sum+=$3}END {print "Linear_B2   = " sum/'$iter'}'

grep Tanh_F        $logName | awk '{sum+=$3}END {print "Tanh_F      = " sum/'$iter'}'
grep Tanh_B        $logName | awk '{sum+=$3}END {print "Tanh_B      = " sum/'$iter'}'

grep Sigmoid_F     $logName | awk '{sum+=$3}END {print "Sigmoid_F   = " sum/'$iter'}'
grep Sigmoid_B     $logName | awk '{sum+=$3}END {print "Sigmoid_B   = " sum/'$iter'}'

grep MM_F          $logName | awk '{sum+=$3}END {print "MM_F        = " sum/'$iter'}'
grep MM_B          $logName | awk '{sum+=$3}END {print "MM_B        = " sum/'$iter'}'

grep CAddTable_F   $logName | awk '{sum+=$3}END {print "CAddTable_F = " sum/'$iter'}'
grep CAddTable_B   $logName | awk '{sum+=$3}END {print "CAddTable_B = " sum/'$iter'}'

grep CMulTable_F   $logName | awk '{sum+=$3}END {print "CMulTable_F = " sum/'$iter'}'
grep CMulTable_B   $logName | awk '{sum+=$3}END {print "CMulTable_B = " sum/'$iter'}'

grep JoinTable_F   $logName | awk '{sum+=$3}END {print "JoinTable_F = " sum/'$iter'}'
grep JoinTable_B   $logName | awk '{sum+=$3}END {print "JoinTable_B = " sum/'$iter'}'
