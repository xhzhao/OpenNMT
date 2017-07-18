sed -i "s/=/ /g" result.log
sed -i "s/,/ /g" result.log
cat result.log | awk '{print $2 "\t" $4 "\t" $6 "\t" $8}' OFS="," > test2.xls
