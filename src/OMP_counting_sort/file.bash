#!/bin/bash

YELLOW='\033[0;92m'
NORMAL='\033[0;39m'
BLUE='\033[0;34m'

TIMEFORMAT='%3U;%3E;%3S;%P'
TIMESTAMP=$(date +%F.%T)

NMEASURES=100
ARRAY_SIZE=(10000 50000 100000 500000 1000000) #carico
ARRAY_THS=(0 1 2 4 6 8 12 16 24 32)
ARRAY_OPT=(0 1 2 3)


SCRIPTPATH=$2

for size in "${ARRAY_SIZE[@]}"; do
	for nTh in "${ARRAY_THS[@]}"; do
		for opt in "${ARRAY_OPT[@]}"; do

            nThStr=$(printf "%02d" $nTh)

            OUT_FILE=$SCRIPTPATH/measure/$TIMESTAMP/SIZE-$size-O$opt/SIZE-$size-NTH-$nThStr-O$opt.csv

            mkdir -p $(dirname $OUT_FILE)

            printf "${NORMAL}"
            echo $(basename $OUT_FILE)
            echo 'dimArray;timeInt;user;real;sys;pCPU'>$OUT_FILE

            for (( nExec = 0 ; nExec < $NMEASURES ; nExec += 1 )) ; do
								if [[ $nTh -eq 0 ]]; then
									(time $1/eseguibileO$opt 1 $size)2>&1 | sed -e 's/,/./g' -e 's/\n/;/g'  >> $OUT_FILE
								else
									(time $1/eseguibileO$opt 2 $size $nTh)2>&1 | sed -e 's/,/./g' -e 's/\n/;/g'  >> $OUT_FILE
								fi
                printf "\r${YELLOW}> ${BLUE}%5d/%d ${YELLOW}%3.1d%% [ " $(expr $nExec + 1) $NMEASURES $(expr \( \( $nExec + 1  \) \* 100 \) / $NMEASURES)
                printf "¿%.0s" $(seq -s " " 1 $(expr \( $nExec \* 40 \) / $NMEASURES))
                printf " %.0s" $(seq -s " " 1 $(expr 40 - \( $nExec \* 40 \) / $NMEASURES))
                printf "] "

            done

            printf "\n"

        done
    done
done
