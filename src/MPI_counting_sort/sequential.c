/*
 * Course: High Performance Computing 2021/2022
 *
 * Lecturer: Francesco Moscato	fmoscato@unisa.it
 *
 * Group:
 * Battipaglia	 Lucia  		 0622701758  	l.battipaglia6@studenti.unisa.it
 * Canzolino	   Gianluca  	 0622701806  	g.canzolino3@studenti.unisa.it
 * Farina		     Luigi		   0622701754  	l.farina16@studenti.unisa.it
 *
 * Copyright (C) 2021 - All Rights Reserved
 *
 * This file is part of CommonAssignment1.
 *
 * CommonAssignment1 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CommonAssignment1 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CommonAssignment1.  If not, see <http://www.gnu.org/licenses/>.

 HOW TO USE

  ./file 1 [Dim_Array]

 */

 /**
	@file main_seq.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STARTTIME(id)                           \
  clock_t start_time_42_##id, end_time_42_##id; \
  start_time_42_##id = clock();

#define STOPTIME(id)        \
  end_time_42_##id = clock(); \
  double time##id = ((double)(end_time_42_##id - start_time_42_##id)) / CLOCKS_PER_SEC;

#define PRINTTIME(id)\
  printf("%lf;",time##id);

void countingSortSequential(int *A, int n){
    int max = A[0];
    int min = A[0];
		int *C;

    //Fase 1
    //Complex: O(n)
    for(int i=1; i<n; i++){
        if(A[i] > max){
            max = A[i];
        }
        if(A[i] < min){
            min = A[i];
        }
    }

    //Fase 2
    //Complex: O(1)
    C = (int*)malloc((max-min+1)*sizeof(int));

    //Fase 3
    //Complex: O(k) where k is range of value (max - min + 1)
    for(int i=0;i<(max-min+1);i++){
        C[i]=0;
    }

    //Fase 4
    //Complex: O(n)
    for(int i=0;i<n;i++){
        C[A[i]-min]=C[A[i]-min]+1;
    }


		//Fase 5
    //Complex: O(n)
    int k=0;
    for(int i=0;i<(max-min+1);i++){
        while (C[i]>0){
            A[k]=i+min;
            k++;
            C[i]--;
        }
    }
}

/*
#define stampa
//*/

int main(int argc, char** argv) {
		int n_valori = atoi(argv[1]);
    int *A;
		srand(time(NULL));
		int max_value = 256;

		A = (int*)malloc((n_valori)*sizeof(int));

		for(int i=0;i<n_valori;i++){
		   A[i]=i%max_value;
			 //A[i]=abs(rand())%(1000);
			 //A[i]=abs(rand())%(max_value);
		}

    STARTTIME(2)
    countingSortSequential(A,n_valori);
		STOPTIME(2)
		//printf("Sequenziale:\n");
    PRINTTIME(2)
		//
		#ifdef stampa
			for(int i=0; i<n_valori-1; i++){
					if(A[i] > A[i+1]){
						printf("NON ORDINATO! A[%d]:%d > A[%d]:%d\n", i, A[i], i+1, A[i+1]);
						break;
					}
			}
			printf("ORDINATO\n");
		#endif

		free(A);

		return (EXIT_SUCCESS);
}
