/* 
 * Course: High Performance Computing 2021/2022
 * 
 * Lecturer: Francesco Moscato	fmoscato@unisa.it
 *
 * Group:
 * Battipaglia	Lucia  		0622701758  	l.battipaglia6@studenti.unisa.it               
 * Canzolino	Gianluca  	06227001806  	g.canzolino3@studenti.unisa.it 
 * Farina		Luigi		0622701754  	l.farina16@studenti.unisa.it
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

	For sequential: ./file.c 1 [Dim_Array]
	For parallel: 	./file.c 2 [Dim_Array] [n_threads]

 */

 /**
	@file main.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define STARTTIME(ID)\
	double start42##ID;\
	double stop42##ID;\
	start42##ID = omp_get_wtime();

#define STOPTIME(ID)\
	stop42##ID = omp_get_wtime();\
	double time##ID;\
	time##ID= stop42##ID-start42##ID;\

#define PRINTTIME(ID)\
				printf("%d;%lf;",n_valori,time##ID);\

/**
 * @brief This function sorts array in parallel mode.
 * @param A        	pointer to the vector to sort.
 * @param n  	   	size of array.
 * @param n_threads number of threads.
 */
void countingSortParallel(int *A, int n,int n_threads){
    int max = A[0];
    int min = A[0];
		int *B,*C;
		int index;

		#pragma omp parallel default(none) shared(n, A, min, max, B, C, n_threads, index) num_threads(n_threads)
		{
				//FASE 1
				//*Fase di ricerca del minimo e del massimo
				#pragma omp for reduction(min:min) reduction(max:max)
						for(int i=1; i<n; i++){
									if(A[i] > max){
											max = A[i];
									}
									if(A[i] < min){
											min = A[i];
									}
						}

				//FASE 2
				//Inizializzo un vettore C grande quanto l'intervallo tra min e max
				#pragma omp single
				{
						C = (int*)malloc((max-min+1)*sizeof(int));
				}

				//FASE 3
				//Fase di reset del vettore C a 0
				#pragma omp for
				    for(int i=0;i<(max-min+1);i++){
				        C[i]=0;
				    }

				//FASE 4
				//Fase di popolamento del vettore C, incrementando il vettore
				#pragma omp for
					for(index=0;index<n;index++){
							C[A[index]-min]=C[A[index]-min]+1;
					}

				//FASE 5
				//SOTTOFASE 5_1: creo un vettore di n_threads+1 elementi
				#pragma omp barrier
				{
					#pragma omp single
					{
							B = (int*)malloc((n_threads+1)*sizeof(int));
					}
				}

				//SOTTOFASE 5_2: riempio il vettore B con tutti 0
				#pragma omp for
				    for(int i=0;i<=n_threads;i++){
				        B[i]=0;
				    }

				//SOTTOFASE 5_3: riempio il vettore B con la somma parziale delle occorrenze contenute in C
				#pragma omp barrier
				{
				    for(int i=(omp_get_thread_num()*((max-min+1)/n_threads));i<((omp_get_thread_num()+1)*((max-min+1)/n_threads));i++){
								B[omp_get_thread_num()+1]+= C[i];
				    }
				}

				//SOTTOFASE 5_4: riempio il vettore B con tutti gli indici di avvio di ogni singolo thread
				#pragma omp barrier
				{
					#pragma omp single
					{
							for(int i=0;i<n_threads-1;i++){
									B[i+1] = B[i] + B[i+1];
							}
							B[n_threads]=n;
					}
				}

				//SOTTOFASE 5_5: riempio il vettore A a seconda delle occorrenze contenute in C
				#pragma omp barrier
				{
					int start_index = B[omp_get_thread_num()];
					int start = omp_get_thread_num()*((max-min+1)/n_threads);
					int stop;
					if(omp_get_thread_num()+1 == n_threads){
						stop = max-min+1;
					}
					else{
						stop = (omp_get_thread_num()+1)*((max-min+1)/n_threads);
					}
					int start_value = min;
					for(int i=start;i<stop;i++){
						  while (C[i]>0){
			            A[start_index]=i+start_value;
			            start_index++;
			            C[i]--;
			        }
			    }
				}
		}

		free(B);
		free(C);

}

/**
 * @brief This function sorts array in sequential mode.
 * @param A        	pointer to the vector to sort.
 * @param n  	   	size of array.
 */
void countingSortSequential(int *A, int n){
    int max = A[0];
    int min = A[0];
		int *C;

		// Fase1
    for(int i=1; i<n; i++){
        if(A[i] > max){
            max = A[i];
        }
        if(A[i] < min){
            min = A[i];
        }
    }

		#// Fase2
    C = (int*)malloc((max-min+1)*sizeof(int));

		// Fase3
    for(int i=0;i<(max-min+1);i++){
        C[i]=0;
    }

		// Fase4
    for(int i=0;i<n;i++){
        C[A[i]-min]=C[A[i]-min]+1;
    }

		//Fase5
    int k=0;

    for(int i=0;i<(max-min+1);i++){
        while (C[i]>0){
            A[k]=i+min;
            k++;
            C[i]--;
        }
    }
}


int main(int argc, char** argv) {
		int type = atoi(argv[1]);
		int n_valori = atoi(argv[2]);
    int *A;

		A = (int*)malloc((n_valori)*sizeof(int));

		for(int i=0;i<n_valori;i++){
		   A[i]=n_valori - i;
			 //A[i]=abs(rand())%(1000);
		}

		if(type == 1){
	    STARTTIME(2)
	    countingSortSequential(A,n_valori);
			STOPTIME(2)
	    PRINTTIME(2)
		}

		if(type == 2){
			STARTTIME(1)
	    countingSortParallel(A,n_valori,atoi(argv[3]));
			STOPTIME(1)
	    PRINTTIME(1)
		}

		free(A);

		return (EXIT_SUCCESS);
}
