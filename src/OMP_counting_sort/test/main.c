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



/*
#define debug//*/

//*
#define parallel_test//*/
//*
#define sequential//*/

/*
#define stampa//*/
/*
#define debug_stampa_iterazioni//*/

//FASI
//*
#define Fase1//*/
//*
#define Fase2//*/
//*
#define Fase3//*/
//*
#define Fase4//*/
//*
#define Fase5//*/

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
				#ifdef debug_stampa_iterazioni
					printf("Sono thread %d:\n", omp_get_thread_num());
					int num = 0;
				#endif

				#pragma omp for reduction(min:min) reduction(max:max)
						for(int i=1; i<n; i++){
									if(A[i] > max){
											max = A[i];
									}
									if(A[i] < min){
											min = A[i];
									}
									#ifdef debug_stampa_iterazioni
										num++;
									#endif
						}

				#ifdef debug_stampa_iterazioni
					printf("Thread: %d, Fase 1: iterazioni: %d su %d\n", omp_get_thread_num(),num, n);
				#endif

				#ifdef debug
					printf("Max: %d, Min: %d\n", max, min);
				#endif

				//FASE 2
				//Inizializzo un vettore C grande quanto l'intervallo tra min e max
				#ifdef debug_stampa_iterazioni
					num = 0;
				#endif
				#pragma omp single
				{
						C = (int*)malloc((max-min+1)*sizeof(int));
						#ifdef debug_stampa_iterazioni
							num++;
						#endif
				}
				#ifdef debug_stampa_iterazioni
					printf("Thread: %d, Fase 2: iterazioni: %d su %d\n", omp_get_thread_num(),num, 1);
				#endif

				//FASE 3
				//Fase di reset del vettore C a 0
				#ifdef debug_stampa_iterazioni
					num = 0;
				#endif
				#pragma omp for
				    for(int i=0;i<(max-min+1);i++){
				        C[i]=0;
								#ifdef debug_stampa_iterazioni
									num++;
								#endif
				    }
				#ifdef debug_stampa_iterazioni
					printf("Thread: %d, Fase 3: iterazioni: %d su %d\n", omp_get_thread_num(),num, (max-min+1));
				#endif

				#ifdef debug
					#pragma omp single
					{
							for(int i=0;i<(max-min+1);i++){
									printf("C Val %d: %d\n", i, C[i]);
							}
					}
				#endif

				//FASE 4
				//Fase di popolamento del vettore C, incrementando il vettore
				#ifdef debug_stampa_iterazioni
					num = 0;
				#endif
				#pragma omp for
					for(index=0;index<n;index++){
							C[A[index]-min]=C[A[index]-min]+1;
					}

				#ifdef debug_stampa_iterazioni
					printf("Thread: %d, Fase 4: iterazioni: %d su %d\n", omp_get_thread_num(),num, n);
				#endif

				#ifdef debug
					#pragma omp single
					{
							for(int i=0;i<(max-min+1);i++){
									printf("C Val %d: %d\n", i, C[i]);
							}
					}
				#endif

				//FASE 5

				//SOTTOFASE 5_1: creo un vettore di n_threads+1 elementi
				#ifdef debug_stampa_iterazioni
					num = 0;
				#endif
				#pragma omp barrier
				{
					#pragma omp single
					{
							B = (int*)malloc((n_threads+1)*sizeof(int));
							#ifdef debug_stampa_iterazioni
								num++;
							#endif
					}
				}
				#ifdef debug_stampa_iterazioni
					printf("Thread: %d, Fase 5_1: iterazioni: %d su %d\n", omp_get_thread_num(),num, 1);
				#endif

				//SOTTOFASE 5_2: riempio il vettore B con tutti 0
				#ifdef debug_stampa_iterazioni
					num = 0;
				#endif

				#pragma omp for
				    for(int i=0;i<=n_threads;i++){
				        B[i]=0;
								#ifdef debug_stampa_iterazioni
									num++;
								#endif
				    }

				#ifdef debug_stampa_iterazioni
					printf("Thread: %d, Fase 5_2: iterazioni: %d su %d\n", omp_get_thread_num(),num, n_threads+1);
				#endif

				#ifdef debug
					#pragma omp single
					{
							for(int i=0;i<=n_threads;i++){
									printf("B Val %d: %d\n", i, B[i]);
							}
					}
				#endif


				//SOTTOFASE 5_3: riempio il vettore B con la somma parziale delle occorrenze contenute in C
				#ifdef debug_stampa_iterazioni
					num=0;
				#endif
				#pragma omp barrier
				{
				    for(int i=(omp_get_thread_num()*((max-min+1)/n_threads));i<((omp_get_thread_num()+1)*((max-min+1)/n_threads));i++){
								B[omp_get_thread_num()+1]+= C[i];
								#ifdef debug_stampa_iterazioni
									num++;
								#endif
				    }
				}
				#ifdef debug_stampa_iterazioni
					printf("Thread: %d, Fase 5_3: iterazioni: %d su %d\n", omp_get_thread_num(),num, (max-min+1));
				#endif

				#ifdef debug
					#pragma omp single
					{
							for(int i=0;i<=n_threads;i++){
									printf("B Val %d: %d\n", i, B[i]);
							}
					}
				#endif


				//SOTTOFASE 5_4: riempio il vettore B con tutti gli indici di avvio di ogni singolo thread
				#ifdef debug_stampa_iterazioni
					num=0;
				#endif
				#pragma omp barrier
				{
					#pragma omp single
					{
							for(int i=0;i<n_threads-1;i++){
									B[i+1] = B[i] + B[i+1];
									#ifdef debug_stampa_iterazioni
										num++;
									#endif
							}
							B[n_threads]=n;
							#ifdef debug_stampa_iterazioni
								n++;
							#endif
					}
				}

				#ifdef debug_stampa_iterazioni
					printf("Thread: %d, Fase 5_4: iterazioni: %d su %d\n", omp_get_thread_num(),num, n_threads);
				#endif


				#ifdef debug
					#pragma omp single
					{
							for(int i=0;i<=n_threads;i++){
									printf("B Val %d: %d\n", i, B[i]);
							}
					}
				#endif

				//SOTTOFASE 5_5: riempio il vettore A a seconda delle occorrenze contenute in C
				#ifdef debug_stampa_iterazioni
					num=0;
				#endif
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
					#ifdef debug_stampa_iterazioni
						printf("Thread: %d, Fase 5_5: start_index: %d, stop_index: %d\n", omp_get_thread_num(), start, stop);
					#endif
					for(int i=start;i<stop;i++){
						  while (C[i]>0){
			            A[start_index]=i+start_value;
			            start_index++;
			            C[i]--;
									#ifdef debug_stampa_iterazioni
										num++;
									#endif
			        }
			    }
				}
				#ifdef debug_stampa_iterazioni
					printf("Thread: %d, Fase 5_5: iterazioni: %d su %d\n", omp_get_thread_num(),num, B[omp_get_thread_num()+1]-B[omp_get_thread_num()]);
				#endif

				#ifdef debug
					#pragma omp single
					{
							for(int i=0;i<n;i++){
									printf("A Val %d: %d\n", i, A[i]);
							}
					}
				#endif

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

		#ifdef Fase1
    for(int i=1; i<n; i++){
        if(A[i] > max){
            max = A[i];
        }
        if(A[i] < min){
            min = A[i];
        }
    }
		#endif

		#ifdef Fase2
    C = (int*)malloc((max-min+1)*sizeof(int));
		#endif

		#ifdef Fase3
    for(int i=0;i<(max-min+1);i++){

        C[i]=0;
    }
		#endif

		#ifdef Fase4
    for(int i=0;i<n;i++){

        C[A[i]-min]=C[A[i]-min]+1;
    }
		#endif

		#ifdef Fase5
    int k=0;

    for(int i=0;i<(max-min+1);i++){
        while (C[i]>0){
            A[k]=i+min;
            k++;
            C[i]--;
        }
    }
		#endif
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
			#ifdef sequential
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
			//
			#endif
		}

		if(type == 2){
			#ifdef parallel_test
			STARTTIME(1)
	    countingSortParallel(A,n_valori,atoi(argv[3]));
			STOPTIME(1)
			//printf("Parallel TEST:\n");
	    PRINTTIME(1)
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
			//
			#endif
		}

		free(A);

		return (EXIT_SUCCESS);
}
