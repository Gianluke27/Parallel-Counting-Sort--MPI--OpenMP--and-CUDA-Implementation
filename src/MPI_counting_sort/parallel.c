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
*/

/**
  @file main.c
  @brief Parallel counting sort
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define STARTTIME(id)                           \
  clock_t start_time_42_##id, end_time_42_##id; \
  start_time_42_##id = clock();

#define STOPTIME(id)        \
  end_time_42_##id = clock(); \
  double time##id = ((double)(end_time_42_##id - start_time_42_##id)) / CLOCKS_PER_SEC;

#define PRINTTIME(id)\
  printf("%lf;",time##id);

//#define DEBUG
/*
#ifdef DEBUG
  if(rank == 0){
    start_time_1 = clock();
  }
#endif

#ifdef DEBUG
  if(rank == 0){
    printf("Time fase scatter: %lf\n",((double)(clock() - start_time_2)) / CLOCKS_PER_SEC);
  }
#endif
*/

int main(int argc, char* argv[]){
  int start_time_1;
  int start_time_2;
  int start_time_3;
  int start_time_4;
  int start_time_5;
  int start_time_6;
  int start_time_7;
  int start_time_8;
  int start_time_9;

  int *A = NULL;
  int *A_final = NULL;
  int *subA = NULL;
  int n_valori;
  int n_per_proc;
  int n_effective;
  int globalmax;
  int globalmin;

  int rank, size;

  srand(time(NULL));
  int max_value = 256;//rand()%1000;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  n_valori = atoi(argv[1]);

  //rank 0 initialize the array
  if(rank == 0){
    A = (int*)malloc((n_valori)*sizeof(int));
    A_final = (int*)malloc((n_valori)*sizeof(int));

    for(int i=0;i<n_valori;i++){
       A[i]=i%max_value;
       //A[i]=abs(rand())%(max_value);
       //printf("A[%d]=%d\n",i,A[i]);
    }
  }

  //allocation of a sub array for each rank
  n_per_proc = ceil((float)n_valori/size);
  subA = (int*)malloc(n_per_proc*sizeof(int));

  //Start time
  //STARTTIME(2)

  #ifndef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      start_time_1 = clock();
    }
  #endif

  #ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      start_time_1 = clock();
    }
  #endif

  //Fase 1: Max and Min of array
  //Each rank has a subarray
  MPI_Scatter(A, n_per_proc, MPI_INT, subA, n_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

  #ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      printf("Time Fase 1 calcolo scatter: %lf\n",((double)(clock() - start_time_1)) / CLOCKS_PER_SEC);
    }
  #endif

  #ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      start_time_1 = clock();
    }
  #endif

  if(rank == size-1){
    n_effective = n_valori - (size-1)*ceil((float)n_valori/size);
  }else{
    n_effective = n_per_proc;
  }

  int localmax = subA[0];
  int localmin = subA[0];
  for(int i=1; i<n_effective; i++){
    if(subA[i] > localmax){
      localmax = subA[i];
    }
    else if(subA[i] < localmin){
      localmin = subA[i];
    }
  }

  //printf("Local of rank %d -> min %d, max %d\n", rank, localmin, localmax);

  #ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      printf("Time Fase 1 calcolo max: %lf\n",((double)(clock() - start_time_1)) / CLOCKS_PER_SEC);
    }
  #endif

  #ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      start_time_9 = clock();
    }
  #endif

  MPI_Allreduce(&localmin, &globalmin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&localmax, &globalmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  #ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      printf("Time Fase 1 All_reduce: %lf\n",((double)(clock() - start_time_9)) / CLOCKS_PER_SEC);
    }
  #endif

  #ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      start_time_2 = clock();
    }
  #endif
  //printf("Global of rank %d -> min %d, max %d\n", rank, globalmin, globalmax);

  //FASE 2

  int *localC = (int*)malloc((globalmax-globalmin+1)*sizeof(int));
  int *globalC = (int*)malloc((globalmax-globalmin+1)*sizeof(int));

  //FASE 3
  for(int i=0; i<globalmax-globalmin+1; i++){
    localC[i] = 0;
  }

  //FASE 4
  for(int i=0; i<n_effective; i++){
    localC[subA[i]-globalmin]=localC[subA[i]-globalmin]+1;
  }

  #ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      printf("Time Fase 2-3-4 calcolo occorrenze: %lf\n",((double)(clock() - start_time_2)) / CLOCKS_PER_SEC);
    }
  #endif

  #ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      start_time_3 = clock();
    }
  #endif
  /*for(int i=0; i<globalmax-globalmin+1; i++){
    printf("Rank %d, C[%d]=%d\n", rank, i, localC[i]);
  }*/

  MPI_Allreduce(localC, globalC, globalmax-globalmin+1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  //TEST
  //FASE 5
  //Primo passo:
  //Invio spezzettato il vettore C (vettore delle occorrenze)
  int *subC;
  int n_per_proc_C = ceil((float)(globalmax-globalmin+1)/size);
  subC = (int*)malloc(n_per_proc_C*sizeof(int));

  MPI_Scatter(globalC, n_per_proc_C, MPI_INT, subC, n_per_proc_C, MPI_INT, 0, MPI_COMM_WORLD);


  //Secondo passo:
  //Mi calcolo il numero effettivo degli elementi.
  //I primi n-1 thread hanno dimensione fissa, l'ultimo invece no
  if(rank == size-1){
    n_effective = globalmax-globalmin+1 - (size-1)*n_per_proc_C;
  }else{
    n_effective = n_per_proc_C;
  }

  //Terzo passo:
  //Calcolo il numero delle subOccorrenze per ogni subArray C

  //printf("rank: %d, val effettivi: %d, start:%d\n", rank, n_effective, (n_per_proc_C * rank) + globalmin);
  int valSubArray = 0;
  for(int i=0; i<n_effective; i++){
    for(int j=0; j<subC[i]; j++){
      valSubArray++;
    }
  }

  //Quarto passo:
  //Istanzio il subArray che contiene tutti i subvalori ordinati
  //Inserisco all'interno i valori
  int *subArray = (int*)malloc(valSubArray*sizeof(int));
  int x = 0;
  for(int i=0; i<n_effective; i++){
    for(int j=0; j<subC[i]; j++){
      subArray[x] = (n_per_proc_C * rank) + globalmin + i;
      x++;
    }
  }

  //Quinto passo:
  //Invio alla root il numero degli elementi per ogni thread
  //in modo tale da calcolare il displacement
  //Faccio una gather per ottenere il displacement
  int *recvcounts = NULL;

  if(rank == 0){
    start_time_8 = clock();
  }

  if (rank == 0){
    recvcounts = malloc(size*sizeof(int));
  }

  MPI_Gather(&valSubArray, 1, MPI_INT,
              recvcounts, 1, MPI_INT,
              0, MPI_COMM_WORLD);

  //Sesto passo:
  //Calcolo il displacement
  int totlen = 0;
  int *displs = NULL;

  if (rank == 0) {
      displs = malloc(size*sizeof(int));

      displs[0] = 0;

      for (int i=1; i<size; i++) {
         displs[i] = displs[i-1] + recvcounts[i-1];
      }
  }

  //Settimo e ultimo passo:
  //Invio alla root tutti i subArray contenenti i valori ordinati
  MPI_Gatherv(subArray, valSubArray, MPI_INT,
              A_final, recvcounts, displs, MPI_INT,
              0, MPI_COMM_WORLD);

  #ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      printf("Time Fase 5 ordinamento: %lf\n",((double)(clock() - start_time_3)) / CLOCKS_PER_SEC);
    }
  #endif

  #ifndef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
      printf("%lf;",((double)(clock() - start_time_1)) / CLOCKS_PER_SEC);
    }
  #endif

  //Stampa dei valori
  /*if (rank == 0) {
    printf("VALORI FINALI\n");
    for(int i=0; i<n_valori; i++){
      printf("%d\n", A_final[i]);
    }
  }*/

  /*if(rank == 0){
    STOPTIME(2)
    PRINTTIME(2)
  }*/

  //Free di tutti i puntatori
  if (rank == 0) {
    free(A);
    free(A_final);
    free(recvcounts);
    free(displs);
  }
  free(subA);
  free(localC);
  free(globalC);
  free(subC);
  free(subArray);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  
  //printf("Sequenziale:\n");
  exit(EXIT_SUCCESS);
}
