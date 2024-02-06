#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>

#define DATA int

#define START \
  cudaEvent_t start,stop;\
  cudaEventCreate(&start);\
  cudaEventCreate(&stop);\
  cudaEventRecord(start,0);\

#define STOP \
  cudaEventRecord(stop,0);\
  cudaEventSynchronize(stop);\
  float elapsed;\
  cudaEventElapsedTime(&elapsed,start,stop);\
  elapsed/=1000.f;\
  cudaEventDestroy(start);\
  cudaEventDestroy(stop);\
  printf("Kernel elapsed time %fs \n", elapsed);

#define START_GF \
  cudaEventCreate(&start_gf);\
  cudaEventCreate(&stop_gf);\
  cudaEventRecord(start_gf,0);\

#define STOP_GF \
  cudaEventRecord(stop_gf,0);\
  cudaEventSynchronize(stop_gf);\
  cudaEventElapsedTime(&gf_elapsed,start_gf,stop_gf);\
  gf_elapsed/=1000.f;\
  cudaEventDestroy(start_gf);\
  cudaEventDestroy(stop_gf);\
  double dNumOps = (2 * N);\
  printf("Num ope: %f\n", dNumOps);\
  printf("Gflop time: %f\n", gf_elapsed);\
  gflops = 1.0e-9 * dNumOps/gf_elapsed;\
  printf("Gflops: %f\n", gflops);\

void make_csv(float time, int N, int gridsize, int thread_per_block){
  FILE* fp;
  char root_filename[] = "CS_texture";

  char* filename = (char*) malloc(sizeof(char) * (strlen(root_filename) + 10*sizeof(char)));
  sprintf(filename,"%s_CountingSort_%d_v_%d_b_%d_tpb.csv",root_filename,N,gridsize,thread_per_block);
  
  if ( access( filename, F_OK ) == 0 ) {
     fp = fopen(filename,"a"); 

  } else {
     fp = fopen(filename,"w");
     fprintf(fp, "N; BlockSize; GridSize; time_sec\n");
  }
  fprintf(fp, "%d; %d; %d; %f\n", N, thread_per_block, gridsize, time);
  fclose(fp);
}


texture<DATA,1> text_mem;

__global__ void occurrence(DATA *C,int N,DATA val){
  //Calcolo l'index di base dove effettuare la somma se per ogni thread
  //sono presenti più celle di memoria da calcolare
  int idx_basic = blockIdx.x + threadIdx.x * gridDim.x;
  
  //Creo una variabile contatore
  int occ_counter = 0;
  
  //Creo un index per l'iterazione
  int idx_iter = idx_basic;

  //Calcolo tutte le occorrenze
  while (idx_iter < N) {
    if(idx_iter < N){
      if(tex1Dfetch(text_mem,idx_iter) == val){
        occ_counter += 1;
      }
    }
    idx_iter += blockDim.x * gridDim.x;
  }


  //Le associo al vettore in memoria
  C[idx_basic] = occ_counter;

  //Sincronizzo i thread
  __syncthreads();

  //Eseguo la somma in parallelo
  int i = blockDim.x/2;
  while(i != 0){
      if(idx_basic < i*gridDim.x){
          C[idx_basic] = C[idx_basic]+C[idx_basic+(i*gridDim.x)];
      }
      __syncthreads();
      i = i/2;
  }
}

int main(int argc,char ** argv){
  if(argc<4) {
    fprintf(stderr,"ERROR too few arguments of: %s\n",argv[0]);
    exit(1);
  }

  DATA *A,*C,*devA,*devC;
  DATA max,min;

  int N = atoi(argv[1]);

  //Assegno il numero dei blocchi
  int blocks = atoi(argv[2]);
  dim3 dimGrid(blocks);

  //Assegno il numero dei thread per blocco
  int th_p_block = atoi(argv[3]);
  dim3 dimBlock(th_p_block);

  //alloco il vettore da ordinare su host e device
  A = (DATA*) malloc(N*sizeof(DATA));
  cudaMalloc((void**)&devA,N*sizeof(DATA));

  //inizializzo il vettore
  for(int i=0;i<N;i++){
      A[i] = (N-i)%100;
  }

  //calcolo il massimo e il minimo
  max = A[0];
  min = A[0];
  for(int i=1;i<N;i++){
      if(A[i] > max){
          max = A[i];
      }else if(A[i] < min){
          min = A[i];
      }
  }

  //alloco il vettore di supporto su host e device
  C = (DATA*) malloc(blocks*sizeof(DATA));
  cudaMalloc((void**)&devC,N*sizeof(DATA));

  //trasferisco i dati sul device
  cudaMemcpy(devA,A,N*sizeof(DATA),cudaMemcpyHostToDevice);
  //cudaDeviceSynchronize();
  cudaChannelFormatDesc channel = cudaCreateChannelDesc<DATA>();
  cudaBindTexture(0,text_mem,devA,channel);

  int occorrenze = 0;
  int index_sorted = 0;
  START
  for(int i_occurr = min; i_occurr <= max; i_occurr++){
      //Per ogni elemento che va da min a max 
      //vado a cercarlo tramite chiamata a kernel
      occurrence<<<dimGrid,dimBlock>>>(devC,N,i_occurr);
      cudaDeviceSynchronize();
      cudaMemcpy(C,devC,blocks*sizeof(DATA),cudaMemcpyDeviceToHost);

      //Calcolo le occorrenze restituite dal kernel
      for(int i=0;i<blocks;i++){
        occorrenze += C[i];
        for(int j=0;j<C[i];j++){
          A[index_sorted] = i_occurr;
          index_sorted++;
        }
      }
  }
  STOP
  cudaUnbindTexture(text_mem);

  make_csv(elapsed, N, blocks, th_p_block);

  int ordinato = 1;
  for(int i=0;i<N;i++){
      if(i < N-2){
        if(A[i] > A[i+1]){
          ordinato = 0;
        }
      }
  }
  printf("\noccorrenze trovate: %d, occorrenze calcolate: %d\n",index_sorted,occorrenze);
  printf("ordinato = %d\n",ordinato);

  //libero la memoria
  free(A);
  free(C);
  cudaFree(devA);
  cudaFree(devC);
  
  return 0;
}