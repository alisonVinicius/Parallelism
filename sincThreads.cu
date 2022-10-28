/**
 * Aqui, neste programa, é executado um
 * algoritmo que força uma sincronização
 * da GPU. Esta sincronização irá manter
 * a impressão de cada thread ordenada
 * onde a cada rodada um número de thread
 * aumenta e diminui.
 * Exemplo:
 * t0
 * t0 t1
 * t0 t1 t2
 * t1 t2
 * t2
*/


/**
 * Biblioteca comum para C++
*/
#include<iostream>


/**
 * Bibliotecas para capturas de 
 * erro e execução de algumas
 * ferramentas cuda.
*/
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>



using namespace std;

/**
 * Este kernel executa a impressão
 * sincronizada das threads. */
__global__ void kernel(int n){

    /**
     * Declara variável compartilhada entre as threads.
     * Esta varíavel é restrita ao bloco das threads.
    */
    __shared__ int sinc;

    /**
     * Recolhe o índice da thread na dimensão x.
    */
    int i = threadIdx.x;

    /**
     * Declara variável local para thread.
    */
    int j = 0;

    /**
     * Para manter a sincronização, foi escolhida
     * a thread zero. Pode ser qualquer thread.
    */
    if(i == 0){

        /**
         * A thread 0 irá inicializar a variável
         * compartilhada sinc.
        */
        sinc = 0;

    }
    
    /**
     * Com esta barreira de sincronização, todas
     * as threads irão esperar a thread zero
     * terminar de inicializar sinc.
    */
    __syncthreads();





    /**
     * Como esta execução segue com uma aparência de onda,
     * no caso antidiagonais, é necessário ver de uma forma
     * matricial. Sendo uma matriz n x n, o total de 
     * rodas (chamadas de antidiagonais) é a soma das posições
     * de 1 linha com a soma das posições de uma coluna.
     * Então, se uma linha tem n colunas, e uma coluna tem
     * n linhas, então a execução terá 2*n rodadas.
     * Para ficar claro, desenhe uma matriz para vizualizar.
    */
    while(sinc <= 2*n){
   
        /**
         * Se o índice da thread i é menor ou igual a sinc
         * e j é menor do que n, então
        */
        if(i <= sinc && j < n){
            
            // executa o codigo

            /**
             * Faz uma impressão de j
            */
            printf("%d ", j);

            /**
             * Incrementa j
            */
            j++;

            
        }

        

        /**
         * Aqui novamente a thread zero
         * irá atualizar sinc.
        */
        if(i == 0){
          sinc++;
          printf("\n");
        }

        /**
         * barreira de sincronização
        */
        __syncthreads();

    }

  




}


int main(int argc, char *argv[]){

    /**
     * O número de blocos para este
     * algoritmo deve ser fixo em 1.
    */
    int numB = 1;

    /**
     * O número de threads é no máximo 1024.
    */
    int numT = 5;

    /**
     * Realiza a chamada de kernel.
    */
    kernel<<<numB,numT>>>(numT);
    cudaDeviceSynchronize();





    return 0;
}