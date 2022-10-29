#include<iostream>



using namespace std;

__global__ void kernel(int *d_v, int n){



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
            printf("%d ", d_v[i*n + j]);




            // t = (s[i] != r[j] ? 1 : 0);
			// a = d[i][j-1] + 1;
			// b = d[i-1][j] + 1;
			// c = d[i-1][j-1] + t;
			// // Calcula d[i][j] = min(a, b, c)
			// if (a < b)
			// 	min = a;
			// else
			// 	min = b;
			// if (c < min)
			// 	min = c;
			// d[i][j] = min;



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




int main(int argc, char **argv){

    int  *v = (int*)malloc(5 * 5 * sizeof(int));

    int *d_v;

    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            v[i*5 + j] = i + j;
        }
    }


    cudaMalloc((void**)&d_v, 5 * 5 * sizeof(int));


    cudaMemcpy(d_v, v, 5 * 5 * sizeof(int), cudaMemcpyHostToDevice);


    int n = 5;

    kernel<<<1,5>>>(d_v, n);
    cudaDeviceSynchronize();


    a = 5;
    b = 6;
    c = soma(a, b);


    a 

    
    soma(int a, int b){

        a = 12;

        return a + b;
    }

    return 0;
}