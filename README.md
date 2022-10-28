# Parallelism

O programa sincThreads.cu realiza a sincronização de threads de 
acordo com antidiagonais. Fica mais compreensível desenhando uma
matriz e olhando as antidiagonais, sendo a execução semelhante a
ondas percorrendo a matriz.

  1        2        3        4        5        6
0 0 0    1 0 0    1 2 0    1 2 3    1 2 3    1 2 3
0 0 0 -> 0 0 0 -> 1 0 0 -> 1 2 0 -> 1 2 3 -> 1 2 3
0 0 0    0 0 0    0 0 0    1 0 0    1 2 0    1 2 3