#include<iostream>
using namespace std;

int main(void){
    cout << "tudo ok até aqui!" << endl;
    cout << "teste " << endl;
    return 0;
}

__global__ void teste(){

    printf("oi %d\n",threadIdx.x);
}

