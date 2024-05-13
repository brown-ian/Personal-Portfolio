
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util.h"



__global__ void encrypt_helper(char* message);
__global__ void encrypt(char *message);

void encrypt_helper(char* message){
  size_t index = threadIdx.x;

  // all translation statements:
  if (message[index] == 'A' || message[index] == 'a'){
   message[index] = '.- ';
  }
  if (message[index] == 'B' || message[index] == 'b'){
    message[index] = '-... ';
  }
  if (message[index] == 'C' || message[index] == 'c'){
    message[index] = '-.-. ';
  }
  if (message[index] == 'D' || message[index] == 'd'){
      message[index] = '-.. ';
  }
  if (message[index] == 'E' || message[index] == 'e'){
      message[index] = '. ';
  }
  if (message[index] == 'F' || message[index] == 'f'){
      message[index] = '..-. ';
  }
  if (message[index] == 'G' || message[index] == 'g'){
      message[index] = '--. ';
  }
  if (message[index] == 'H' || message[index] == 'h'){
      message[index] = '.... ';
  }
  if (message[index] == 'I' || message[index] == 'i'){
      message[index] = '.. ';
  }
  if (message[index] == 'J' || message[index] == 'j'){
      message[index] = '.--- ';
  }
  if (message[index] == 'K' || message[index] == 'k'){
      message[index] = '-.- ';
  }
  if (message[index] == 'L' || message[index] == 'l'){
      message[index] = '.-.. ';
  }
  if (message[index] == 'M' || message[index] == 'm'){
      message[index] = '-- ';
  }
  if (message[index] == 'N' || message[index] == 'n'){
      message[index] = '-. ';
  }
  if (message[index] == 'O' || message[index] == 'o'){
      message[index] = '--- ';
  }
  if (message[index] == 'P' || message[index] == 'p'){
      message[index] = '.--. ';
  }
  if (message[index] == 'Q' || message[index] == 'q'){
      message[index] = '--.- ';
  }
  if (message[index] == 'R' || message[index] == 'r'){
      message[index] = '.-. ';
  }
  if (message[index] == 'S' || message[index] == 's'){
     message[index] = '... ';
  }
  if (message[index] == 'T' || message[index] == 't'){
     message[index] = '- ';
  }
  if (message[index] == 'U' || message[index] == 'u'){
     message[index] = '..- ';
  }
  if (message[index] == 'V' || message[index] == 'v'){
     message[index] = '...- ';
  }
  if (message[index] == 'W' || message[index] == 'w'){
     message[index] = '.-- ';
  }
  if (message[index] == 'X' || message[index] == 'x'){
     message[index] = '-..- ';
  }
  if (message[index] == 'Y' || message[index] == 'y'){
     message[index] = '-.-- ';
  }
  if (message[index] == 'Z' || message[index] == 'z'){
     message[index] = '--.. ';
  }
  if (message[index] == '1'){
     message[index] = '.---- ';
  }
  if (message[index] == '2'){
     message[index] = '..--- ';
  }
  if (message[index] == '3'){
     message[index] = '...-- ';
  }
  if (message[index] == '4'){
     message[index] = '....- ';
  }
  if (message[index] == '5'){
     message[index] = '..... ';
  }
  if (message[index] == '6'){
     message[index] = '-.... ';
  }
  if (message[index] == '7'){
     message[index] = '--... ';
  }
  if (message[index] == '8'){
     message[index] = '---.. ';
  }
  if (message[index] == '9'){
     message[index] = '----. ';
  }
  if (message[index] == '0'){
     message[index] = '----- ';
  }


}



void encrypt(char *message){
// Allocate arrays for X and Y on the CPU. This memory is only usable on the CPU
  int message_size = strlen(message);

  char* cpu_x = (char*)malloc(sizeof(char) * message_size);
  //float* cpu_y = (float*)malloc(sizeof(float) * N);

  cpu_x = message;

  // The gpu_x and gpu_y pointers will only be usable on the GPU (which uses separate memory)
  char* gpu_x;

  // Allocate space for the x array on the GPU
  if(cudaMalloc(&gpu_x, sizeof(char) * message_size) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate message array on GPU\n");
    exit(2);
  }

  // Allocate space for the y array on the GPU

  // Copy the cpu's x array to the gpu with cudaMemcpy
  if(cudaMemcpy(gpu_x, cpu_x, sizeof(char) * message_size, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy X to the GPU\n");
  }

  // Copy the cpu's y array to the gpu with cudaMemcpy

  // Calculate the number of blocks to run, rounding up to include all threads

  // Run the decrypt helper kernel
  encrypt_helper<<<1, message_size>>>(message);j

  // Wait for the kernel to finish
  if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

  // Copy the y array back from the gpu to the cpu
  if(cudaMemcpy(cpu_x, gpu_x, sizeof(char) * message_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy message from the GPU\n");
  }

  // Print the updated y array
  printf("%s ", cpu_x);

  cudaFree(gpu_x);
  cudaFree(gpu_y);
  free(cpu_x);
  free(cpu_y);

  return 0;
}

int main(){
   char* message = "Hello World";
   printf("%s\n", message);
   encrypt(message);
   printf("%s\n", message);


}