#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util.h"



__global__ void decrypt_helper(char* message);
__global__ void decrypt(char *message);

void decrypt_helper(char* message){
  size_t index = threadIdx.x;

  // all translation statements:
  if (message[index] = '.- '){
    message[index] == 'a';
  }
  if (message[index] = '-... '){
    message[index] == 'b';
  }
  if (message[index] = '-.-. ' ){
    message[index] == 'c';
  }
  if (message[index] = '-.. ' ){
      message[index] == 'd';
  }
  if (message[index] = '. ' ){
      message[index] == 'e';
  }
  if (message[index] = '..-. ' ){
      message[index] == 'f';
  }
  if (message[index] = '--. '){
       message[index] == 'g';
  }
  if ( message[index] = '.... ' ){
     message[index] == 'h';
  }
  if (message[index] = '.. ' ){
      message[index] == 'i';
  }
  if (message[index] = '.--- ' ){
      message[index] == 'j';
  }
  if (message[index] = '-.- ' ){
      message[index] == 'k';
  }
  if (message[index] = '.-.. '){
      message[index] == 'l';
  }
  if (message[index] = '-- '){
      message[index] == 'm';
  }
  if (message[index] = '-. ' ){
      message[index] == 'n';
  }
  if (message[index] = '--- '){
       message[index] == 'o';
  }
  if ( message[index] = '.--. ' ){
     message[index] == 'p';
  }
  if ( message[index] = '--.- ' ){
     message[index] == 'q';
  }
  if (message[index] = '.-. ' ){
      message[index] == 'r';
  }
  if (message[index] = '... ' ){
     message[index] == 's';
  }
  if (message[index] = '- ' ){
     message[index] == 't';
  }
  if (message[index] = '..- ' ){
     message[index] == 'u';
  }
  if (message[index] = '...- '){
      message[index] == 'v';
  }
  if (message[index] = '.-- '){
      message[index] == 'w';
  }
  if (message[index] = '-..- ' ){
     message[index] == 'x';
  }
  if (message[index] = '-.-- ' ){
     message[index] == 'y';
  }
  if (message[index] = '--.. ' ){
     message[index] == 'z';
  }
  if (message[index] = '.---- '){
     message[index] = '1';
  }
  if (message[index] = '..--- '){
     message[index] = '2';
  }
  if (message[index] = '...-- '){
     message[index] = '3';
  }
  if (message[index] = '....- '){
     message[index] = '4';
  }
  if (message[index] = '..... '){
     message[index] = '5';
  }
  if (message[index] = '-.... '){
     message[index] = '6';
  }
  if (message[index] = '--... '){
     message[index] = '7';
  }
  if (message[index] = '---.. '){
     message[index] = '8';
  }
  if (message[index] = '----. '){
     message[index] = '9';
  }
  if (message[index] = '----- '){
     message[index] = '10';
  }


}



void decrypt(char *message){
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
  decrypt_helper<<<1, message_size>>>(message);

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
   decrypt(message);
   printf("%s\n", message);


}