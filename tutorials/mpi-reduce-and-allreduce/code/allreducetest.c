#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
float *create_rand_nums(int num_elements) {
  float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (float)RAND_MAX);
  }
  return rand_nums;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: avg num_elements_per_proc\n");
    exit(1);
  }

  clock_t start, end;

  int num_elements_per_proc = atoi(argv[1]);

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Comm cross_comm;
  MPI_Comm_split(MPI_COMM_WORLD, world_rank%8, world_rank, &cross_comm);

  // Create a random array of elements on all processes.
  srand(time(NULL)*world_rank); // Seed the random number generator of processes uniquely
  float *rand_nums = NULL;
  rand_nums = create_rand_nums(num_elements_per_proc);

  // Reduce all of the local sums into the global sum in order to
  // calculate the mean
  float* global_sum = (float*)malloc(num_elements_per_proc*sizeof(float));

  MPI_Barrier(MPI_COMM_WORLD);

  start = clock();
  MPI_Allreduce(rand_nums, global_sum, num_elements_per_proc, MPI_FLOAT, MPI_SUM,
                cross_comm);
  end = clock();

  // Clean up
  free(rand_nums);
  free(global_sum);

  if (world_rank < 1) {
    printf("%f\n", ((double)(end-start))/CLOCKS_PER_SEC);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
