#include <stdio.h>
#include <string.h>
#include "linear.h"

clock_t start;
clock_t end;

/* Read file */
//static
//void create_dataset(linear_param_t * params, data_t * dataset) {
//cl_device_id device;
//cl_context context;
//cl_program program;
//cl_command_queue queue;

extern void create_dataset(linear_param_t *params, data_t *dataset);
extern void temperature_regression(results_t *results);
extern void house_regression(results_t *results);


static void print_results(results_t *results)
{
    PRINT_RESULT("Parallelized", results->parallelized);
    PRINT_RESULT("Iterative", results->iterative);
}

static void write_results(results_t *results, const char *restricts)
{
    FILE *file = fopen(RESULT_FILENAME, restricts);
    WRITE_RESULT(file, results->parallelized);
    WRITE_RESULT(file, results->iterative);
    fclose(file);
}

int main(int argc, char *argv[])
{
    results_t results = {{0}};

    init_opencl();

    house_regression(&results);
    write_results(&results, "w");

    if (argc == 1 || strcmp(argv[1], "-no_print") > 0)
    {
        printf("\n> HOUSE REGRESSION (%d)\n\n", HOUSE_SIZE);
        print_results(&results);
    }

    temperature_regression(&results);
    write_results(&results, "a");

    if (argc == 1 || strcmp(argv[1], "-no_print") > 0)
    {
        printf("\n> TEMPERATURE REGRESSION (%d)\n\n", TEMP_SIZE);
        print_results(&results);
    }

    free_opencl();

    return 0;
}
