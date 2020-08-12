#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>


void winograd_GgGt_2x2(float *input, float *output, int K, int C) {
    int total_filter = K * C;
    int in_c_stride = 9, in_k_stride = in_c_stride * C;
    int out_c_stride = 16, out_k_stride = out_c_stride * C;

    #pragma omp parallel for
    for (int global_id = 0; global_id < total_filter; global_id ++) {
        int k = global_id / C;
        int c = global_id % C;

        float tile[3][3], t_tile[4][3], f_tile[4][4];
        for (int i = 0; i < 3; i ++) {
            for (int j = 0; j < 3; j ++) {
                tile[i][j] = input[in_k_stride * k + in_c_stride * c + 3 * i + j];
            }
        }

        // G * g
        for (int j = 0; j < 3; j ++) {
            t_tile[0][j] = tile[0][j];
            t_tile[1][j] = 0.5 * tile[0][j] + 0.5 * tile[1][j] + 0.5 * tile[2][j];
            t_tile[2][j] = 0.5 * tile[0][j] - 0.5 * tile[1][j] + 0.5 * tile[2][j];
            t_tile[3][j] = tile[2][j];
        }
        // g * Gt
        for (int i = 0; i < 4; i ++) {
            f_tile[i][0] = t_tile[i][0];
            f_tile[i][1] = 0.5 * t_tile[i][0] + 0.5 * t_tile[i][1] + 0.5 * t_tile[i][2];
            f_tile[i][2] = 0.5 * t_tile[i][0] - 0.5 * t_tile[i][1] + 0.5 * t_tile[i][2];
            f_tile[i][3] = tile[i][2];
        }

        for (int i = 0; i < 4; i ++) {
            for (int j = 0; j < 4; j ++) {
                output[out_k_stride * k + out_c_stride * c + 4 * i + j] = f_tile[i][j];
            }
        }
    }
}


void winograd_BtdB_2x2(float *input, float *output, 
int batch_size, int C, int tile_n, int map_size) {

    int total_tile = batch_size * C * tile_n * tile_n;
    int in_n_stride = map_size * map_size * C, in_c_stride = map_size * map_size, x_stride = map_size, y_stride = 1;
    int out_n_stride = tile_n * tile_n * 16 * C, out_c_stride = tile_n * tile_n * 16;
    int tilei_stride = tile_n * 16, tilej_stride = 16; 

    #pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id ++) {
        int n = global_id / (C * tile_n * tile_n);
        int remain = global_id % (C * tile_n * tile_n);
        int c = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[4][4], t_tile[4][4];
        for (int i = 0; i < 4; i ++) {
            for (int j = 0; j < 4; j ++) {
                int x = 2 * tile_i + i;
                int y = 2 * tile_j + j;
                if (x >= map_size || y >= map_size) {
                    tile[i][j] = 0;
                    continue;
                }
                tile[i][j] = input[n * in_n_stride + c * in_c_stride + x * x_stride + y * y_stride];
            }
        }

        // Bt * d
        for (int j = 0; j < 4; j ++) {
            t_tile[0][j] = tile[0][j] - tile[2][j];
            t_tile[1][j] = tile[1][j] + tile[2][j];
            t_tile[2][j] = -tile[1][j] + tile[2][j];
            t_tile[3][j] = tile[1][j] - tile[3][j];
        }
        // d * B
        for (int i = 0; i < 4; i ++) {
            tile[i][0] = t_tile[i][0] - t_tile[i][2];
            tile[i][1] = t_tile[i][1] + t_tile[i][2];
            tile[i][2] = -t_tile[i][1] + t_tile[i][2];
            tile[i][3] = t_tile[i][1] - t_tile[i][3];
        }

        for (int i = 0; i < 4; i ++) {
            for (int j = 0; j < 4; j ++) {
                output[n * out_n_stride + c * out_c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 4 * i + j] = tile[i][j];
            }
        }
    }
}


void winograd_BtdB_padding_2x2(float *input, float *output, 
int batch_size, int C, int tile_n, int map_size) {

    int total_tile = batch_size * C * tile_n * tile_n;
    int in_n_stride = map_size * map_size * C, in_c_stride = map_size * map_size, x_stride = map_size, y_stride = 1;
    int out_n_stride = tile_n * tile_n * 16 * C, out_c_stride = tile_n * tile_n * 16;
    int tilei_stride = tile_n * 16, tilej_stride = 16; 

    #pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id ++) {
        int n = global_id / (C * tile_n * tile_n);
        int remain = global_id % (C * tile_n * tile_n);
        int c = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[4][4], t_tile[4][4];
        for (int i = 0; i < 4; i ++) {
            for (int j = 0; j < 4; j ++) {
                int x = 2 * tile_i + i;
                int y = 2 * tile_j + j;
                if (x == 0 || y == 0 || x >= (map_size + 1) || y >= (map_size + 1)) {
                    tile[i][j] = 0;
                }
                else {
                    tile[i][j] = input[n * in_n_stride + c * in_c_stride + (x - 1) * x_stride + (y - 1) * y_stride];
                }
            }
        }

        // Bt * d
        for (int j = 0; j < 4; j ++) {
            t_tile[0][j] = tile[0][j] - tile[2][j];
            t_tile[1][j] = tile[1][j] + tile[2][j];
            t_tile[2][j] = -tile[1][j] + tile[2][j];
            t_tile[3][j] = tile[1][j] - tile[3][j];
        }
        // d * B
        for (int i = 0; i < 4; i ++) {
            tile[i][0] = t_tile[i][0] - t_tile[i][2];
            tile[i][1] = t_tile[i][1] + t_tile[i][2];
            tile[i][2] = -t_tile[i][1] + t_tile[i][2];
            tile[i][3] = t_tile[i][1] - t_tile[i][3];
        }

        for (int i = 0; i < 4; i ++) {
            for (int j = 0; j < 4; j ++) {
                output[n * out_n_stride + c * out_c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 4 * i + j] = tile[i][j];
            }
        }
    }
}


void winograd_outerProduct_AtIA_2x2(float *input, float *weight, float *bias, float *output, 
int batch_size, int K, int tile_n, int out_map_size, int C) {

    int total_tile = batch_size * K * tile_n * tile_n;
    int c_stride = tile_n * tile_n * 16, in_n_stride = C * c_stride;
    int tilei_stride = tile_n * 16, tilej_stride = 16;
    int w_c_stride = 16, w_k_stride = C * 16;
    int out_k_stride = out_map_size * out_map_size, out_n_stride = out_k_stride * K;
    int x_stride = out_map_size, y_stride = 1;

    #pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id ++) {
        int n = global_id / (K * tile_n * tile_n);
        int remain = global_id % (K * tile_n * tile_n);
        int k = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[4][4] = {0};
        for (int c = 0; c < C; c ++) {
            for (int i = 0; i < 4; i ++) {
                for (int j = 0; j < 4; j ++) {
                    tile[i][j] += input[n * in_n_stride + c * c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 4 * i + j] 
                                    * weight[k * w_k_stride + c * w_c_stride + 4 * i + j];
                }
            }   
        }

        float t_tile[2][4], f_tile[2][2];
        // At * I
        for (int j = 0; j < 4; j ++) {
            t_tile[0][j] = tile[0][j] + tile[1][j] + tile[2][j];
            t_tile[1][j] = tile[1][j] - tile[2][j] - tile[3][j];
        }
        // I * A
        for (int i = 0; i < 2; i ++) {
            f_tile[i][0] = t_tile[i][0] + t_tile[i][1] + t_tile[i][2];
            f_tile[i][1] = t_tile[i][1] - t_tile[i][2] - t_tile[i][3];
        }
        // bias
        for (int i = 0; i < 2; i ++) {
            for (int j = 0; j < 2; j ++) {
                f_tile[i][j] += bias[k];
            }
        }
        for (int i = 0; i < 2; i ++) {
            for (int j = 0; j < 2; j ++) {
                int x = 2 * tile_i + i;
                int y = 2 * tile_j + j;
                if (x >= out_map_size || y >= out_map_size) {
                    continue;
                }
                output[n * out_n_stride + k * out_k_stride + x * x_stride + y * y_stride] = f_tile[i][j];
            }
        }
    }
}


void winograd_convolution_2x2
(float *input,    /* NxCxHxW */ 
float *weight,    /* KxCx3x3 */ 
float *bias,      /* K */
float *my_res,    /* NxKxH'xW'*/  
int batch_size, int C, int K, int map_size, int padding) {

    // filter transformation
    float *trans_filter = (float *)malloc(K * C * 16 * sizeof(float));  // transformed filters
    if (trans_filter == NULL) {
        printf("bad malloc trans_filter\n");
    }
    winograd_GgGt_2x2(weight, trans_filter, K, C);

    int out_map_size = (map_size + padding * 2) - 2;  // kernel size = 3, stride = 1 in Winograd algorithm
    int tile_n = (out_map_size + 1) / 2;

    float *trans_input = (float *)malloc(batch_size * tile_n * tile_n * C * 16 * sizeof(float));  // transformed input
    if (trans_input == NULL) {
        printf("bad malloc trans_input\n");
    }

    // input transformation
    if (padding == 0) {
        winograd_BtdB_2x2(input, trans_input, batch_size, C, tile_n, map_size);
    }
    else if (padding == 1) {
        winograd_BtdB_padding_2x2(input, trans_input, batch_size, C, tile_n, map_size);
    }
    
    // element-wise multiplication & output transformation
    winograd_outerProduct_AtIA_2x2(trans_input, trans_filter, bias, my_res, batch_size, K, tile_n, out_map_size, C);

    free(trans_input);
    free(trans_filter);
    return;
}