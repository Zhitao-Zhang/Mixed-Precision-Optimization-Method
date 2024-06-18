#include <stdio.h>
#pragma argsused
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include "outputfunc.h"

int divUp(int a, int b)
{
    return (a - 1) / b + 1;
}

float **space2d(int nr, int nc)
{
    float **a;
    int i;
    a = (float **)calloc(nr, sizeof(float *));
    for (i = 0; i < nr; i++)
        a[i] = (float *)calloc(nc, sizeof(float));

    return a;
}

// 閲婃斁浜岀淮鍔ㄦ�佹暟缁
void free_space2d(float **a, int nr)
{
    int i;
    for (i = 0; i < nr; i++)
        free(a[i]);
    free(a);
}

// 灏嗕簩杩涘埗鏁版嵁鍐欏叆鏂囦欢
void wfile(char filename[], float **data, int nr, int nc)
{
    int i, j;
    FILE *fp = fopen(filename, "wt");
    for (i = 0; i < nr; i++)
    {
        for (j = 0; j < nc; j++)
        {
            fprintf(fp, "%e ", data[i][j]);
            if ((j + 1) % nc == 0)
                fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

// 鐢宠涓夌淮鍔ㄦ�佹暟缁
float ***space3d(int nr, int ny, int nc)
{
    float ***a;
    int i, j, k;
    a = (float ***)malloc(sizeof(float **) * nr);
    for (i = 0; i < nr; i++)
    {
        a[i] = (float **)malloc(sizeof(float *) * ny);
    }
    for (i = 0; i < nr; i++)
    {
        for (j = 0; j < ny; j++)
        {
            a[i][j] = (float *)malloc(sizeof(float) * nc); // sizeof(float) nc鏀逛负sizeof(float)*nc
        }
    }
    for (i = 0; i < nr; i++)
        for (j = 0; j < ny; j++)
            for (k = 0; k < nc; k++)
            {
                a[i][j][k] = 0.0f;
            }
    return a;
}

// 閲婃斁涓夌淮鍔ㄦ�佹暟缁
void free_space3d(float ***a, int nr, int ny)
{
    int i, j;
    for (i = 0; i < nr; i++)
    {
        for (j = 0; j < ny; j++)
        {
            free(a[i][j]);
        }
    }
    for (i = 0; i < nr; i++)
    {
        free(a[i]);
    }
    free(a);
}

// 灏嗕簩杩涘埗鏁版嵁鍐欏叆鏂囦欢鈥斾笁缁
void wfile3d(char filename[], float ***data, int nr, int ny, int nc)
{
    int i, j, k;
    FILE *fp = fopen(filename, "wt");
    /*
    for (int i = 0; i<nr; i++)
    {
    fwrite(data[i], sizeof(float), nc, fp);
    }
    */
    {
        for (i = 0; i < nr; i++)
        {
            for (k = 0; k < nc; k++)
            {
                j = ny / 2; // y杞村垏鐗
                // j = ny - 1;//y杞村垏鐗
                fprintf(fp, "%e ", data[i][j][k]);
                if ((k + 1) % nc == 0)
                    fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }
    }
    //          fwrite(&data[i][j],1,sizeof(float),fp);
    fclose(fp);
}

void wfile1d(char filename[], float *data, int nr, int ny, int nc)
{
    int i, j, k;
    FILE *fp = fopen(filename, "wt");
    /*
    for (int i = 0; i<nr; i++)
    {
    fwrite(data[i], sizeof(float), nc, fp);
    }
    */
    {
        for (i = 0; i < nr; i++)
        {
            for (k = 0; k < nc; k++)
            {
                j = ny / 2; // y杞村垏鐗
                // j = ny - 1;//y杞村垏鐗
                fprintf(fp, "%e ", data[i * ny * nc + j * nc + k]);
                if ((k + 1) % nc == 0)
                    fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }
    }
    //          fwrite(&data[i][j],1,sizeof(float),fp);
    fclose(fp);
}

void create_model_all(float ***vp, float ***vs, float ***rhos, float ***vf, float ***vfg, float ***rho, float ***rhof, float ***M, float ***C, float ***C1,
                      float ***C2, float ***HH, float ***H2u, float ***mu, int nr, int ny, int nc)
{
    // 这里进行修改，最好是可视化设计或从文件读入
    int ix, iy, iz;
    double Ks1, Kb1, Kf1, a1, D1, tao, eta, porousm, perm, por, Kw, Kg;
    // float saturationGas = 0.8; // 含气饱和度
    // float densityWater = 1000;
    // float densityGas = 139.8;
    float acr = sqrt(3);
    // float Expfactor = 8; // 火成岩为8
    porousm = 8;
    // eta = pow(0.001, 1 - saturationGas) * pow(0.000022, saturationGas);
    eta = 1.0 * pow(10.0, -3);
    for (iz = 0; iz < nr; iz++)
    {
        for (iy = 0; iy < ny; iy++)
        {
            for (ix = 0; ix < nc; ix++)
            {
                if (((ix - nc / 2) * (ix - nc / 2) + (iy - ny / 2) * (iy - ny / 2)) <= 100)
                // 竖直井孔%%%%%%\// if(iz<(nr-14-ix)||iz>=(nr+14-ix))//45度倾斜井孔//if (iz>= 50)//if(ix>=50)
                {
                    vp[iz][iy][ix] = 1500.0f;
                    vs[iz][iy][ix] = 0.0f;
                    vf[iz][iy][ix] = 1500.0f;
                    rhos[iz][iy][ix] = 1000.0f;
                    rhof[iz][iy][ix] = 1000.0f;
                    rho[iz][iy][ix] = 1000.0f;
                    por = 1.0;
                    mu[iz][iy][ix] = 0.0f;
                    Kb1 = 0.0;                                                // 骨架压缩模量，声波测井原理与应用，P39
                    Ks1 = rhof[iz][iy][ix] * vp[iz][iy][ix] * vp[iz][iy][ix]; // 岩石固态颗粒的体积模量
                    Kf1 = rhof[iz][iy][ix] * vf[iz][iy][ix] * vf[iz][iy][ix]; // 孔隙流体的体积压缩模量
                    C[iz][iy][ix] = Kf1;
                    M[iz][iy][ix] = Kf1;
                    HH[iz][iy][ix] = Kf1;
                    H2u[iz][iy][ix] = Kf1;
                    C1[iz][iy][ix] = 0.0f;
                    C2[iz][iy][ix] = rhof[iz][iy][ix];
                }
                else
                {
                    vp[iz][iy][ix] = 5000.0f;
                    vs[iz][iy][ix] = 3300.0f;
                    rhos[iz][iy][ix] = 2450.0f;
                    rhof[iz][iy][ix] = 1000.0f;
                    vf[iz][iy][ix] = 1500.0f;
                    por = 0.1;
                    rho[iz][iy][ix] = (1 - por) * rhos[iz][iy][ix] + por * rhof[iz][iy][ix]; // 地层的密度
                    perm = 2 * pow(10.0, -12);
                    mu[iz][iy][ix] = (1 - por) * rhos[iz][iy][ix] * vs[iz][iy][ix] * vs[iz][iy][ix];
                    Kb1 = rhos[iz][iy][ix] * (1 - por) * (vp[iz][iy][ix] * vp[iz][iy][ix] - vs[iz][iy][ix] * vs[iz][iy][ix] * 4.0 / 3.0); // 干岩石体积弹性模量，声波测井原理与应用，P39
                    Ks1 = rhos[iz][iy][ix] * (vp[iz][iy][ix] * vp[iz][iy][ix] - vs[iz][iy][ix] * vs[iz][iy][ix] * 4.0 / 3.0);             // 岩石固态颗粒的体积模量
                    Kf1 = rhof[iz][iy][ix] * vf[iz][iy][ix] * vf[iz][iy][ix];                                                             // 孔隙流体的体积压缩模量

                    tao = 3.0;
                    a1 = 1 - Kb1 / Ks1;
                    M[iz][iy][ix] = Kf1 * Ks1 / (por * Ks1 + (a1 - por) * Kf1);
                    HH[iz][iy][ix] = a1 * a1 * M[iz][iy][ix] + Kb1 + mu[iz][iy][ix] * 4.0 / 3.0;
                    H2u[iz][iy][ix] = HH[iz][iy][ix] - 2 * mu[iz][iy][ix];
                    C[iz][iy][ix] = M[iz][iy][ix] * a1;

                    C1[iz][iy][ix] = eta / perm;
                    C2[iz][iy][ix] = (1 + 2 / porousm) * tao * rhof[iz][iy][ix] / por;
                }
            }
        }
    }
}
