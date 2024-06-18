#include <stdio.h>
#pragma argsused
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>

int divUp(int a, int b);
float **space2d(int nr, int nc);
float ***space3d(int nr, int ny, int nc);
void free_space2d(float **a, int nr);
void free_space3d(float ***b, int nr, int ny);
void wfile(char filename[], float **data, int nr, int nc);
void wfile3d(char filename[], float ***data, int nr, int ny, int nc);
void wfile1d(char filename[], float *data, int nr, int ny, int nc);
void create_model_all(float ***vp, float ***vs, float ***rhos, float ***vf, float ***vfg, float ***rho, float ***rhof, float ***M, float ***C, float ***C1, float ***C2, float ***HH, float ***H2u, float ***mu, int nr, int ny, int nc);
