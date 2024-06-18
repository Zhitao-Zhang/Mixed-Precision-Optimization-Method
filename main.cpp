#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda_fp16.h"
#pragma argsused
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <iostream>
#include "common.h"
#include "outputfunc.h"
#define PI 3.1415926535
#define e 2.718281828
// #define NZ 320
// #define NY 128   //此处是y轴的网格点数
// #define NX 128
// #define NP 32
#define NZ 300
#define NY 180 // 此处是y轴的网格点数
#define NX 180
#define NP 30
#define NL 200 // 变网格处
#define m 5	   // 连续变换
#define MM 1   // 连续变换
// #define BLOCK_SIZE_ZZ 2 //共享内存
//   定义xyz方向的进程数
#define PX 3
#define PY 3
#define PZ 3
#define tran 1 // MPI过渡带宽
// 缩放因子
#define d_Cvwp1 1.0
#define d_Cvwp2 1.0
#define d_Cvwp3 1.0
#define d_Cvup1 1.0
#define d_Cvup2 1.0
#define d_Cpp1 0.01
#define d_Cpp2 0.01
#define d_Csp1 0.01
#define d_Csp2 0.01
#define d_Csp3 0.01
#define d_Csp4 0.01
#define d_Cpmll 100.0

#define CUDA_CHECK(call)                                                         \
	do                                                                           \
	{                                                                            \
		cudaError_t result = call;                                               \
		if (result != cudaSuccess)                                               \
		{                                                                        \
			std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
					  << cudaGetErrorString(result) << std::endl;                \
			exit(EXIT_FAILURE);                                                  \
		}                                                                        \
	} while (0)

void callCUDAKernel(blockInfo *parameter, int nxt_pro, int nyt_pro, int nzt_pro, float DT, float F0, int sn, int *coord, int myid, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int sxnew, int synew, int sznew, int NT);
//---------------------------------------------------------------------------------------------------------------
// 主函�?
int main(int argc, char **argv)
{
	// int deviceCount = 0;
	// cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	// if (error_id != cudaSuccess) {
	//     printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
	//     printf("Result = FAIL\n");
	//     return -1;
	// }
	// if (deviceCount == 0) {
	//     printf("There are no available device(s) that support CUDA\n");
	// } else {
	//     printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	// }
	// cudaSetDevice(0);

	// 给定参数
	int NX_ext = NX + 2 * NP;
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;
	int nxt, nyt, nzt;
	int myid;			 // 进程
	int sx = NX_ext / 2; // 震源坐标点号
	int sy = NY_ext / 2;
	int sz = 20 + NP;
	int NT = 3500; // 时间层数
	// int NT1 = NT * m * m; // 连续变换2
	float H = 0.01; // 空间步长
	float RC = 1.0 * pow(10.0, -6);
	float DT = 1.0 * pow(10.0, -6); // 时间步长
	float DP = NP * H;
	float DT_H = DT / H;
	float F0 = 10 * pow(10.0, 3); // 震源主频,主频太高会造成频散
	float T0 = 1.2 / F0;
	float Vpmax = 4270.0; // 模型最大纵波速度,用于稳定性计
	float Vpmin = 1500.0; // 模型最小纵波速度,用于控制数值频
	float Vsmax = 2650.0;
	float Vsmin = 1500.0;
	//---------------------------------------------------------------------------
	// mpi初始�?
	MPI_Comm MCW, MC1;
	int numersize;
	MPI_Init(&argc, &argv);

	cudaSetDevice(2);

	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numersize);
	MPI_Comm_dup(MPI_COMM_WORLD, &MCW);
	MPI_Request request[4];
	// 定义维度进程 周期 坐标 排序 错误信息
	int back, forward, left, right, up, down;
	int dim[3], period[3], coord[3], reorder, err;
	dim[0] = PX;
	dim[1] = PY;
	dim[2] = PZ;
	reorder = 1;
	// x y z 都是非周期环
	period[0] = 0;
	period[1] = 0;
	period[2] = 0;
	// xyz方向每个进程中有多少个网
	nxt = NX_ext / PX;
	nyt = NY_ext / PY;
	nzt = NZ_ext / PZ;
	printf("nxt = %d  nyt = %d   nzt = %d\n", nxt, nyt, nzt);
	// 计算点源所在的进程 向上取整�?
	int x0pro, y0pro, z0pro;
	x0pro = (sx + nxt - 1) / nxt;
	y0pro = (sy + nyt - 1) / nyt;
	z0pro = (sz + nzt - 1) / nzt;
	// 创建笛卡尔拓�?
	err = MPI_Cart_create(MCW, 3, dim, period, reorder, &MC1);
	// 得到拓扑结构的标�?  之后能通过标识来吧对应的数据放入对应的进程�?   eg. 之后进行数据通信的时候，对本进程想要往左边进程传递数�?  只需要调用left
	err = MPI_Cart_shift(MC1, 0, 1, &left, &right);
	err = MPI_Cart_shift(MC1, 1, 1, &back, &forward);
	err = MPI_Cart_shift(MC1, 2, 1, &down, &up);
	err = MPI_Cart_coords(MC1, myid, 3, coord);
	printf("rank = %d\t  coord[0] = %d\t  coord[1] = %d\t   coord[2] = %d\t   left = %d\t  right = %d\t  back = %d\t  forward = %d\t  up = %d\t down = %d\n", myid, coord[0], coord[1], coord[2], left, right, back, forward, up, down);
	size_t mem_sizeHalf = nzt * nyt * nxt * sizeof(half); // half内存大小

	// 计算点源所在进程的索引
	int sxnew, synew, sznew;
	sxnew = (sx % nxt);
	synew = (sy % nyt);
	sznew = (sz % nzt);
	int sn = sznew * nxt * nyt + synew * nxt + sxnew;
	int coordx = coord[0];
	int coordy = coord[1];
	int coordz = coord[2];
	// printf("sn = %d    nxt = %d    nyt = %d    nzt = %d  \n", sn, nxt, nyt, nzt);
	//----------------------------------------------------------------------------------
	//----------------------------------------------------------------------------------
	// 应力和速度分量内存开�?
	float ***txx50 = space3d(nzt, nyt, nxt);
	float ***txx100 = space3d(nzt, nyt, nxt);
	float ***txx150 = space3d(nzt, nyt, nxt);
	float ***txx200 = space3d(nzt, nyt, nxt);
	float ***txx250 = space3d(nzt, nyt, nxt);
	float ***txx300 = space3d(nzt, nyt, nxt);
	float ***vuz50 = space3d(nzt, nyt, nxt);
	float ***vwz50 = space3d(nzt, nyt, nxt);
	float ***vux50 = space3d(nzt, nyt, nxt);
	float ***vwx50 = space3d(nzt, nyt, nxt);
	//------------------------------------------------------------------------------------------------
	// 定义密度，粘度系数分�?
	float ***rho_tempx;
	float ***rho_tempy;
	float ***rho_tempz;
	float ***rhof_extx;
	float ***rhof_exty;
	float ***rhof_extz;
	float ***VelocityWParameter1x;
	float ***VelocityWParameter1y;
	float ***VelocityWParameter1z;
	float ***VelocityWParameter2x;
	float ***VelocityWParameter2y;
	float ***VelocityWParameter2z;
	float ***VelocityWParameter3x;
	float ***VelocityWParameter3y;
	float ***VelocityWParameter3z;
	float ***VelocityUParameter1x;
	float ***VelocityUParameter1y;
	float ***VelocityUParameter1z;
	float ***VelocityUParameter2x;
	float ***VelocityUParameter2y;
	float ***VelocityUParameter2z;
	float ***PressParameter1;
	float ***PressParameter2;
	float ***StressParameter1;
	float ***StressParameter2;
	float ***StressParameter3;
	float ***StressParameterxy;
	float ***StressParameterxz;
	float ***StressParameteryz;
	float ***muxy, ***muxz, ***muyz;
	float ***C1x, ***C1y, ***C1z;
	float ***C2x, ***C2y, ***C2z;
	float ***M_ext;
	float ***C_ext;
	float ***HH_ext;
	float ***H2u_ext;
	muxy = space3d(nzt, nyt, nxt);
	muxz = space3d(nzt, nyt, nxt);
	muyz = space3d(nzt, nyt, nxt);
	HH_ext = space3d(nzt, nyt, nxt);
	H2u_ext = space3d(nzt, nyt, nxt);
	C_ext = space3d(nzt, nyt, nxt);
	M_ext = space3d(nzt, nyt, nxt);
	rho_tempx = space3d(nzt, nyt, nxt);
	rho_tempy = space3d(nzt, nyt, nxt);
	rho_tempz = space3d(nzt, nyt, nxt);
	rhof_extx = space3d(nzt, nyt, nxt);
	rhof_exty = space3d(nzt, nyt, nxt);
	rhof_extz = space3d(nzt, nyt, nxt);
	VelocityWParameter1x = space3d(nzt, nyt, nxt);
	VelocityWParameter1y = space3d(nzt, nyt, nxt);
	VelocityWParameter1z = space3d(nzt, nyt, nxt);
	VelocityWParameter2x = space3d(nzt, nyt, nxt);
	VelocityWParameter2y = space3d(nzt, nyt, nxt);
	VelocityWParameter2z = space3d(nzt, nyt, nxt);
	VelocityWParameter3x = space3d(nzt, nyt, nxt);
	VelocityWParameter3y = space3d(nzt, nyt, nxt);
	VelocityWParameter3z = space3d(nzt, nyt, nxt);
	VelocityUParameter1x = space3d(nzt, nyt, nxt);
	VelocityUParameter1y = space3d(nzt, nyt, nxt);
	VelocityUParameter1z = space3d(nzt, nyt, nxt);
	VelocityUParameter2x = space3d(nzt, nyt, nxt);
	VelocityUParameter2y = space3d(nzt, nyt, nxt);
	VelocityUParameter2z = space3d(nzt, nyt, nxt);
	PressParameter1 = space3d(nzt, nyt, nxt);
	PressParameter2 = space3d(nzt, nyt, nxt);
	StressParameter1 = space3d(nzt, nyt, nxt);
	StressParameter2 = space3d(nzt, nyt, nxt);
	StressParameter3 = space3d(nzt, nyt, nxt);
	StressParameterxy = space3d(nzt, nyt, nxt);
	StressParameterxz = space3d(nzt, nyt, nxt);
	StressParameteryz = space3d(nzt, nyt, nxt);
	C1x = space3d(nzt, nyt, nxt);
	C1y = space3d(nzt, nyt, nxt);
	C1z = space3d(nzt, nyt, nxt);
	C2x = space3d(nzt, nyt, nxt);
	C2y = space3d(nzt, nyt, nxt);
	C2z = space3d(nzt, nyt, nxt);
	// 缩放因子
	float Cv = 1.0 * pow(10.0, 8); // 速度项的缩放因子
	float Cp = 1;				   // 应力和压力项的缩放因�?
	float Cvwp1 = 1.0;
	float Cvwp2 = 1.0;
	float Cvwp3 = 1.0;
	float Cvup1 = 1.0;
	float Cvup2 = 1.0;
	float Cpp1 = 100;
	float Cpp2 = 100;
	float Csp1 = 100;
	float Csp2 = 100;
	float Csp3 = 100;
	float Csp4 = 100;
	float zero = 0.0;
	float Cpml = 0.01;
	int ix, iy, iz, it;
	float *muallxz = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *muallxy = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *muallyz = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *rhof_extx_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *rhof_exty_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *rhof_extz_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *rho_tempx_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *rho_tempy_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *rho_tempz_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *C1x_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *C1y_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *C1z_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *C2x_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *C2y_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *C2z_all = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *M_ext_alll = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *C_ext_alll = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *HH_ext_alll = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	float *H2u_ext_alll = (float *)malloc(sizeof(float *) * NY_ext * NZ_ext * NX_ext);
	memset(muallxz, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(muallxy, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(muallyz, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(rhof_extx_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(rhof_exty_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(rhof_extz_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(rho_tempx_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(rho_tempy_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(rho_tempz_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(C1x_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(C1y_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(C1z_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(C2x_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(C2y_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(C2z_all, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(M_ext_alll, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(C_ext_alll, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(HH_ext_alll, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));
	memset(H2u_ext_alll, 0, NY_ext * NZ_ext * NX_ext * sizeof(float));

	if (myid == 0)
	{
		float ***vs_ext_all, ***vp_ext_all, ***vf_ext_all, ***vfg_ext_all, ***rho_ext_all, ***rhof_ext_all, ***mu_ext_all, ***M_ext_all, ***C_ext_all, ***HH_ext_all, ***H2u_ext_all, ***C1_ext_all, ***C2_ext_all, ***rhos_ext_all;
		vs_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		vp_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		vf_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		vfg_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		rho_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		rhof_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		rhos_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		mu_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		HH_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		H2u_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		C1_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		C2_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		C_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		M_ext_all = space3d(NZ_ext, NY_ext, NX_ext);
		create_model_all(vp_ext_all, vs_ext_all, rhos_ext_all, vf_ext_all, vfg_ext_all, rho_ext_all, rhof_ext_all, M_ext_all, C_ext_all, C1_ext_all, C2_ext_all, HH_ext_all, H2u_ext_all, mu_ext_all, NZ_ext, NY_ext, NX_ext);

		for (iz = 1; iz < NZ_ext - 1; iz++)
			for (iy = 1; iy < NY_ext - 1; iy++)
				for (ix = 1; ix < NX_ext - 1; ix++)
				{
					if (mu_ext_all[iz][iy][ix] == 0 || mu_ext_all[iz][iy][ix + 1] == 0 || mu_ext_all[iz + 1][iy][ix] == 0 || mu_ext_all[iz + 1][iy][ix + 1] == 0)
						muallxz[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.0;
					else
						muallxz[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 1 / (0.25 * (1 / mu_ext_all[iz][iy][ix] + 1 / mu_ext_all[iz][iy][ix + 1] + 1 / mu_ext_all[iz + 1][iy][ix] + 1 / mu_ext_all[iz + 1][iy][ix + 1]));
					if (mu_ext_all[iz][iy][ix] == 0 || mu_ext_all[iz][iy + 1][ix] == 0 || mu_ext_all[iz + 1][iy][ix] == 0 || mu_ext_all[iz + 1][iy + 1][ix] == 0)
						muallyz[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.0;
					else
						muallyz[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 1 / (0.25 * (1 / mu_ext_all[iz][iy][ix] + 1 / mu_ext_all[iz][iy + 1][ix] + 1 / mu_ext_all[iz + 1][iy][ix] + 1 / mu_ext_all[iz + 1][iy + 1][ix]));
					if (mu_ext_all[iz][iy][ix] == 0 || mu_ext_all[iz][iy][ix + 1] == 0 || mu_ext_all[iz][iy + 1][ix] == 0 || mu_ext_all[iz][iy + 1][ix + 1] == 0)
						muallxy[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.0;
					else
						muallxy[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 1 / (0.25 * (1 / mu_ext_all[iz][iy][ix] + 1 / mu_ext_all[iz][iy][ix + 1] + 1 / mu_ext_all[iz][iy + 1][ix] + 1 / mu_ext_all[iz][iy + 1][ix + 1]));

					rhof_extx_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (rhof_ext_all[iz][iy][ix] + rhof_ext_all[iz][iy][ix + 1]);
					rhof_exty_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (rhof_ext_all[iz][iy][ix] + rhof_ext_all[iz][iy + 1][ix]);
					rhof_extz_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (rhof_ext_all[iz][iy][ix] + rhof_ext_all[iz + 1][iy][ix]);

					rho_tempx_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (rho_ext_all[iz][iy][ix] + rho_ext_all[iz][iy][ix + 1]);
					rho_tempy_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (rho_ext_all[iz][iy][ix] + rho_ext_all[iz][iy + 1][ix]);
					rho_tempz_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (rho_ext_all[iz][iy][ix] + rho_ext_all[iz + 1][iy][ix]);

					C1x_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (C1_ext_all[iz][iy][ix] + C1_ext_all[iz][iy][ix + 1]);
					C1y_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (C1_ext_all[iz][iy][ix] + C1_ext_all[iz][iy + 1][ix]);
					C1z_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (C1_ext_all[iz][iy][ix] + C1_ext_all[iz + 1][iy][ix]);

					C2x_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (C2_ext_all[iz][iy][ix] + C2_ext_all[iz][iy][ix + 1]);
					C2y_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (C2_ext_all[iz][iy][ix] + C2_ext_all[iz][iy + 1][ix]);
					C2z_all[iz * NX_ext * NY_ext + iy * NX_ext + ix] = 0.5 * (C2_ext_all[iz][iy][ix] + C2_ext_all[iz + 1][iy][ix]);

					M_ext_alll[iz * NX_ext * NY_ext + iy * NX_ext + ix] = M_ext_all[iz][iy][ix];
					C_ext_alll[iz * NX_ext * NY_ext + iy * NX_ext + ix] = C_ext_all[iz][iy][ix];
					HH_ext_alll[iz * NX_ext * NY_ext + iy * NX_ext + ix] = HH_ext_all[iz][iy][ix];
					H2u_ext_alll[iz * NX_ext * NY_ext + iy * NX_ext + ix] = H2u_ext_all[iz][iy][ix];
				}
		free_space3d(vs_ext_all, NZ_ext, NY_ext);
		free_space3d(vp_ext_all, NZ_ext, NY_ext);
		free_space3d(vf_ext_all, NZ_ext, NY_ext);
		free_space3d(vfg_ext_all, NZ_ext, NY_ext);
		free_space3d(rho_ext_all, NZ_ext, NY_ext);
		free_space3d(rhof_ext_all, NZ_ext, NY_ext);
		free_space3d(mu_ext_all, NZ_ext, NY_ext);
		free_space3d(M_ext_all, NZ_ext, NY_ext);
		free_space3d(C_ext_all, NZ_ext, NY_ext);
		free_space3d(HH_ext_all, NZ_ext, NY_ext);
		free_space3d(H2u_ext_all, NZ_ext, NY_ext);
		free_space3d(C1_ext_all, NZ_ext, NY_ext);
		free_space3d(C2_ext_all, NZ_ext, NY_ext);
		free_space3d(rhos_ext_all, NZ_ext, NY_ext);
	}

	MPI_Bcast(muallxz, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(muallxy, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(muallyz, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rhof_extx_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rhof_exty_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rhof_extz_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rho_tempx_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rho_tempy_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rho_tempz_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(C1x_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(C1y_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(C1z_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(C2x_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(C2y_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(C2z_all, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(M_ext_alll, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(C_ext_alll, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(HH_ext_alll, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(H2u_ext_alll, NZ_ext * NY_ext * NX_ext, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MCW);

	for (iz = 0; iz < nzt; iz++)
	{
		for (iy = 0; iy < nyt; iy++)
		{
			for (ix = 0; ix < nxt; ix++)
			{
				int findindex = (iz + nzt * coord[2]) * NX_ext * NY_ext + (iy + nyt * coord[1]) * NX_ext + (ix + nxt * coord[0]);
				muxz[iz][iy][ix] = muallxz[findindex];
				muxy[iz][iy][ix] = muallxy[findindex];
				muyz[iz][iy][ix] = muallyz[findindex];

				// 计算参数
				rhof_extx[iz][iy][ix] = rhof_extx_all[findindex];
				rhof_exty[iz][iy][ix] = rhof_exty_all[findindex];
				rhof_extz[iz][iy][ix] = rhof_extz_all[findindex];

				rho_tempx[iz][iy][ix] = rho_tempx_all[findindex];
				rho_tempy[iz][iy][ix] = rho_tempy_all[findindex];
				rho_tempz[iz][iy][ix] = rho_tempz_all[findindex];

				C1x[iz][iy][ix] = C1x_all[findindex];
				C1y[iz][iy][ix] = C1y_all[findindex];
				C1z[iz][iy][ix] = C1z_all[findindex];

				C2x[iz][iy][ix] = C2x_all[findindex];
				C2y[iz][iy][ix] = C2y_all[findindex];
				C2z[iz][iy][ix] = C2z_all[findindex];

				M_ext[iz][iy][ix] = M_ext_alll[findindex];
				C_ext[iz][iy][ix] = C_ext_alll[findindex];
				HH_ext[iz][iy][ix] = HH_ext_alll[findindex];
				H2u_ext[iz][iy][ix] = H2u_ext_alll[findindex];

				// 计算流体速度方程中的参数
				if (C1x[iz][iy][ix] == 0 || C1y[iz][iy][ix] == 0 || C1z[iz][iy][ix] == 0)
				{
					VelocityWParameter1x[iz][iy][ix] = 0;
					VelocityWParameter2x[iz][iy][ix] = 0;
					VelocityWParameter3x[iz][iy][ix] = 0;

					VelocityWParameter1y[iz][iy][ix] = 0;
					VelocityWParameter2y[iz][iy][ix] = 0;
					VelocityWParameter3y[iz][iy][ix] = 0;

					VelocityWParameter1z[iz][iy][ix] = 0;
					VelocityWParameter2z[iz][iy][ix] = 0;
					VelocityWParameter3z[iz][iy][ix] = 0;
				}
				else
				{
					VelocityWParameter1x[iz][iy][ix] = Cvwp1 * (C2x[iz][iy][ix] - (rhof_extx[iz][iy][ix] * rhof_extx[iz][iy][ix] / rho_tempx[iz][iy][ix]) - 0.5 * DT * C1x[iz][iy][ix]) / (C2x[iz][iy][ix] - (rhof_extx[iz][iy][ix] * rhof_extx[iz][iy][ix] / rho_tempx[iz][iy][ix]) + 0.5 * DT * C1x[iz][iy][ix]);
					VelocityWParameter2x[iz][iy][ix] = Cv * DT * Cvwp2 * (rhof_extx[iz][iy][ix] / rho_tempx[iz][iy][ix]) / (Cp * (C2x[iz][iy][ix] - (rhof_extx[iz][iy][ix] * rhof_extx[iz][iy][ix] / rho_tempx[iz][iy][ix]) + 0.5 * DT * C1x[iz][iy][ix]));
					VelocityWParameter3x[iz][iy][ix] = Cv * DT * Cvwp3 / (Cp * (C2x[iz][iy][ix] - rhof_extx[iz][iy][ix] * rhof_extx[iz][iy][ix] / rho_tempx[iz][iy][ix] + 0.5 * DT * C1x[iz][iy][ix]));

					VelocityWParameter1y[iz][iy][ix] = Cvwp1 * (C2y[iz][iy][ix] - (rhof_exty[iz][iy][ix] * rhof_exty[iz][iy][ix] / rho_tempy[iz][iy][ix]) - 0.5 * DT * C1y[iz][iy][ix]) / (C2y[iz][iy][ix] - (rhof_exty[iz][iy][ix] * rhof_exty[iz][iy][ix] / rho_tempy[iz][iy][ix]) + 0.5 * DT * C1y[iz][iy][ix]);
					VelocityWParameter2y[iz][iy][ix] = Cv * DT * Cvwp2 * (rhof_exty[iz][iy][ix] / rho_tempy[iz][iy][ix]) / (Cp * (C2y[iz][iy][ix] - (rhof_exty[iz][iy][ix] * rhof_exty[iz][iy][ix] / rho_tempy[iz][iy][ix]) + 0.5 * DT * C1y[iz][iy][ix]));
					VelocityWParameter3y[iz][iy][ix] = Cv * DT * Cvwp3 / (Cp * (C2y[iz][iy][ix] - rhof_exty[iz][iy][ix] * rhof_exty[iz][iy][ix] / rho_tempy[iz][iy][ix] + 0.5 * DT * C1y[iz][iy][ix]));

					VelocityWParameter1z[iz][iy][ix] = Cvwp1 * (C2z[iz][iy][ix] - (rhof_extz[iz][iy][ix] * rhof_extz[iz][iy][ix] / rho_tempz[iz][iy][ix]) - 0.5 * DT * C1z[iz][iy][ix]) / (C2z[iz][iy][ix] - (rhof_extz[iz][iy][ix] * rhof_extz[iz][iy][ix] / rho_tempz[iz][iy][ix]) + 0.5 * DT * C1z[iz][iy][ix]);
					VelocityWParameter2z[iz][iy][ix] = Cv * DT * Cvwp2 * (rhof_extz[iz][iy][ix] / rho_tempz[iz][iy][ix]) / (Cp * (C2z[iz][iy][ix] - (rhof_extz[iz][iy][ix] * rhof_extz[iz][iy][ix] / rho_tempz[iz][iy][ix]) + 0.5 * DT * C1z[iz][iy][ix]));
					VelocityWParameter3z[iz][iy][ix] = Cv * DT * Cvwp3 / (Cp * (C2z[iz][iy][ix] - rhof_extz[iz][iy][ix] * rhof_extz[iz][iy][ix] / rho_tempz[iz][iy][ix] + 0.5 * DT * C1z[iz][iy][ix]));
				}

				// 计算固体速度方程中的参数

				if (rho_tempx[iz][iy][ix] == 0 || rho_tempy[iz][iy][ix] == 0 || rho_tempz[iz][iy][ix] == 0)
				{
					VelocityUParameter1x[iz][iy][ix] = 0;
					VelocityUParameter1y[iz][iy][ix] = 0;
					VelocityUParameter1z[iz][iy][ix] = 0;

					VelocityUParameter2x[iz][iy][ix] = 0;
					VelocityUParameter2y[iz][iy][ix] = 0;
					VelocityUParameter2z[iz][iy][ix] = 0;
				}
				else
				{
					VelocityUParameter1x[iz][iy][ix] = Cv * DT * Cvup1 / (rho_tempx[iz][iy][ix] * Cp);
					VelocityUParameter1y[iz][iy][ix] = Cv * DT * Cvup1 / (rho_tempy[iz][iy][ix] * Cp);
					VelocityUParameter1z[iz][iy][ix] = Cv * DT * Cvup1 / (rho_tempz[iz][iy][ix] * Cp);

					VelocityUParameter2x[iz][iy][ix] = rhof_extx[iz][iy][ix] * Cvup2 / (rho_tempx[iz][iy][ix]);
					VelocityUParameter2y[iz][iy][ix] = rhof_exty[iz][iy][ix] * Cvup2 / (rho_tempy[iz][iy][ix]);
					VelocityUParameter2z[iz][iy][ix] = rhof_extz[iz][iy][ix] * Cvup2 / (rho_tempz[iz][iy][ix]);
				}
				// ����ѹ�������еĲ���
				PressParameter1[iz][iy][ix] = C_ext[iz][iy][ix] * DT * Cp * Cpp1 / Cv;
				PressParameter2[iz][iy][ix] = M_ext[iz][iy][ix] * DT * Cp * Cpp2 / Cv;

				// ������Ӧ�������еĲ���
				StressParameter1[iz][iy][ix] = Cp * DT * H2u_ext[iz][iy][ix] * Csp1 / Cv;
				StressParameter2[iz][iy][ix] = Cp * DT * HH_ext[iz][iy][ix] * Csp2 / Cv;
				StressParameter3[iz][iy][ix] = Cp * DT * C_ext[iz][iy][ix] * Csp3 / Cv;

				// ����ƫӦ�������еĲ���
				StressParameterxy[iz][iy][ix] = Cp * DT * muxy[iz][iy][ix] * Csp4 / Cv;
				StressParameterxz[iz][iy][ix] = Cp * DT * muxz[iz][iy][ix] * Csp4 / Cv;
				StressParameteryz[iz][iy][ix] = Cp * DT * muyz[iz][iy][ix] * Csp4 / Cv;
			}
		}
	}
	free(muallxz);
	free(muallxy);
	free(muallyz);
	free(rhof_extx_all);
	free(rhof_exty_all);
	free(rhof_extz_all);
	free(rho_tempx_all);
	free(rho_tempy_all);
	free(rho_tempz_all);
	free(C1x_all);
	free(C1y_all);
	free(C1z_all);
	free(C2x_all);
	free(C2y_all);
	free(C2z_all);
	free(C_ext);
	free(M_ext);
	free(H2u_ext);
	free(HH_ext);
	//---------------------------------------------------------------------
	// pml吸收边界设置，开辟内存，赋�
	float ***dxi, ***dxi2;
	float ***dyj, ***dyj2;
	float ***dzk, ***dzk2;
	float ***e_dxi, ***e_dxi2;
	float ***e_dyj, ***e_dyj2;
	float ***e_dzk, ***e_dzk2;
	dxi = space3d(nzt, nyt, nxt);
	dxi2 = space3d(nzt, nyt, nxt);
	dyj = space3d(nzt, nyt, nxt);
	dyj2 = space3d(nzt, nyt, nxt);
	dzk = space3d(nzt, nyt, nxt);
	dzk2 = space3d(nzt, nyt, nxt);
	e_dxi = space3d(nzt, nyt, nxt);
	e_dxi2 = space3d(nzt, nyt, nxt);
	e_dyj = space3d(nzt, nyt, nxt);
	e_dyj2 = space3d(nzt, nyt, nxt);
	e_dzk = space3d(nzt, nyt, nxt);
	e_dzk2 = space3d(nzt, nyt, nxt);
	float tt, x, y, z, xoleft, xoright, yoleft, yoright, zoleft, zoright, d0, best_dt, v0;
	xoleft = DP;
	xoright = (nxt - 1) * H - DP;
	yoleft = DP;
	yoright = (nyt - 1) * H - DP;
	zoleft = DP;
	zoright = (nzt - 1) * H - DP;
	// 用于对vx_x[iz][ix]等的求解，加入吸收边界，使数值衰�?
	//  检查左边界
	if ((left < 0) && (right >= 0) && (back >= 0) && (forward >= 0) && (up >= 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					x = ix * H;
					y = iy * H;
					z = iz * H;
					if (x >= 0 && x < xoleft)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((xoleft - x) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left < 0) && (right >= 0) && (back >= 0) && (forward >= 0) && (up < 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= 0 && x < xoleft)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((xoleft - x) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dzk[iz][iy][ix] = d0 * pow(((z - zoright) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left < 0) && (right >= 0) && (back >= 0) && (forward >= 0) && (up >= 0) && (down < 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= 0 && x < xoleft)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((xoleft - x) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dzk[iz][iy][ix] = d0 * pow(((zoleft - z) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	//======================================================
	if ((left < 0) && (right >= 0) && (back < 0) && (forward >= 0) && (up >= 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= 0 && x < xoleft)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((xoleft - x) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dyj[iz][iy][ix] = d0 * pow(((yoleft - y) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left < 0) && (right >= 0) && (back < 0) && (forward >= 0) && (up < 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= 0 && x < xoleft)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((xoleft - x) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dyj[iz][iy][ix] = d0 * pow(((yoleft - y) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dzk[iz][iy][ix] = d0 * pow(((z - zoright) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left < 0) && (right >= 0) && (back < 0) && (forward >= 0) && (up >= 0) && (down < 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= 0 && x < xoleft)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((xoleft - x) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dyj[iz][iy][ix] = d0 * pow(((yoleft - y) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dzk[iz][iy][ix] = d0 * pow(((zoleft - z) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	//==================================================================
	if ((left < 0) && (right >= 0) && (back >= 0) && (forward < 0) && (up >= 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= 0 && x < xoleft)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((xoleft - x) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dyj[iz][iy][ix] = d0 * pow(((y - yoright) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left < 0) && (right >= 0) && (back >= 0) && (forward < 0) && (up < 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= 0 && x < xoleft)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((xoleft - x) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dyj[iz][iy][ix] = d0 * pow(((y - yoright) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dzk[iz][iy][ix] = d0 * pow(((z - zoright) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left < 0) && (right >= 0) && (back >= 0) && (forward < 0) && (up >= 0) && (down < 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= 0 && x < xoleft)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((xoleft - x) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dyj[iz][iy][ix] = d0 * pow(((y - yoright) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dzk[iz][iy][ix] = d0 * pow(((zoleft - z) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	// 妫€鏌ュ彸杈圭晫
	if ((left >= 0) && (right < 0) && (back >= 0) && (forward >= 0) && (up >= 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= xoright && x < nxt * H)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((x - xoright) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left >= 0) && (right < 0) && (back >= 0) && (forward >= 0) && (up < 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= xoright && x < nxt * H)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((x - xoright) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dzk[iz][iy][ix] = d0 * pow(((z - zoright) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left >= 0) && (right < 0) && (back >= 0) && (forward >= 0) && (up >= 0) && (down < 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= xoright && x < nxt * H)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((x - xoright) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dzk[iz][iy][ix] = d0 * pow(((zoleft - z) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	//======================================================
	if ((left >= 0) && (right < 0) && (back < 0) && (forward >= 0) && (up >= 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= xoright && x < nxt * H)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((x - xoright) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dyj[iz][iy][ix] = d0 * pow(((yoleft - y) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left >= 0) && (right < 0) && (back < 0) && (forward >= 0) && (up < 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= xoright && x < nxt * H)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((x - xoright) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dyj[iz][iy][ix] = d0 * pow(((yoleft - y) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dzk[iz][iy][ix] = d0 * pow(((z - zoright) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left >= 0) && (right < 0) && (back < 0) && (forward >= 0) && (up >= 0) && (down < 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= xoright && x < nxt * H)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((x - xoright) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dyj[iz][iy][ix] = d0 * pow(((yoleft - y) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dzk[iz][iy][ix] = d0 * pow(((zoleft - z) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	//===================================================
	if ((left >= 0) && (right < 0) && (back >= 0) && (forward < 0) && (up >= 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= xoright && x < nxt * H)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((x - xoright) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dyj[iz][iy][ix] = d0 * pow(((y - yoright) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left >= 0) && (right < 0) && (back >= 0) && (forward < 0) && (up < 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= xoright && x < nxt * H)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((x - xoright) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dyj[iz][iy][ix] = d0 * pow(((y - yoright) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dzk[iz][iy][ix] = d0 * pow(((z - zoright) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left >= 0) && (right < 0) && (back >= 0) && (forward < 0) && (up >= 0) && (down < 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if (x >= xoright && x < nxt * H)
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dxi[iz][iy][ix] = d0 * pow(((x - xoright) / DP), 2);
						dxi2[iz][iy][ix] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
						e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
						e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dyj[iz][iy][ix] = d0 * pow(((y - yoright) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dzk[iz][iy][ix] = d0 * pow(((zoleft - z) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	// �??
	if ((left >= 0) && (right >= 0) && (back < 0) && (forward >= 0) && (up >= 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dyj[iz][iy][ix] = d0 * pow(((yoleft - y) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left >= 0) && (right >= 0) && (back < 0) && (forward >= 0) && (up < 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dyj[iz][iy][ix] = d0 * pow(((yoleft - y) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dzk[iz][iy][ix] = d0 * pow(((z - zoright) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left >= 0) && (right >= 0) && (back < 0) && (forward >= 0) && (up >= 0) && (down < 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dyj[iz][iy][ix] = d0 * pow(((yoleft - y) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dzk[iz][iy][ix] = d0 * pow(((zoleft - z) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	// �??
	if ((left >= 0) && (right >= 0) && (back >= 0) && (forward < 0) && (up >= 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dyj[iz][iy][ix] = d0 * pow(((y - yoright) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left >= 0) && (right >= 0) && (back >= 0) && (forward < 0) && (up < 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dyj[iz][iy][ix] = d0 * pow(((y - yoright) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dzk[iz][iy][ix] = d0 * pow(((z - zoright) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	if ((left >= 0) && (right >= 0) && (back >= 0) && (forward < 0) && (up >= 0) && (down < 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dyj[iz][iy][ix] = d0 * pow(((y - yoright) / DP), 2);
						dyj2[iz][iy][ix] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
						e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
						e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
					}
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dzk[iz][iy][ix] = d0 * pow(((zoleft - z) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	// �??
	if ((left >= 0) && (right >= 0) && (back >= 0) && (forward >= 0) && (up < 0) && (down >= 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						dzk[iz][iy][ix] = d0 * pow(((z - zoright) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}
	// �??
	if ((left >= 0) && (right >= 0) && (back >= 0) && (forward >= 0) && (up >= 0) && (down < 0))
	{
		for (iz = 0; iz < nzt; iz++)
			for (iy = 0; iy < nyt; iy++)
				for (ix = 0; ix < nxt; ix++)
				{
					x = ix * H;
					y = iy * H;
					z = iz * H;
					int pmlindex = iz * nxt * nyt + iy * nxt + ix;
					if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
					{
						v0 = 4000.0f;
						d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
						// d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
						dzk[iz][iy][ix] = d0 * pow(((zoleft - z) / DP), 2);
						dzk2[iz][iy][ix] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
						e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
						e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
					}
				}
	}

	//-----------------------------------------------并行计算-------------------------------------------------------------------------
	//-------------------------------------------------------------------------------------------------
	// 在主机端CPU定义参数，分配内�?
	// 地层参数
	blockInfo parameter;
	parameter.h_VelocityWParameter1x = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityWParameter1y = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityWParameter1z = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityWParameter2x = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityWParameter2y = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityWParameter2z = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityWParameter3x = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityWParameter3y = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityWParameter3z = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityUParameter1x = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityUParameter1y = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityUParameter1z = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityUParameter2x = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityUParameter2y = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_VelocityUParameter2z = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_PressParameter1 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_PressParameter2 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_StressParameter1 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_StressParameter2 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_StressParameter3 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_StressParameterxy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_StressParameterxz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_StressParameteryz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	// 速度应力
	parameter.h_vwx = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_vwy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_vwz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_ss = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_vux = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_vuy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_vuz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_txx = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_tyy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_tzz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_txz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_txy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_tyz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	// 前一时刻的速度
	parameter.h_vwx2 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_vwy2 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_vwz2 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	// pml内的差分�?
	parameter.h_pmlxSxx = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlySxy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlzSxz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlxSxy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlySyy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlzSyz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlxSxz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlySyz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlzSzz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlxVux = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlyVuy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlzVuz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlxVuy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlyVux = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlxVuz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlzVux = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlyVuz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlzVuy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlxVwx = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlyVwy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlzVwz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlxss = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlyss = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_pmlzss = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	// 前一时刻差分�?
	parameter.h_SXxx = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_SXxy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_SXxz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_SYxy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_SYyy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_SYyz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_SZxz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_SZyz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_SZzz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_SXss = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_SYss = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_SZss = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vuxx = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vuyy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vuzz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vuxy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vuyx = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vuyz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vuzy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vuxz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vuzx = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vwxx = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vwyy = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_Vwzz = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	// pml参数
	parameter.h_dxi = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_dyj = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_dzk = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_dxi2 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_dyj2 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_dzk2 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_e_dxi = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_e_dyj = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_e_dzk = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_e_dxi2 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_e_dyj2 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	parameter.h_e_dzk2 = (half *)calloc(nzt * nyt * nxt, sizeof(half));
	//---------------------------------------------------------------------------------
	// 在主机端将三维参数转化为一维参�?
	for (int k = 0; k < nzt; k++)
	{
		for (int j = 0; j < nyt; j++)
		{
			for (int i = 0; i < nxt; i++)
			{
				parameter.h_VelocityWParameter1x[i + j * nxt + k * nxt * nyt] = VelocityWParameter1x[k][j][i];
				parameter.h_VelocityWParameter1y[i + j * nxt + k * nxt * nyt] = VelocityWParameter1y[k][j][i];
				parameter.h_VelocityWParameter1z[i + j * nxt + k * nxt * nyt] = VelocityWParameter1z[k][j][i];
				parameter.h_VelocityWParameter2x[i + j * nxt + k * nxt * nyt] = VelocityWParameter2x[k][j][i];
				parameter.h_VelocityWParameter2y[i + j * nxt + k * nxt * nyt] = VelocityWParameter2y[k][j][i];
				parameter.h_VelocityWParameter2z[i + j * nxt + k * nxt * nyt] = VelocityWParameter2z[k][j][i];
				parameter.h_VelocityWParameter3x[i + j * nxt + k * nxt * nyt] = VelocityWParameter3x[k][j][i];
				parameter.h_VelocityWParameter3y[i + j * nxt + k * nxt * nyt] = VelocityWParameter3y[k][j][i];
				parameter.h_VelocityWParameter3z[i + j * nxt + k * nxt * nyt] = VelocityWParameter3z[k][j][i];
				parameter.h_VelocityUParameter1x[i + j * nxt + k * nxt * nyt] = VelocityUParameter1x[k][j][i];
				parameter.h_VelocityUParameter1y[i + j * nxt + k * nxt * nyt] = VelocityUParameter1y[k][j][i];
				parameter.h_VelocityUParameter1z[i + j * nxt + k * nxt * nyt] = VelocityUParameter1z[k][j][i];
				parameter.h_VelocityUParameter2x[i + j * nxt + k * nxt * nyt] = VelocityUParameter2x[k][j][i];
				parameter.h_VelocityUParameter2y[i + j * nxt + k * nxt * nyt] = VelocityUParameter2y[k][j][i];
				parameter.h_VelocityUParameter2z[i + j * nxt + k * nxt * nyt] = VelocityUParameter2z[k][j][i];
				parameter.h_PressParameter1[i + j * nxt + k * nxt * nyt] = PressParameter1[k][j][i];
				parameter.h_PressParameter2[i + j * nxt + k * nxt * nyt] = PressParameter2[k][j][i];
				parameter.h_StressParameter1[i + j * nxt + k * nxt * nyt] = StressParameter1[k][j][i];
				parameter.h_StressParameter2[i + j * nxt + k * nxt * nyt] = StressParameter2[k][j][i];
				parameter.h_StressParameter3[i + j * nxt + k * nxt * nyt] = StressParameter3[k][j][i];
				parameter.h_StressParameterxy[i + j * nxt + k * nxt * nyt] = StressParameterxy[k][j][i];
				parameter.h_StressParameterxz[i + j * nxt + k * nxt * nyt] = StressParameterxz[k][j][i];
				parameter.h_StressParameteryz[i + j * nxt + k * nxt * nyt] = StressParameteryz[k][j][i];
				parameter.h_dxi[i + j * nxt + k * nxt * nyt] = dxi[k][j][i] * Cpml;
				parameter.h_dyj[i + j * nxt + k * nxt * nyt] = dyj[k][j][i] * Cpml;
				parameter.h_dzk[i + j * nxt + k * nxt * nyt] = dzk[k][j][i] * Cpml;
				parameter.h_dxi2[i + j * nxt + k * nxt * nyt] = dxi2[k][j][i] * Cpml;
				parameter.h_dyj2[i + j * nxt + k * nxt * nyt] = dyj2[k][j][i] * Cpml;
				parameter.h_dzk2[i + j * nxt + k * nxt * nyt] = dzk2[k][j][i] * Cpml;
				parameter.h_e_dxi[i + j * nxt + k * nxt * nyt] = e_dxi[k][j][i] * Cpml;
				parameter.h_e_dyj[i + j * nxt + k * nxt * nyt] = e_dyj[k][j][i] * Cpml;
				parameter.h_e_dzk[i + j * nxt + k * nxt * nyt] = e_dzk[k][j][i] * Cpml;
				parameter.h_e_dxi2[i + j * nxt + k * nxt * nyt] = e_dxi2[k][j][i] * Cpml;
				parameter.h_e_dyj2[i + j * nxt + k * nxt * nyt] = e_dyj2[k][j][i] * Cpml;
				parameter.h_e_dzk2[i + j * nxt + k * nxt * nyt] = e_dzk2[k][j][i] * Cpml;
			}
		}
	}
	//------------------------------------------------------------------------------------------------------------
	// 在设备端GPU定义参数，分配内�?
	cudaMalloc(&parameter.d_VelocityWParameter1x, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityWParameter1y, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityWParameter1z, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityWParameter2x, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityWParameter2y, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityWParameter2z, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityWParameter3x, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityWParameter3y, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityWParameter3z, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityUParameter1x, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityUParameter1y, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityUParameter1z, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityUParameter2x, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityUParameter2y, mem_sizeHalf);
	cudaMalloc(&parameter.d_VelocityUParameter2z, mem_sizeHalf);
	cudaMalloc(&parameter.d_PressParameter1, mem_sizeHalf);
	cudaMalloc(&parameter.d_PressParameter2, mem_sizeHalf);
	cudaMalloc(&parameter.d_StressParameter1, mem_sizeHalf);
	cudaMalloc(&parameter.d_StressParameter2, mem_sizeHalf);
	cudaMalloc(&parameter.d_StressParameter3, mem_sizeHalf);
	cudaMalloc(&parameter.d_StressParameterxy, mem_sizeHalf);
	cudaMalloc(&parameter.d_StressParameterxz, mem_sizeHalf);
	cudaMalloc(&parameter.d_StressParameteryz, mem_sizeHalf);
	cudaMalloc(&parameter.d_vux, mem_sizeHalf);
	cudaMalloc(&parameter.d_vuy, mem_sizeHalf);
	cudaMalloc(&parameter.d_vuz, mem_sizeHalf);
	cudaMalloc(&parameter.d_txx, mem_sizeHalf);
	cudaMalloc(&parameter.d_tyy, mem_sizeHalf);
	cudaMalloc(&parameter.d_tzz, mem_sizeHalf);
	cudaMalloc(&parameter.d_txz, mem_sizeHalf);
	cudaMalloc(&parameter.d_txy, mem_sizeHalf);
	cudaMalloc(&parameter.d_tyz, mem_sizeHalf);
	cudaMalloc(&parameter.d_ss, mem_sizeHalf);
	cudaMalloc(&parameter.d_vwx, mem_sizeHalf);
	cudaMalloc(&parameter.d_vwy, mem_sizeHalf);
	cudaMalloc(&parameter.d_vwz, mem_sizeHalf);
	cudaMalloc(&parameter.d_vwx2, mem_sizeHalf);
	cudaMalloc(&parameter.d_vwy2, mem_sizeHalf);
	cudaMalloc(&parameter.d_vwz2, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlxSxx, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlySxy, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlzSxz, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlxSxy, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlySyy, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlzSyz, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlxSxz, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlySyz, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlzSzz, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlxVux, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlyVuy, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlzVuz, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlxVuy, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlyVux, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlxVuz, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlzVux, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlyVuz, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlzVuy, mem_sizeHalf);
	cudaMalloc(&parameter.d_SXxx, mem_sizeHalf);
	cudaMalloc(&parameter.d_SXxy, mem_sizeHalf);
	cudaMalloc(&parameter.d_SXxz, mem_sizeHalf);
	cudaMalloc(&parameter.d_SYxy, mem_sizeHalf);
	cudaMalloc(&parameter.d_SYyy, mem_sizeHalf);
	cudaMalloc(&parameter.d_SYyz, mem_sizeHalf);
	cudaMalloc(&parameter.d_SZxz, mem_sizeHalf);
	cudaMalloc(&parameter.d_SZyz, mem_sizeHalf);
	cudaMalloc(&parameter.d_SZzz, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vuxx, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vuyy, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vuzz, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vuxy, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vuyx, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vuxz, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vuzx, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vuyz, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vuzy, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlxVwx, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlyVwy, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlzVwz, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlxss, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlyss, mem_sizeHalf);
	cudaMalloc(&parameter.d_pmlzss, mem_sizeHalf);
	cudaMalloc(&parameter.d_SXss, mem_sizeHalf);
	cudaMalloc(&parameter.d_SYss, mem_sizeHalf);
	cudaMalloc(&parameter.d_SZss, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vwxx, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vwyy, mem_sizeHalf);
	cudaMalloc(&parameter.d_Vwzz, mem_sizeHalf);
	cudaMalloc(&parameter.d_dxi, mem_sizeHalf);
	cudaMalloc(&parameter.d_dyj, mem_sizeHalf);
	cudaMalloc(&parameter.d_dzk, mem_sizeHalf);
	cudaMalloc(&parameter.d_dxi2, mem_sizeHalf);
	cudaMalloc(&parameter.d_dyj2, mem_sizeHalf);
	cudaMalloc(&parameter.d_dzk2, mem_sizeHalf);
	cudaMalloc(&parameter.d_e_dxi, mem_sizeHalf);
	cudaMalloc(&parameter.d_e_dyj, mem_sizeHalf);
	cudaMalloc(&parameter.d_e_dzk, mem_sizeHalf);
	cudaMalloc(&parameter.d_e_dxi2, mem_sizeHalf);
	cudaMalloc(&parameter.d_e_dyj2, mem_sizeHalf);
	cudaMalloc(&parameter.d_e_dzk2, mem_sizeHalf);
	//---------------------------------------------------------------------------------------------
	// 开始时间递推
	// 这里的代码应该修正一下，应该从数据里面提取最大最小速度
	// 检查稳定性条�?
	best_dt = 6.0 * H / (7.0 * sqrt(2.0) * Vpmax);
	if (DT >= best_dt)
		printf("时间步长过大，应该小�?  %f\n", best_dt);
	// 控制网格频散
	if (Vsmin / (F0 * H) < 15)
		printf("空间步长太大,可能引起明显的网格频散\n");
	//--------------------------------------------------------------------------------------------
	// 进行参数传递，将主机端CPU参数传递到设备端GPU
	cudaMemcpy(parameter.d_VelocityWParameter1x, parameter.h_VelocityWParameter1x, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityWParameter1y, parameter.h_VelocityWParameter1y, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityWParameter1z, parameter.h_VelocityWParameter1z, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityWParameter2x, parameter.h_VelocityWParameter2x, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityWParameter2y, parameter.h_VelocityWParameter2y, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityWParameter2z, parameter.h_VelocityWParameter2z, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityWParameter3x, parameter.h_VelocityWParameter3x, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityWParameter3y, parameter.h_VelocityWParameter3y, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityWParameter3z, parameter.h_VelocityWParameter3z, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityUParameter1x, parameter.h_VelocityUParameter1x, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityUParameter1y, parameter.h_VelocityUParameter1y, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityUParameter1z, parameter.h_VelocityUParameter1z, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityUParameter2x, parameter.h_VelocityUParameter2x, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityUParameter2y, parameter.h_VelocityUParameter2y, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_VelocityUParameter2z, parameter.h_VelocityUParameter2z, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_PressParameter1, parameter.h_PressParameter1, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_PressParameter2, parameter.h_PressParameter2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_StressParameter1, parameter.h_StressParameter1, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_StressParameter2, parameter.h_StressParameter2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_StressParameter3, parameter.h_StressParameter3, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_StressParameterxy, parameter.h_StressParameterxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_StressParameterxz, parameter.h_StressParameterxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_StressParameteryz, parameter.h_StressParameteryz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_dxi, parameter.h_dxi, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_dyj, parameter.h_dyj, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_dzk, parameter.h_dzk, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_dxi2, parameter.h_dxi2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_dyj2, parameter.h_dyj2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_dzk2, parameter.h_dzk2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_e_dxi, parameter.h_e_dxi, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_e_dyj, parameter.h_e_dyj, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_e_dzk, parameter.h_e_dzk, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_e_dxi2, parameter.h_e_dxi2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_e_dyj2, parameter.h_e_dyj2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_e_dzk2, parameter.h_e_dzk2, mem_sizeHalf, cudaMemcpyHostToDevice);

	cudaMemcpy(parameter.d_pmlxSxx, parameter.h_pmlxSxx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlySxy, parameter.h_pmlySxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlzSxz, parameter.h_pmlzSxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlxSxy, parameter.h_pmlxSxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlySyy, parameter.h_pmlySyy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlzSyz, parameter.h_pmlzSyz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlxSxz, parameter.h_pmlxSxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlySyz, parameter.h_pmlySyz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlzSzz, parameter.h_pmlzSzz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SXxx, parameter.h_SXxx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SXxy, parameter.h_SXxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SXxz, parameter.h_SXxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SYxy, parameter.h_SYxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SYyy, parameter.h_SYyy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SYyz, parameter.h_SYyz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SZxz, parameter.h_SZxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SZyz, parameter.h_SZyz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SZzz, parameter.h_SZzz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlxss, parameter.h_pmlxss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlyss, parameter.h_pmlyss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlzss, parameter.h_pmlzss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SXss, parameter.h_SXss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SYss, parameter.h_SYss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_SZss, parameter.h_SZss, mem_sizeHalf, cudaMemcpyHostToDevice);

	cudaMemcpy(parameter.d_pmlxVux, parameter.h_pmlxVux, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlyVuy, parameter.h_pmlyVuy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlzVuz, parameter.h_pmlzVuz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlxVuy, parameter.h_pmlxVuy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlyVux, parameter.h_pmlyVux, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlxVuz, parameter.h_pmlxVuz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlzVux, parameter.h_pmlzVux, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlyVuz, parameter.h_pmlyVuz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlzVuy, parameter.h_pmlzVuy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vuxx, parameter.h_Vuxx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vuyy, parameter.h_Vuyy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vuzz, parameter.h_Vuzz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vuxy, parameter.h_Vuxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vuyx, parameter.h_Vuyx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vuxz, parameter.h_Vuxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vuzx, parameter.h_Vuzx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vuyz, parameter.h_Vuyz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vuzy, parameter.h_Vuzy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlxVwx, parameter.h_pmlxVwx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlyVwy, parameter.h_pmlyVwy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_pmlzVwz, parameter.h_pmlzVwz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vwxx, parameter.h_Vwxx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vwyy, parameter.h_Vwyy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_Vwzz, parameter.h_Vwzz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_vwx2, parameter.h_vwx2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_vwy2, parameter.h_vwy2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(parameter.d_vwz2, parameter.h_vwz2, mem_sizeHalf, cudaMemcpyHostToDevice);

	CUDA_CHECK(cudaMemcpy(parameter.d_txx, parameter.h_txx, mem_sizeHalf, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(parameter.d_tyy, parameter.h_tyy, mem_sizeHalf, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(parameter.d_tzz, parameter.h_tzz, mem_sizeHalf, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(parameter.d_txy, parameter.h_txy, mem_sizeHalf, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(parameter.d_txz, parameter.h_txz, mem_sizeHalf, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(parameter.d_tyz, parameter.h_tyz, mem_sizeHalf, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(parameter.d_ss, parameter.h_ss, mem_sizeHalf, cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(parameter.d_vux, parameter.h_vux, mem_sizeHalf, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(parameter.d_vuy, parameter.h_vuy, mem_sizeHalf, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(parameter.d_vuz, parameter.h_vuz, mem_sizeHalf, cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(parameter.d_vwx, parameter.h_vwx, mem_sizeHalf, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(parameter.d_vwy, parameter.h_vwy, mem_sizeHalf, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(parameter.d_vwz, parameter.h_vwz, mem_sizeHalf, cudaMemcpyHostToDevice));
	//---------------------------------------------------------------------------------------------------------------------
	// 在时间上进行迭代
	clock_t start, finish;
	start = clock();

	callCUDAKernel(&parameter, nxt, nyt, nzt, DT, F0, sn, coord, myid, up, down, back, forward, left, right, MCW, sxnew, synew, sznew, NT);
	MPI_Barrier(MCW);
	finish = clock();
	printf("%f seconds\n", (float)(finish - start) / CLOCKS_PER_SEC);
	//--------------------------------------------------------------------------------------------------------------
	// 输出波场快照
	printf("NZ_ext = %d NY_ext = %d NX_ext = %d\n", NZ_ext, NY_ext, NX_ext);
	printf("NZ_ext = %d  NT = %d\n", NZ_ext, NT);
	//---------------------------------------------------------------------------------------------------------------
	// 主机端释放内�?
	free(parameter.h_dxi);
	free(parameter.h_dxi2);
	free(parameter.h_dyj);
	free(parameter.h_dyj2);
	free(parameter.h_dzk);
	free(parameter.h_dzk2);
	free(parameter.h_e_dxi);
	free(parameter.h_e_dxi2);
	free(parameter.h_e_dyj);
	free(parameter.h_e_dyj2);
	free(parameter.h_e_dzk);
	free(parameter.h_e_dzk2);
	free(parameter.h_pmlxSxx);
	free(parameter.h_pmlxSxy);
	free(parameter.h_pmlxSxz);
	free(parameter.h_pmlxVux);
	free(parameter.h_pmlxVuy);
	free(parameter.h_pmlxVuz);
	free(parameter.h_pmlySxy);
	free(parameter.h_pmlySyy);
	free(parameter.h_pmlySyz);
	free(parameter.h_pmlyVux);
	free(parameter.h_pmlyVuy);
	free(parameter.h_pmlyVuz);
	free(parameter.h_pmlzSxz);
	free(parameter.h_pmlzSyz);
	free(parameter.h_pmlzSzz);
	free(parameter.h_pmlzVux);
	free(parameter.h_pmlzVuy);
	free(parameter.h_pmlzVuz);
	free(parameter.h_SXxx);
	free(parameter.h_SXxy);
	free(parameter.h_SXxz);
	free(parameter.h_SYxy);
	free(parameter.h_SYyy);
	free(parameter.h_SYyz);
	free(parameter.h_SZxz);
	free(parameter.h_SZyz);
	free(parameter.h_SZzz);
	free(parameter.h_txx);
	free(parameter.h_tyy);
	free(parameter.h_tzz);
	free(parameter.h_txy);
	free(parameter.h_txz);
	free(parameter.h_tyz);
	free(parameter.h_vux);
	free(parameter.h_vuy);
	free(parameter.h_vuz);
	free(parameter.h_Vuxx);
	free(parameter.h_Vuxy);
	free(parameter.h_Vuxz);
	free(parameter.h_Vuyx);
	free(parameter.h_Vuyy);
	free(parameter.h_Vuyz);
	free(parameter.h_Vuzx);
	free(parameter.h_Vuzy);
	free(parameter.h_Vuzz);
	free(parameter.h_VelocityWParameter1x);
	free(parameter.h_VelocityWParameter1y);
	free(parameter.h_VelocityWParameter1z);
	free(parameter.h_VelocityWParameter2x);
	free(parameter.h_VelocityWParameter2y);
	free(parameter.h_VelocityWParameter2z);
	free(parameter.h_VelocityWParameter3x);
	free(parameter.h_VelocityWParameter3y);
	free(parameter.h_VelocityWParameter3z);
	free(parameter.h_VelocityUParameter1x);
	free(parameter.h_VelocityUParameter1y);
	free(parameter.h_VelocityUParameter1z);
	free(parameter.h_VelocityUParameter2x);
	free(parameter.h_VelocityUParameter2y);
	free(parameter.h_VelocityUParameter2z);
	free(parameter.h_PressParameter1);
	free(parameter.h_PressParameter2);
	free(parameter.h_StressParameter1);
	free(parameter.h_StressParameter2);
	free(parameter.h_StressParameter3);
	free(parameter.h_StressParameterxy);
	free(parameter.h_StressParameterxz);
	free(parameter.h_StressParameteryz);
	free_space3d(C1x, nzt, nyt);
	free_space3d(C1y, nzt, nyt);
	free_space3d(C1z, nzt, nyt);
	free_space3d(C2x, nzt, nyt);
	free_space3d(C2y, nzt, nyt);
	free_space3d(C2z, nzt, nyt);
	free_space3d(rhof_extx, nzt, nyt);
	free_space3d(rhof_exty, nzt, nyt);
	free_space3d(rhof_extz, nzt, nyt);
	free_space3d(rho_tempx, nzt, nyt);
	free_space3d(rho_tempy, nzt, nyt);
	free_space3d(rho_tempz, nzt, nyt);
	free_space3d(muxy, nzt, nyt);
	free_space3d(muxz, nzt, nyt);
	free_space3d(muyz, nzt, nyt);
	free_space3d(vuz50, nzt, nyt);
	free_space3d(vwz50, nzt, nyt);
	free_space3d(vux50, nzt, nyt);
	free_space3d(vwx50, nzt, nyt);
	free_space3d(txx50, nzt, nyt);
	free_space3d(txx100, nzt, nyt);
	free_space3d(txx150, nzt, nyt);
	free_space3d(txx200, nzt, nyt);
	free_space3d(txx250, nzt, nyt);
	free_space3d(txx300, nzt, nyt);
	free_space3d(dxi, nzt, nyt);
	free_space3d(e_dxi, nzt, nyt);
	free_space3d(dxi2, nzt, nyt);
	free_space3d(e_dxi2, nzt, nyt);
	free_space3d(dyj, nzt, nyt);
	free_space3d(e_dyj, nzt, nyt);
	free_space3d(dyj2, nzt, nyt);
	free_space3d(e_dyj2, nzt, nyt);
	free_space3d(dzk, nzt, nyt);
	free_space3d(e_dzk, nzt, nyt);
	free_space3d(dzk2, nzt, nyt);
	free_space3d(e_dzk2, nzt, nyt);
	free_space3d(VelocityWParameter1x, nzt, nyt);
	free_space3d(VelocityWParameter1y, nzt, nyt);
	free_space3d(VelocityWParameter1z, nzt, nyt);
	free_space3d(VelocityWParameter2x, nzt, nyt);
	free_space3d(VelocityWParameter2y, nzt, nyt);
	free_space3d(VelocityWParameter2z, nzt, nyt);
	free_space3d(VelocityWParameter3x, nzt, nyt);
	free_space3d(VelocityWParameter3y, nzt, nyt);
	free_space3d(VelocityWParameter3z, nzt, nyt);
	free_space3d(VelocityUParameter1x, nzt, nyt);
	free_space3d(VelocityUParameter1y, nzt, nyt);
	free_space3d(VelocityUParameter1z, nzt, nyt);
	free_space3d(VelocityUParameter2x, nzt, nyt);
	free_space3d(VelocityUParameter2y, nzt, nyt);
	free_space3d(VelocityUParameter2z, nzt, nyt);
	free_space3d(PressParameter1, nzt, nyt);
	free_space3d(PressParameter2, nzt, nyt);
	free_space3d(StressParameter1, nzt, nyt);
	free_space3d(StressParameter2, nzt, nyt);
	free_space3d(StressParameter3, nzt, nyt);
	free_space3d(StressParameterxy, nzt, nyt);
	free_space3d(StressParameterxz, nzt, nyt);
	free_space3d(StressParameteryz, nzt, nyt);
	// 设备端释放内�?
	cudaFree(parameter.d_VelocityWParameter1x);
	cudaFree(parameter.d_VelocityWParameter1y);
	cudaFree(parameter.d_VelocityWParameter1z);
	cudaFree(parameter.d_VelocityWParameter2x);
	cudaFree(parameter.d_VelocityWParameter2y);
	cudaFree(parameter.d_VelocityWParameter2z);
	cudaFree(parameter.d_VelocityWParameter3x);
	cudaFree(parameter.d_VelocityWParameter3y);
	cudaFree(parameter.d_VelocityWParameter3z);
	cudaFree(parameter.d_VelocityUParameter1x);
	cudaFree(parameter.d_VelocityUParameter1y);
	cudaFree(parameter.d_VelocityUParameter1z);
	cudaFree(parameter.d_VelocityUParameter2x);
	cudaFree(parameter.d_VelocityUParameter2y);
	cudaFree(parameter.d_VelocityUParameter2z);
	cudaFree(parameter.d_PressParameter1);
	cudaFree(parameter.d_PressParameter2);
	cudaFree(parameter.d_StressParameter1);
	cudaFree(parameter.d_StressParameter2);
	cudaFree(parameter.d_StressParameter3);
	cudaFree(parameter.d_StressParameterxy);
	cudaFree(parameter.d_StressParameterxz);
	cudaFree(parameter.d_StressParameteryz);
	cudaFree(parameter.d_dxi);
	cudaFree(parameter.d_dxi2);
	cudaFree(parameter.d_dyj);
	cudaFree(parameter.d_dyj2);
	cudaFree(parameter.d_dzk);
	cudaFree(parameter.d_dzk2);
	cudaFree(parameter.d_e_dxi);
	cudaFree(parameter.d_e_dxi2);
	cudaFree(parameter.d_e_dyj);
	cudaFree(parameter.d_e_dyj2);
	cudaFree(parameter.d_e_dzk);
	cudaFree(parameter.d_e_dzk2);
	cudaFree(parameter.d_pmlxSxx);
	cudaFree(parameter.d_pmlxSxy);
	cudaFree(parameter.d_pmlxSxz);
	cudaFree(parameter.d_pmlxVux);
	cudaFree(parameter.d_pmlxVuy);
	cudaFree(parameter.d_pmlxVuz);
	cudaFree(parameter.d_pmlySxy);
	cudaFree(parameter.d_pmlySyy);
	cudaFree(parameter.d_pmlySyz);
	cudaFree(parameter.d_pmlyVux);
	cudaFree(parameter.d_pmlyVuy);
	cudaFree(parameter.d_pmlyVuz);
	cudaFree(parameter.d_pmlzSxz);
	cudaFree(parameter.d_pmlzSyz);
	cudaFree(parameter.d_pmlzSzz);
	cudaFree(parameter.d_pmlzVux);
	cudaFree(parameter.d_pmlzVuy);
	cudaFree(parameter.d_pmlzVuz);
	cudaFree(parameter.d_SXxx);
	cudaFree(parameter.d_SXxy);
	cudaFree(parameter.d_SXxz);
	cudaFree(parameter.d_SYxy);
	cudaFree(parameter.d_SYyy);
	cudaFree(parameter.d_SYyz);
	cudaFree(parameter.d_SZxz);
	cudaFree(parameter.d_SZyz);
	cudaFree(parameter.d_SZzz);
	cudaFree(parameter.d_txx);
	cudaFree(parameter.d_tyy);
	cudaFree(parameter.d_tzz);
	cudaFree(parameter.d_txy);
	cudaFree(parameter.d_txz);
	cudaFree(parameter.d_tyz);
	cudaFree(parameter.d_vux);
	cudaFree(parameter.d_vuy);
	cudaFree(parameter.d_vuz);
	cudaFree(parameter.d_ss);
	cudaFree(parameter.d_vwx);
	cudaFree(parameter.d_vwy);
	cudaFree(parameter.d_vwz);
	cudaFree(parameter.d_vwx2);
	cudaFree(parameter.d_vwy2);
	cudaFree(parameter.d_vwz2);
	cudaFree(parameter.d_Vuxx);
	cudaFree(parameter.d_Vuxy);
	cudaFree(parameter.d_Vuxz);
	cudaFree(parameter.d_Vuyx);
	cudaFree(parameter.d_Vuyy);
	cudaFree(parameter.d_Vuyz);
	cudaFree(parameter.d_Vuzx);
	cudaFree(parameter.d_Vuzy);
	cudaFree(parameter.d_Vuzz);
	cudaFree(parameter.d_pmlxVwx);
	cudaFree(parameter.d_pmlyVwy);
	cudaFree(parameter.d_pmlzVwz);
	cudaFree(parameter.d_pmlxss);
	cudaFree(parameter.d_pmlyss);
	cudaFree(parameter.d_pmlzss);
	cudaFree(parameter.d_SXss);
	cudaFree(parameter.d_SYss);
	cudaFree(parameter.d_SZss);
	cudaFree(parameter.d_Vwxx);
	cudaFree(parameter.d_Vwyy);
	cudaFree(parameter.d_Vwzz);
	MPI_Finalize();
	printf("\Press any key to exit program...");
	return 0;
}