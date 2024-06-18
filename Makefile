CXX = mpic++
NVCC = nvcc
CXXFLAGS = -w -I/opt/spack/opt/spack/linux-debian11-cascadelake/gcc-10.2.1/cuda-11.7.0-42vtagqikeyleemufys4fi54tnjh75by/include -I/opt/spack/opt/spack/linux-debian11-cascadelake/gcc-10.2.1/openmpi-4.1.3-byrqipvj7nkafujsgj7q6mtjz2jqydnn/include
LDFLAGS = -L/opt/spack/opt/spack/linux-debian11-cascadelake/gcc-10.2.1/cuda-11.7.0-42vtagqikeyleemufys4fi54tnjh75by/lib64 -Wl,-rpath=/opt/spack/opt/spack/linux-debian11-cascadelake/gcc-10.2.1/cuda-11.7.0-42vtagqikeyleemufys4fi54tnjh75by/lib64 /opt/spack/opt/spack/linux-debian11-cascadelake/gcc-10.2.1/cuda-11.7.0-42vtagqikeyleemufys4fi54tnjh75by/lib64/libcudart.so.11.0#-lcudart

OBJ_MAIN = main.o
OBJ_KERNEL = kernel.o
OBJ_OUTPUT = outputfunc.o
EXECUTABLE = my_program

all:$(EXECUTABLE)

$(EXECUTABLE):$(OBJ_MAIN) $(OBJ_KERNEL) $(OBJ_OUTPUT)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_MAIN): main.cpp common.h outputfunc.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_KERNEL): kernel.cu common.h outputfunc.h
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(OBJ_OUTPUT): outputfunc.cpp common.h outputfunc.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY:clean
clean:
	rm -f *.o my_program

