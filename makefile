NVCC    := nvcc
TARGET  := simple_sph
SRC     := simple_sph.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O3 -Xcompiler -Wall -std=c++14 -o $@ $<

clean:
	rm -f $(TARGET) positions.csv