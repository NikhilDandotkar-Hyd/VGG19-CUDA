LIB = -L cudnn/lib64 -L /usr/local/lib
CUDNN_INC = -Icudnn/include

exec: vgg19_lib.o main.o image.o utility.o
	g++ -std=c++11 -o exec vgg19_lib.o main.o image.o utility.o -lcudnn -lcudart -lcublas -lcurand

vgg19_lib.o: vgg19_lib.cu 
	nvcc -std=c++11 $(CUDNN_INC) $(LIB) -c vgg19_lib.cu -lcudnn -lcublas -lcurand -I.

image.o:image.cpp
	g++ -std=c++11 image.cpp -c

main.o:main.cu vgg19_lib.o image.o utility.o
	nvcc -std=c++11  $(CUDNN_INC) $(LIB) -c main.cu -lcudnn -lcublas -lcurand -I.

utility.o:utility.cu image.o
	nvcc $(CUDNN_INC) $(LIB) -c utility.cu -lcudnn -lcublas -lcurand -I.


clean:
	rm -f *.out *.exe *.o *~ *.mod
