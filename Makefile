CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi 

RT: main.cpp
	g++ $(CFLAGS) -o VulkanEngine main.cpp $(LDFLAGS)

debug: main.cpp
	g++ $(CFLAGS) -o VulkanEngine_Debug main.cpp $(LDFLAGS) -D NDEBUG=0

.PHONY: test clean

test: RT
	./RT

clean:
	rm -f RT
	rm -f RT_Debug

re: clean RT