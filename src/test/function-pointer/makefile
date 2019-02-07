all: fp

fp: fp.cc col.o
	$(CXX) -std=c++11 -o $@ $^

col.o: col.cc
	$(CXX) -c -o $@ $<

clean:
	$(RM) -f fp *.o


