all: src/ket.cpp
	sed -i 's/q):/q : quant):/g' src/ket.py 
	sed -i 's/idx):/idx : int):/g' src/ket.py 
	mv src/ket.py ket/ket.py

%.cpp: %.i 
	swig -c++ -doxygen -python -o $@ $<
	
clean:
	rm -f src/ket.cpp ket.py
