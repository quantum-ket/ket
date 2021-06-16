all: src/ket.cpp

%.cpp: %.i 
	swig -c++ -doxygen -python -o $@ $<  
	sed -i 's/q):/q : quant):/g' src/ket.py 
	sed -i 's/idx):/idx : int):/g' src/ket.py 
	sed -i 's/Notes:/Note:/g' src/ket.py 
	sed -i 's/std::vector< double,std::allocator< double > >/List[float]/g' src/ket.py
	mv src/ket.py ket/ket.py

clean:
	rm -f src/ket.cpp ket.py
