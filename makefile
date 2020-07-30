all: src/ket.cpp

%.cpp: %.i 
	swig -c++ -python -o $@ $<  
	mv src/ket.py ket/__init__.py

clean:
	rm -rf src/ket.cpp ket/__init__.py
