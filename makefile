all: src/ket.cpp

%.cpp: %.i 
	swig -c++ -doxygen -python -o $@ $<  
	sed -i 's/Notes:/Note:/g' src/ket.py 
	mv src/ket.py ket/__init__.py

clean:
	rm -f src/ket.cpp ket/__init__.py
