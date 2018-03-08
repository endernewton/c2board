.PHONY: default
default: all

.PHONY: install
install:
	python2 setup.py install

.PHONY: all
all:
	protoc c2board/src/*.proto --python_out=.

.PHONY: clean
clean:
	rm -rf c2board/src/*_pb2.py*
	rm -rf c2board/*.pyc
