.PHONY: default
default: all

.PHONY: all
all:
	protoc c2board/src/*.proto --python_out=.
	python2 setup.py develop --user

.PHONY: clean
clean:
	python2 setup.py develop --uninstall --user
	rm -rf c2board/src/*_pb2.py*
