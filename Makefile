
.PHONY: all kmeans gmm clean

all: gmm

kmeans:
	$(MAKE) -C kmeans

gmm:
	$(MAKE) -C gmm

clean:
	$(MAKE) -C kmeans clean
	$(MAKE) -C gmm clean
