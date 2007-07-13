# makefile for images

Sources = dirt.svg vacuum.svg

Targets = $(Sources:.svg=.png)

%.png: %.svg
	convert $< $@

.PHONY: all

all: $(Targets)

.PHONY: clean

clean:
	rm -f $(Targets)
