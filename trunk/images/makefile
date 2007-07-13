# makefile for images

Sources = dirt.svg vacuum.svg

Targets = $(Sources:.svg=.png)

ImageScale = 50x50

Temporary = tmp.jpg

.PHONY: all

all: $(Targets)

.PHONY: clean

clean:
	rm -f $(Targets) $(Temporary)

%.png: %.svg
	convert -scale $(ImageScale) $< $@

%-icon.jpg: %.svg
	convert -scale $(ImageScale) $< $@

%-icon.jpg: %.jpg
	convert -scale $(ImageScale) $< $@

wall-icon.jpg: b15woods003.jpg
	convert -crop 764x764+0+0 $< tmp.jpg
	convert -resize 50x50+0+0 tmp.jpg $@

vacuum-icon.jpg: vacuum.svg
	convert -scale $(ImageScale) -transparent white $< $@
