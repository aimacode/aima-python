#!/bin/sh

for p in submissions/*/*vac*[^2].py
do
    mod=`echo $p | sed 's|/|.|g'`
    echo $mod
    echo "import agents; import $mod" | python3
done