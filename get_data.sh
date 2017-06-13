cd data

if [ ! -f r8-all-terms.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-train-all-terms.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-test-all-terms.txt
    cat r8-*-all-terms.txt > r8-all-terms.txt
    rm r8-*-all-terms.txt
fi

if [ ! -f r52-all-terms.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-train-all-terms.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-test-all-terms.txt
    cat r52-*-all-terms.txt > r52-all-terms.txt
    rm r52-*-all-terms.txt
fi

if [ ! -f 20ng-all-terms.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-train-all-terms.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-test-all-terms.txt
    cat 20ng-*-all-terms.txt > 20ng-all-terms.txt
    rm 20ng-*-all-terms.txt
fi


if [ ! -f webkb-stemmed.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/webkb-train-stemmed.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/webkb-test-stemmed.txt
    cat webkb-*-stemmed.txt > webkb-stemmed.txt
    rm webkb-*-stemmed.txt
fi
