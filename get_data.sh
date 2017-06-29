cd data

if [ ! -f r8-all-terms.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-train-all-terms.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-test-all-terms.txt
    cat r8-*-all-terms.txt > r8-all-terms.txt
    rm r8-*-all-terms.txt
fi

if [ ! -f r8-no-short.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-train-no-short.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-test-no-short.txt
    cat r8-*-no-short.txt > r8-no-short.txt
    rm r8-*-no-short.txt
fi

if [ ! -f r8-no-stop.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-train-no-stop.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-test-no-stop.txt
    cat r8-*-no-stop.txt > r8-no-stop.txt
    rm r8-*-no-stop.txt
fi

if [ ! -f r8-stemmed.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-train-stemmed.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-test-stemmed.txt
    cat r8-*-stemmed.txt > r8-stemmed.txt
    rm r8-*-stemmed.txt
fi

if [ ! -f r52-all-terms.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-train-all-terms.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-test-all-terms.txt
    cat r52-*-all-terms.txt > r52-all-terms.txt
    rm r52-*-all-terms.txt
fi

if [ ! -f r52-no-short.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-train-no-short.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-test-no-short.txt
    cat r52-*-no-short.txt > r52-no-short.txt
    rm r52-*-no-short.txt
fi

if [ ! -f r52-no-stop.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-train-no-stop.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-test-no-stop.txt
    cat r52-*-no-stop.txt > r52-no-stop.txt
    rm r52-*-no-stop.txt
fi

if [ ! -f r52-stemmed.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-train-stemmed.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-test-stemmed.txt
    cat r52-*-stemmed.txt > r52-stemmed.txt
    rm r52-*-stemmed.txt
fi

if [ ! -f 20ng-all-terms.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-train-all-terms.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-test-all-terms.txt
    cat 20ng-*-all-terms.txt > 20ng-all-terms.txt
    rm 20ng-*-all-terms.txt
fi

if [ ! -f 20ng-no-short.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-train-no-short.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-test-no-short.txt
    cat 20ng-*-no-short.txt > 20ng-no-short.txt
    rm 20ng-*-no-short.txt
fi

if [ ! -f 20ng-no-stop.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-train-no-stop.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-test-no-stop.txt
    cat 20ng-*-no-stop.txt > 20ng-no-stop.txt
    rm 20ng-*-no-stop.txt
fi

if [ ! -f 20ng-stemmed.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-train-stemmed.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-test-stemmed.txt
    cat 20ng-*-stemmed.txt > 20ng-stemmed.txt
    rm 20ng-*-stemmed.txt
fi


if [ ! -f webkb-stemmed.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/webkb-train-stemmed.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/webkb-test-stemmed.txt
    cat webkb-*-stemmed.txt > webkb-stemmed.txt
    rm webkb-*-stemmed.txt
fi


if [ ! glove.6B ]; then
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
fi
