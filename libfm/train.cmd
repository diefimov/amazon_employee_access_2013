
name="$1"
iters="$2"

if [ -f "./$name.tr.libsvm" ]; then
  convert -ifile $name.tr.libsvm -ofilex $name.tr.libsvm.x -ofiley $name.tr.libsvm.y
  transpose -ifile $name.tr.libsvm.x -ofile $name.tr.libsvm.xt
  rm $name.tr.libsvm
fi

if [ -f "./$name.test.libsvm" ]; then
  convert -ifile $name.test.libsvm -ofilex $name.test.libsvm.x -ofiley $name.test.libsvm.y
  transpose -ifile $name.test.libsvm.x -ofile $name.test.libsvm.xt
  rm $name.test.libsvm
fi

for it in $(seq 1 $iters); do 
  echo "Iteration $it of $iters"
  libfm -train $name.tr.libsvm -test $name.test.libsvm -out $name.test.pred.$it -init_stdev 0.5 -method mcmc -dim '1,1,12' -task c -iter 1500
done
