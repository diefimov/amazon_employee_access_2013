@echo off
set name=%1
set train_name="%name%.tr.libsvm"
set test_name="%name%.test.libsvm"
set iters=%2
set /a it=1

:loop
if %it%==%iters% GOTO end
echo Iteration %it% of %iters%
libfm -train %train_name% -test %test_name% -out name.test.pred.%it% -init_stdev 0.5 -method mcmc -dim '1,1,12' -task c -iter 1500
set /a it=%it%+1
GOTO loop

:end
echo Done

