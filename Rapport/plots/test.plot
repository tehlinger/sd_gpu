set terminal lua fulldoc \
solid originreset plotsize 9,6.5
set output 'ntest.tex'

set format x "%2.0f"
set format y "%3.2f"

unset key
set label 1 '$\mu$' at 5.5, 0.0193014, 0 left norotate back nopoint offset character 0, 0, 0
set label 2 '$\sigma$' at 7.73607, 0.108213, 0 left norotate back nopoint offset character 0, 0, 0
set arrow 1 from 5, 0, 0 to 5, 0.178412, 0 nohead back nofilled linetype -1 linewidth 1.000
set arrow 2 from 5, 0.108213, 0 to 7.23607, 0.108213, 0 nohead back nofilled linetype -1 linewidth 1.000
set samples 200, 200
set ytics border in scale 1,0.5 mirror norotate  offset character 0, 0, 0 0.00000,0.0193014,0.193014
set title "poisson PDF using normal approximation"
set xlabel '$k, x$ \tikz[baseline]{\draw[yshift=.7ex,->](0,0)--(.7,0);}'
set xrange [ -2.00000 : 15.0000 ] noreverse nowriteback
set ylabel 'probability density \tikz[baseline]{\draw[yshift=.7ex,->](0,0)--(.7,0);}'
set yrange [ 0.00000 : 0.193014 ] noreverse nowriteback

isint(x)=(int(x)==x)

normal(x,mu,sigma)=sigma<=0?1/0:invsqrt2pi/sigma*exp(-0.5*((x-mu)/sigma)**2)
poisson(x,mu)=mu<=0?1/0:!isint(x)?1/0:x<0?0.0:exp(x*log(mu)-lgamma(x+1)-mu)

rnd(x) = floor(x+0.5)
fourinvsqrtpi = 2.25675833419103
invsqrt2pi = 0.398942280401433
twopi = 6.28318530717959
mu = 5.0
sigma = 2.23606797749979

plot poisson(rnd(x), mu) with histeps,\
normal(x, mu, sigma) # title '$\mathrm{normal}(x, \mu, \sigma)$'
