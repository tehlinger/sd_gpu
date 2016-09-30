set output "result.png"
set title "Speedup en fonction du nombre de noeuds"
set terminal pngcairo background rgb 'white' size 900,600
set datafile separator ";"
set ylabel 'Speedup (sans unité)' tc rgb 'black'
set xlabel 'Nb noeuds' tc rgb 'black'
set border lc rgb 'black'
set key tc rgb 'black'
set linetype 1 lc rgb 'black'
#set logscale y 2
plot 	'../data/1p_1th_k1.csv'  using 1:4  with lines title '1 pr. - 1 thread' lt rgb "blue" lw 2,\
	'../data/1p_2th_k1.csv'  using 1:4  with lines title '1 pr. - 2 threads' lt rgb  "green" lw 2
#	'../data/2p_1th_k1_v.csv'  using 1:4  with lines title '2 pr. - 1 threads' lt rgb  "pink" lw 2,\
#	'../data/2p_2th_k1_v.csv'  using 1:4  with lines title '2 pr. - 2 threads' lt rgb  "red" lw 2
