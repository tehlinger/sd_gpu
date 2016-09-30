set output "result.png"
set title "Puissance de calcul (Gigaflops) en fonction du nombre de threads et de processus par noeud"
set terminal pngcairo background rgb 'white' size 900,600
set datafile separator ";"
set ylabel 'Puissance de calcul (Gigaflops)' tc rgb 'black'
set xlabel 'Nb noeuds' tc rgb 'black'
set border lc rgb 'black'
set key tc rgb 'black'
set linetype 1 lc rgb 'black'
#set yrange [1:16]
#set xrange [1:16]
#set logscale y 10
plot 	'data/2p_2th_k1_v.csv'  using 1:2  with lines title '2 Processus 2 thread' lt rgb "red" lw 2,\
	'data/2p_1th_k1_v.csv'   using 1:2  with lines title '2 Processus 1 thread' lt rgb "red" lw 2 dashtype 2,\
	'data/1p_2th_k1.csv'     using 1:2  with lines title '1 Processus 2 thread' lt rgb "blue" lw 2
