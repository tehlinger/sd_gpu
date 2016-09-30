set output "result.png"
set title "Différence de temps dexécution entre une répartition round robin et une répartition par voisin"
set terminal pngcairo background rgb 'white' size 900,600
set datafile separator ";"
set ylabel 'Temps dexécution (secondes)' tc rgb 'black'
set xlabel 'Nb noeuds' tc rgb 'black'
set border lc rgb 'black'
set key tc rgb 'black'
set linetype 1 lc rgb 'black'
set yrange [0:10]
set xrange [1:16]
#set logscale y 2
plot 	'data/2p_1th_k1_v.csv'  using 1:2  with lines title 'Réparition par voisin' lt rgb "green" lw 2,\
	'data/2p_1th_k1_rr.csv'  using 1:2  with lines title 'Réparition round robin' lt rgb "red" lw 2
