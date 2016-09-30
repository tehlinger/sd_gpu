set output "bloc_1D_time.png"
set terminal pngcairo background rgb 'white' size 1250,900 enhanced font 'Verdana,30' 
#set datafile separator ""
set font ',22'
set decimalsign ","
set ylabel 'Temps dexécution (secondes)' tc rgb 'black'
set xlabel 'Largeur du bloc' tc rgb 'black'
set border lc rgb 'black'
set key tc rgb 'black'
set linetype 1 lc rgb 'black'
set logscale y 2
set logscale x 2
#set yrange [3.65:4.80]
plot 	'data/K1_blocs1D.csv'  using 1:2  with lines title "Temps d'exécution" lt rgb "blue" lw 3
