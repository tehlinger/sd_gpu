set output "bloc_1D_power.png"

#set title "Puissance de calcul en fonction de la largeur des blocs"

set terminal pngcairo background rgb 'white' size 1250,900 enhanced font 'Verdana,30' 
#set datafile separator ""

set decimalsign ","
set ylabel 'Puissance de calcul (Gigaflops)' tc rgb 'black'
set xlabel 'Largeur du bloc' tc rgb 'black'
set border lc rgb 'black'
set key tc rgb 'black'
set linetype 1 lc rgb 'black'
set logscale y 2
set logscale x 2
#set yrange [3.65:4.80]
plot 	'data/K1_blocs1D.csv'  using 1:3  with lines title "Puissance de calcul" lt rgb "red" lw 3
