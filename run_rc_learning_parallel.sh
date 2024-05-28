startTime=$(date +%s)

let trials=$1
let parallel=$2
let durchgaenge=$3

# start paradigma
for durchgang in $(seq $durchgaenge); do
	startdurchgangTime=$(date +%s)
        for i in $(seq $parallel); do
                let y=$i+$parallel*$((durchgang - 1))
                python run_rc_learning_force.py $y $trials &
        done
        wait
        sleep 5
	echo $durchgang >> fertig.txt
	endTime=$(date +%s)
	dif=$((endTime - startdurchgangTime))
	echo $dif >> zeit.txt
	sleep 5
done

endTime=$(date +%s)
dif=$((endTime - startTime))
echo $dif >> zeit.txt

